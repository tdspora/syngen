import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional
import os

import jinja2
import pandas as pd
import numpy as np
from loguru import logger

from syngen.ml.metrics import (
    UnivariateMetric,
    BivariateMetric,
    JensenShannonDistance,
    Correlations,
    Clustering,
    Utility,
)
from syngen.ml.metrics.utils import transform_to_base64
from syngen.ml.utils import fetch_config, ProgressBarHandler
from syngen.ml.mlflow_tracker import MlflowTracker
from syngen.ml.validation_schema import ReportTypes


class BaseTest(ABC):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        paths: dict,
        table_name: str,
        config: Dict,
    ):
        self.original = original
        self.synthetic = synthetic
        self.paths = paths
        self.table_name = table_name
        self.config = config
        self.plot_exists = any(
            [
                item in ReportTypes().full_list_of_infer_report_types
                for item
                in self.config.get("reports", [])
            ]
        )
        self.reports_path = str()

    @abstractmethod
    def report(
        self,
        cont_columns: List[str],
        categorical_columns: List[str],
        text_columns: List[str],
        date_columns: Optional[List[str]],
    ):
        pass

    def _prepare_dir(self):
        """
        Create the directory where images and reports should be stored
        """
        os.makedirs(self.reports_path, exist_ok=True)

    def _remove_artifacts(self):
        """
        Remove artifacts after creating the reports or fetching the metrics
        """
        shutil.rmtree(self.reports_path)
        base_dir = os.path.dirname(self.reports_path)
        if os.path.exists(base_dir) and not os.listdir(base_dir):
            os.rmdir(base_dir)

    def _log_report_to_mlflow(self, path):
        """
        Log report to mlflow
        """
        try:
            MlflowTracker().log_artifact(path)
        except Exception as error:
            logger.warning(
                f"Logging the report to mlflow has failed due to a permission error. "
                f"File path: '{path}', Error details: {error}.\n"
                f"The report will be saved locally in '{self.reports_path}'"
            )
            pass

    @staticmethod
    def update_progress_bar(message, delta=0):
        ProgressBarHandler().set_progress(
            progress=ProgressBarHandler().progress + delta, delta=None, message=message
        )

    @staticmethod
    def _get_cleaned_config(config):
        """
        Get cleaned config for the report
        """
        filtered_fields = ["reports"]
        cleaned_config = {
            k: v
            for k, v in config.items()
            if k not in filtered_fields
        }

        return cleaned_config


class AccuracyTest(BaseTest):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        paths: dict,
        table_name: str,
        infer_config: Dict
    ):
        super().__init__(original, synthetic, paths, table_name, infer_config)
        self.reports_path = f"{self.paths['reports_path']}/accuracy"
        self.dataset_pickle_path = self.paths["dataset_pickle_path"]
        self.univariate = UnivariateMetric(
            self.original,
            self.synthetic,
            self.plot_exists,
            self.reports_path
        )
        self.bivariate = BivariateMetric(
            self.original,
            self.synthetic,
            self.plot_exists,
            self.reports_path,
            self.dataset_pickle_path
        )
        self.correlations = Correlations(
            self.original,
            self.synthetic,
            self.plot_exists,
            self.reports_path
        )
        self.clustering = Clustering(
            self.original,
            self.synthetic,
            self.plot_exists,
            self.reports_path
        )
        self.utility = Utility(
            self.original,
            self.synthetic,
            self.plot_exists,
            self.reports_path
        )
        self.acc = JensenShannonDistance(
            self.original,
            self.synthetic,
            self.plot_exists,
            self.reports_path
        )
        self._prepare_dir()

    def _fetch_metrics(self, **kwargs):
        """
        Fetch the main metrics
        """
        delta = ProgressBarHandler().delta / 6

        self.update_progress_bar("Generation of the accuracy heatmap...")
        self.acc.calculate_all(kwargs["categorical_columns"])
        acc_median = round(self.acc.calculate_heatmap_median(self.acc.heatmap), 4)
        logger.info(f"Median accuracy is {acc_median}")
        self.update_progress_bar("The accuracy heatmap has been generated", delta)

        uni_images = dict()
        bi_images = dict()

        if self.plot_exists:
            self.update_progress_bar("Generation of the univariate distributions...")
            uni_images = self.univariate.calculate_all(
                kwargs["cont_columns"], kwargs["categorical_columns"], kwargs["date_columns"]
            )
            self.update_progress_bar(
                "The univariate distributions have been generated", delta
            )

            self.update_progress_bar("Generation of the bivariate distributions...")
            bi_images = self.bivariate.calculate_all(
                kwargs["cont_columns"], kwargs["categorical_columns"], kwargs["date_columns"]
            )
            self.update_progress_bar(
                "The bivariate distributions have been generated", delta
            )

        self.update_progress_bar("Generation of the correlations heatmap...")
        corr_result = self.correlations.calculate_all(
            kwargs["categorical_columns"], kwargs["cont_columns"]
        )
        corr_result = round(int(corr_result) if corr_result == 0 else abs(corr_result), 4)
        logger.info(f"Median of differences of correlations is {corr_result}")
        self.update_progress_bar("The correlations heatmap has been generated", delta)

        self.update_progress_bar("Generation of the clustering metric...")
        clustering_result = self.clustering.calculate_all(
            kwargs["categorical_columns"], kwargs["cont_columns"]
        )
        clustering_result = round(clustering_result, 4) if clustering_result is not None else None
        logger.info(f"Mean clusters homogeneity is {clustering_result}")
        self.update_progress_bar("The clustering metric has been calculated", delta)

        self.update_progress_bar("Generation of the utility metric...")
        utility_result = self.utility.calculate_all(
            kwargs["categorical_columns"], kwargs["cont_columns"]
        )
        self.update_progress_bar("The utility metric has been calculated", delta)

        return (
            acc_median,
            corr_result,
            clustering_result,
            utility_result,
            uni_images,
            bi_images,
        )

    def _generate_report(
        self,
        acc_median,
        corr_result,
        clustering_result,
        utility_result,
        uni_images,
        bi_images,
    ):
        """
        Generate the report
        """
        with open(
            f"{os.path.dirname(os.path.realpath(__file__))}/accuracy_report.html"
        ) as file_:
            template = jinja2.Template(file_.read())

        uni_images = {
            title: transform_to_base64(path) for title, path in uni_images.items()
        }
        bi_images = {
            title: transform_to_base64(path) for title, path in bi_images.items()
        }

        train_config = self._get_cleaned_config(
            config=fetch_config(self.paths["train_config_pickle_path"]).to_dict()
        )
        infer_config = self._get_cleaned_config(config=self.config)

        html = template.render(
            accuracy_value=acc_median,
            accuracy_heatmap=transform_to_base64(
                f"{self.reports_path}/accuracy_heatmap.svg"
            ),
            uni_imgs=uni_images,
            correlations_heatmap=transform_to_base64(
                f"{self.reports_path}/correlations_heatmap.svg"
            ),
            correlation_median=corr_result,
            clusters_barplot=transform_to_base64(
                f"{self.reports_path}/clusters_barplot.svg"
            ),
            clustering_value=clustering_result,
            bi_imgs=bi_images,
            utility_barplot=transform_to_base64(
                f"{self.reports_path}/utility_barplot.svg"
            ),
            utility_table=utility_result.to_html() if utility_result is not None else None,
            is_data_available=(
                False
                if utility_result is None or (utility_result is not None and utility_result.empty)
                else True
            ),
            table_name=self.table_name,
            training_config=train_config,
            inference_config=infer_config,
            time=datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            round=round,
        )

        path_to_accuracy_report = f"{self.paths['reports_path']}/accuracy_report.html"
        with open(
            path_to_accuracy_report, "w", encoding="utf-8"
        ) as f:
            f.write(html)
        self._log_report_to_mlflow(path_to_accuracy_report)

    def report(self, *args, **kwargs):
        metrics = self._fetch_metrics(**kwargs)
        (
            acc_median,
            corr_result,
            clustering_result,
            utility_result,
            uni_images,
            bi_images,
        ) = metrics
        MlflowTracker().log_metrics(
            {
                "Utility_avg": (
                    utility_result["Synth to orig ratio"].mean()
                    if utility_result is not None else None
                ),
                "Clustering": (
                    clustering_result
                    if clustering_result is not None else np.NaN
                ),
                "Accuracy": acc_median,
                "Correlation": corr_result,
            }
        )

        if self.plot_exists:
            self._generate_report(
                acc_median,
                corr_result,
                clustering_result,
                utility_result,
                uni_images,
                bi_images,
            )
        self._remove_artifacts()
