import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional
import os

import jinja2
import pandas as pd
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
from syngen.ml.utils import fetch_training_config, ProgressBarHandler
from syngen.ml.mlflow_tracker import MlflowTracker


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
        self.plot_exists = self.config["print_report"]
        self.draws_path = str()

    @abstractmethod
    def report(
        self,
        cont_columns: List[str],
        categ_columns: List[str],
        text_columns: List[str],
        date_columns: Optional[List[str]],
    ):
        pass

    def _prepare_dir(self):
        """
        Create the directory where images and reports should be stored
        """
        os.makedirs(self.paths["draws_path"], exist_ok=True)
        os.makedirs(self.draws_path, exist_ok=True)

    def _remove_artifacts(self):
        """
        Remove artifacts after creating Accuracy report
        """
        shutil.rmtree(self.draws_path)

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
                f"The report will be saved locally in '{self.paths['draws_path']}'"
            )
            pass

    @staticmethod
    def update_progress_bar(message, delta=0):
        ProgressBarHandler().set_progress(
            progress=ProgressBarHandler().progress + delta, delta=None, message=message
        )

    def _get_cleaned_configs(self):
        """
        Get cleaned configs for the report
        """
        train_config = {
            k: v
            for k, v in fetch_training_config(self.paths["train_config_pickle_path"])
            .to_dict()
            .items()
            if k != "print_report"
        }
        infer_config = {
            k: v
            for k, v in self.config.items()
            if k not in ["print_report", "get_infer_metrics"]
        }
        return infer_config, train_config


class AccuracyTest(BaseTest):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        paths: dict,
        table_name: str,
        infer_config: Dict,
    ):
        super().__init__(original, synthetic, paths, table_name, infer_config)
        self.draws_path = f"{self.paths['draws_path']}/accuracy"
        self.univariate = UnivariateMetric(
            self.original, self.synthetic, self.plot_exists, self.draws_path
        )
        self.bivariate = BivariateMetric(
            self.original, self.synthetic, self.plot_exists, self.draws_path
        )
        self.correlations = Correlations(
            self.original, self.synthetic, self.plot_exists, self.draws_path
        )
        self.clustering = Clustering(
            self.original, self.synthetic, self.plot_exists, self.draws_path
        )
        self.utility = Utility(
            self.original, self.synthetic, self.plot_exists, self.draws_path
        )
        self.acc = JensenShannonDistance(
            self.original, self.synthetic, self.plot_exists, self.draws_path
        )
        self._prepare_dir()

    def _fetch_metrics(self, **kwargs):
        """
        Fetch the main metrics
        """
        delta = ProgressBarHandler().delta / 6

        self.update_progress_bar("Generation of the accuracy heatmap...")
        self.acc.calculate_all(kwargs["categ_columns"])
        acc_median = "%.4f" % self.acc.calculate_heatmap_median(self.acc.heatmap)
        logger.info(f"Median accuracy is {acc_median}")
        self.update_progress_bar("The accuracy heatmap has been generated", delta)

        uni_images = dict()
        bi_images = dict()

        if self.plot_exists:
            self.update_progress_bar("Generation of the univariate distributions...")
            uni_images = self.univariate.calculate_all(
                kwargs["cont_columns"], kwargs["categ_columns"], kwargs["date_columns"]
            )
            self.update_progress_bar(
                "The univariate distributions have been generated", delta
            )

            self.update_progress_bar("Generation of the bivariate distributions...")
            bi_images = self.bivariate.calculate_all(
                kwargs["cont_columns"], kwargs["categ_columns"], kwargs["date_columns"]
            )
            self.update_progress_bar(
                "The bivariate distributions have been generated", delta
            )

        self.update_progress_bar("Generation of the correlations heatmap...")
        corr_result = self.correlations.calculate_all(
            kwargs["categ_columns"], kwargs["cont_columns"]
        )
        corr_result = int(corr_result) if corr_result == 0 else abs(corr_result)
        logger.info(f"Median of differences of correlations is {round(corr_result, 4)}")
        self.update_progress_bar("The correlations heatmap has been generated", delta)

        self.update_progress_bar("Generation of the clustering metric...")
        clustering_result = "%.4f" % self.clustering.calculate_all(
            kwargs["categ_columns"], kwargs["cont_columns"]
        )
        logger.info(f"Median clusters homogeneity is {clustering_result}")
        self.update_progress_bar("The clustering metric has been calculated", delta)

        self.update_progress_bar("Generation of the utility metric...")
        utility_result = self.utility.calculate_all(
            kwargs["categ_columns"], kwargs["cont_columns"]
        )
        logger.info(f"Median clusters homogeneity is {clustering_result}")
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

        draws_acc_path = f"{self.paths['draws_path']}/accuracy"
        uni_images = {
            title: transform_to_base64(path) for title, path in uni_images.items()
        }
        bi_images = {
            title: transform_to_base64(path) for title, path in bi_images.items()
        }

        train_config, infer_config = self._get_cleaned_configs()

        html = template.render(
            accuracy_value=acc_median,
            accuracy_heatmap=transform_to_base64(
                f"{draws_acc_path}/accuracy_heatmap.svg"
            ),
            uni_imgs=uni_images,
            correlations_heatmap=transform_to_base64(
                f"{draws_acc_path}/correlations_heatmap.svg"
            ),
            correlation_median=corr_result,
            clusters_barplot=transform_to_base64(
                f"{draws_acc_path}/clusters_barplot.svg"
            ),
            clustering_value=clustering_result,
            bi_imgs=bi_images,
            utility_barplot=transform_to_base64(
                f"{draws_acc_path}/utility_barplot.svg"
            ),
            utility_table=utility_result.to_html(),
            is_data_available=False if utility_result.empty else True,
            table_name=self.table_name,
            training_config=train_config,
            inference_config=infer_config,
            time=datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            round=round,
        )

        with open(
            f"{self.paths['draws_path']}/accuracy_report.html", "w", encoding="utf-8"
        ) as f:
            f.write(html)
        self._log_report_to_mlflow(f"{self.paths['draws_path']}/accuracy_report.html")
        self._remove_artifacts()

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
                "Utility_avg": utility_result["Synth to orig ratio"].mean(),
                "Clustering": float(clustering_result),
                "Accuracy": float(acc_median),
                "Correlation": round(corr_result, 4),
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
