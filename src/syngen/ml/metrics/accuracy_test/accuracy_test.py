import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional
import os

import jinja2
import pandas as pd

from syngen.ml.metrics import (
    UnivariateMetric,
    BivariateMetric,
    JensenShannonDistance,
    Correlations,
    Clustering,
    Utility
)
from syngen.ml.metrics.utils import transform_to_base64
from syngen.ml.utils import fetch_training_config
from syngen.ml.mlflow.mlflow_tracker import MlflowTracker


class BaseTest(ABC):
    def __init__(
            self,
            original: pd.DataFrame,
            synthetic: pd.DataFrame,
            paths: dict,
            table_name: str,
            config: Dict
    ):
        self.original = original
        self.synthetic = synthetic
        self.paths = paths
        self.table_name = table_name
        self.config = config
        self.draws_path = str()

    @abstractmethod
    def report(
        self,
        cont_columns: List[str],
        categ_columns: List[str],
        text_columns: List[str],
        date_columns: Optional[List[str]]
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
        self.draws_path = f"{self.paths['draws_path']}/accuracy"

    def __prepare_before_report(self):
        """
        Do preparation work before creating the report
        """
        self._prepare_dir()
        univariate = UnivariateMetric(self.original, self.synthetic, True, self.draws_path)
        bivariate = BivariateMetric(self.original, self.synthetic, True, self.draws_path)
        correlations = Correlations(self.original, self.synthetic, True, self.draws_path)
        clustering = Clustering(self.original, self.synthetic, True, self.draws_path)
        utility = Utility(self.original, self.synthetic, True, self.draws_path)
        acc = JensenShannonDistance(self.original, self.synthetic, True, self.draws_path)
        return univariate, bivariate, correlations, clustering, utility, acc

    def log_report_to_mlflow(self, path):
        try:
            self.tracker.log_artifact(path)
        except PermissionError:
            pass

    def track_metrics(self, utility_result, clustering_result, acc_median, corr_result):
        self.tracker = MlflowTracker()
        self.tracker.start_run(
            run_name=f"{self.table_name} | INFER",
            tags={"process": "infer", "table_name": self.table_name}
        )
        self.tracker.log_metrics(
            {
                "Utility_avg": utility_result["Synth to orig ratio"].mean(),
                "Clustering": float(clustering_result),
                "Accuracy": float(acc_median),
                "Correlation": corr_result
            }
        )

    def report(self, **kwargs):
        univariate, bivariate, correlations, clustering, utility, acc = self.__prepare_before_report()
        acc.calculate_all(kwargs["categ_columns"])
        acc_median = "%.4f" % acc.calculate_heatmap_median(acc.heatmap)

        uni_images = univariate.calculate_all(kwargs["cont_columns"], kwargs["categ_columns"], kwargs["date_columns"])
        bi_images = bivariate.calculate_all(kwargs["cont_columns"], kwargs["categ_columns"], kwargs["date_columns"])
        corr_result = correlations.calculate_all(kwargs["categ_columns"], kwargs["cont_columns"])
        clustering_result = "%.4f" % clustering.calculate_all(kwargs["categ_columns"], kwargs["cont_columns"])
        utility_result = utility.calculate_all(kwargs["categ_columns"], kwargs["cont_columns"])

        self.track_metrics(utility_result, clustering_result, acc_median, corr_result)

        # Generate html report
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/accuracy_report.html") as file_:
            template = jinja2.Template(file_.read())

        draws_acc_path = f"{self.paths['draws_path']}/accuracy"
        uni_images = {
            title: transform_to_base64(path) for title, path in uni_images.items()
        }
        bi_images = {
            title: transform_to_base64(path) for title, path in bi_images.items()
        }
        html = template.render(accuracy_value=acc_median,
                               accuracy_heatmap=transform_to_base64(f"{draws_acc_path}/accuracy_heatmap.svg"),
                               uni_imgs=uni_images,
                               correlations_heatmap=transform_to_base64(f"{draws_acc_path}/correlations_heatmap.svg"),
                               clusters_barplot=transform_to_base64(f"{draws_acc_path}/clusters_barplot.svg"),
                               clustering_value=clustering_result,
                               bi_imgs=bi_images,
                               utility_barplot=transform_to_base64(f"{draws_acc_path}/utility_barplot.svg"),
                               utility_table=utility_result.to_html(),
                               is_data_available=False if utility_result.empty else True,
                               table_name=self.table_name,
                               training_config=fetch_training_config(self.paths["train_config_pickle_path"]).to_dict(),
                               inference_config=self.config,
                               time=datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                               )

        with open(f"{self.paths['draws_path']}/accuracy_report.html", 'w', encoding="utf-8") as f:
            f.write(html)
        self.log_report_to_mlflow(f"{self.paths['draws_path']}/accuracy_report.html")
        self.tracker.end_run()
        self._remove_artifacts()
