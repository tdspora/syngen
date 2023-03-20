import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict
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
        self, cont_columns: List[str], categ_columns: List[str], text_columns: List[str]
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
            config: Dict
    ):
        super().__init__(original, synthetic, paths, table_name, config)
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

    def report(self, **kwargs):
        univariate, bivariate, correlations, clustering, utility, acc = self.__prepare_before_report()
        acc.calculate_all(kwargs["categ_columns"])
        acc_median = "%.4f" % acc.calculate_heatmap_median(acc.heatmap)

        uni_images = univariate.calculate_all(kwargs["cont_columns"], kwargs["categ_columns"])
        bi_images = bivariate.calculate_all(kwargs["cont_columns"], kwargs["categ_columns"])
        correlations.calculate_all(kwargs["categ_columns"], kwargs["cont_columns"])
        clustering_result = "%.4f" % clustering.calculate_all(kwargs["categ_columns"], kwargs["cont_columns"])
        utility_result = utility.calculate_all(kwargs["categ_columns"], kwargs["cont_columns"])

        # Generate html report
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/accuracy_report.html") as file_:
            template = jinja2.Template(file_.read())

        draws_acc_path = f"{self.paths['draws_path']}/accuracy"
        uni_images = {
            title: transform_to_base64(path) for title, path in uni_images.items()
            if "word_count" not in title
        }
        bi_images = {
            title: transform_to_base64(path) for title, path in bi_images.items()
            if "char_len" not in title and "word_count" not in title
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
                               table_name=self.table_name,
                               config=self.config,
                               time=datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                               )

        with open(f"{self.paths['draws_path']}/accuracy_report.html", 'w', encoding="utf-8") as f:
            f.write(html)

        self._remove_artifacts()
