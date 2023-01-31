from abc import ABC, abstractmethod
import jinja2
import pandas as pd
from typing import List
import os
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
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame, paths: dict, table_name: str):
        self.original = original
        self.synthetic = synthetic
        self.paths = paths
        self.table_name = table_name

    @abstractmethod
    def report(
        self, cont_columns: List[str], categ_columns: List[str], text_columns: List[str]
    ):
        pass


class AccuracyTest(BaseTest):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame, paths: dict, table_name: str):
        super().__init__(original, synthetic, paths, table_name)

    def __prepare_before_report(self):
        """
        Do preparation work before creating the report
        """
        draws_path = self.paths["draws_path"]
        os.makedirs(draws_path, exist_ok=True)
        acc_draws_path = f"{draws_path}/accuracy"
        os.makedirs(acc_draws_path, exist_ok=True)
        univariate = UnivariateMetric(self.original, self.synthetic, True, acc_draws_path)
        bivariate = BivariateMetric(self.original, self.synthetic, True, acc_draws_path)
        correlations = Correlations(self.original, self.synthetic, True, acc_draws_path)
        clustering = Clustering(self.original, self.synthetic, True, acc_draws_path)
        utility = Utility(self.original, self.synthetic, True, acc_draws_path)
        acc = JensenShannonDistance(self.original, self.synthetic, True, acc_draws_path)
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
        uni_images = {title: transform_to_base64(path) for title, path in uni_images.items()}
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
                               table_name=self.table_name
                               )

        with open(f"{self.paths['draws_path']}/accuracy_report.html", 'w') as f:
            f.write(html)
