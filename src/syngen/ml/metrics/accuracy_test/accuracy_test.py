from abc import ABC, abstractmethod
import jinja2
import base64
import pandas as pd
from typing import List
import os
from syngen.ml.metrics import (
    UnivariateMetric,
    BivariateMetric,
    JensenShannonDistance
)


class BaseTest(ABC):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame, paths: dict):
        self.original = original
        self.synthetic = synthetic
        self.paths = paths

    @abstractmethod
    def report(
        self, cont_columns: List[str], categ_columns: List[str], text_columns: List[str]
    ):
        pass


class AccuracyTest(BaseTest):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame, paths: dict):
        super().__init__(original, synthetic, paths)

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
        acc = JensenShannonDistance(self.original, self.synthetic, True, acc_draws_path)
        return univariate, bivariate, acc

    def __transform_to_base64(self, path):
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return "data:image/gif;base64," + encoded_string.decode('utf-8')

    def report(self, **kwargs):
        univariate, bivariate, acc = self.__prepare_before_report()
        acc.calculate_all(kwargs["categ_columns"])
        acc_median = "%.5f" % acc.calculate_heatmap_median(acc.heatmap)

        univariate.calculate_all(kwargs["cont_columns"], kwargs["categ_columns"])
        bivariate.calculate_all(kwargs["cont_columns"], kwargs["categ_columns"])

        # Generate html report
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/accuracy_report_template.html") as file_:
            template = jinja2.Template(file_.read())

        draws_acc_path = f"{self.paths['draws_path']}/accuracy"
        uni_images = [self.__transform_to_base64(f"{draws_acc_path}/{f}") for f in os.listdir(draws_acc_path)
                      if f.startswith("univariate")]
        bi_images = [self.__transform_to_base64(f"{draws_acc_path}/{f}") for f in os.listdir(draws_acc_path)
                     if f.startswith("bivariate")]
        html = template.render(accuracy_value=acc_median,
                               accuracy_heatmap=self.__transform_to_base64(f"{draws_acc_path}/accuracy_heatmap.png"),
                               draws_acc_path=draws_acc_path,
                               uni_imgs=uni_images,
                               bi_imgs=bi_images
                               )

        with open(f"{self.paths['draws_path']}/accuracy_report.html", 'w') as f:
            f.write(html)
