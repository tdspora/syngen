import os
from typing import Dict
from syngen.ml.metrics import UnivariateMetric, BaseTest
from datetime import datetime
import shutil

import jinja2
import pandas as pd

from syngen.ml.metrics.utils import transform_to_base64


class SampleAccuracyTest(BaseTest):
    def __init__(
        self,
        original: pd.DataFrame,
        sampled: pd.DataFrame,
        paths: dict,
        table_name: str,
        train_config: Dict,
    ):
        super().__init__(original, sampled, paths, table_name, train_config)
        self.draws_path = f"{self.paths['draws_path']}/sample_accuracy"

    def __get_univariate_metric(self):
        """
        Do preparation work before creating the report
        """
        self._prepare_dir()
        return UnivariateMetric(self.original, self.synthetic, True, self.draws_path)

    def __remove_artifacts(self):
        """
        Remove artifacts after creating Sample report
        """
        shutil.rmtree(f"{self.paths['draws_path']}/sample_accuracy")

    def report(self, **kwargs):
        univariate = self.__get_univariate_metric()
        uni_images = univariate.calculate_all(
            kwargs["cont_columns"], kwargs["categ_columns"], kwargs["date_columns"]
        )

        # Generate html report
        with open(
            f"{os.path.dirname(os.path.realpath(__file__))}/sample_report_template.html"
        ) as file_:
            template = jinja2.Template(file_.read())

        uni_images = {
            title: transform_to_base64(path)
            for title, path in uni_images.items()
            if "word_count" not in title
        }

        html = template.render(
            uni_imgs=uni_images,
            table_name=self.table_name,
            config=self.config,
            time=datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
        )

        with open(
            f"{self.paths['draws_path']}/sample_accuracy_report.html",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(html)

        self._remove_artifacts()
