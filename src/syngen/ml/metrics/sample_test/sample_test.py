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
        super().__init__(
            original=original,
            synthetic=sampled,
            paths=paths,
            table_name=table_name,
            config=train_config
        )
        self.reports_path = f"{self.paths['reports_path']}/sample_accuracy"

    def __get_univariate_metric(self):
        """
        Do preparation work before creating the report
        """
        self._prepare_dir()
        return UnivariateMetric(
            self.original,
            self.synthetic,
            True,
            self.reports_path
        )

    def __remove_artifacts(self):
        """
        Remove artifacts after creating Sample report
        """
        shutil.rmtree(self.reports_path)

    def report(self, **kwargs):
        univariate = self.__get_univariate_metric()
        uni_images = univariate.calculate_all(
            kwargs["cont_columns"], kwargs["categorical_columns"], kwargs["date_columns"]
        )

        # Generate html report
        with open(
            f"{os.path.dirname(os.path.realpath(__file__))}/sample_report_template.html"
        ) as file_:
            template = jinja2.Template(file_.read(), autoescape=True)

        uni_images = {
            title: transform_to_base64(path)
            for title, path in uni_images.items()
            if "word_count" not in title
        }

        config = self._get_cleaned_config(config=self.config)

        html = template.render(
            uni_imgs=uni_images,
            table_name=self.table_name,
            config=config,
            time=datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
        )
        path, extension = os.path.splitext(self.paths["path_to_sample_report"])
        path_to_sample_report = (
            f"{path}-{datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}{extension}"
        )
        os.makedirs(os.path.dirname(path_to_sample_report), exist_ok=True)
        with open(
            path_to_sample_report,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(html)

        self._update_list_of_generated_reports(path_to_sample_report, "sample")
        self._remove_artifacts()
