from abc import ABC, abstractmethod
import os
import pandas as pd
from typing import List

from syngen.ml.metrics import (
    UnivariateMetric,
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
        univariate = UnivariateMetric(self.original, self.synthetic, True, draws_path)
        acc = JensenShannonDistance(self.original, self.synthetic, True, draws_path)
        return univariate, acc

    def report(self, **kwargs):
        univariate, acc = self.__prepare_before_report()
        univariate.calculate_all(kwargs["cont_columns"], kwargs["categ_columns"])
        acc.calculate_all(kwargs["categ_columns"])
        acc.calculate_heatmap_median(acc.heatmap)

        # sns.heatmap(acc.heatmap, xticklabels=self.original.columns, yticklabels=self.original.columns, annot=True)
