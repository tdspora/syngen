from abc import abstractmethod
from typing import List
import pandas as pd
from loguru import logger

from syngen.ml.pipeline import (
    data_pipeline,
    get_nan_labels,
    nan_labels_to_float
)
from syngen.ml.metrics import AccuracyTest
from syngen.ml.metrics.utils import text_to_continuous


class Reporter:
    """
    Abstract class for reporters
    """
    def __init__(self, metadata: dict, paths: dict):
        super().__init__()
        self.table_name = metadata["table_name"]
        self.paths = paths

    def preprocess_data(self):
        """
        Preprocess original and synthetic data.
        Return original data, synthetic data, float columns, integer columns, categorical columns
        """
        original = pd.read_csv(self.paths["original_data_path"])
        synthetic = pd.read_csv(self.paths["synthetic_data_path"])
        columns_nan_labels = get_nan_labels(original)
        original = nan_labels_to_float(original, columns_nan_labels)
        synthetic = nan_labels_to_float(synthetic, columns_nan_labels)
        types = data_pipeline(original)
        str_columns, float_columns, categ_columns, date_columns, int_columns, binary_columns = types
        for date_col in date_columns:
            original[date_col] = list(
                map(lambda d: pd.Timestamp(d).value, original[date_col])
            )
            synthetic[date_col] = list(
                map(lambda d: pd.Timestamp(d).value, synthetic[date_col])
            )

        int_columns = date_columns | int_columns
        original = text_to_continuous(original, str_columns).drop(str_columns, axis=1)
        synthetic = text_to_continuous(synthetic, str_columns).drop(str_columns, axis=1)

        for col in [i + "_word_count" for i in str_columns]:
            if original[col].nunique() < 50:  # ToDo check if we need this
                categ_columns = categ_columns | {col}
            else:
                int_columns = int_columns | {col}
        int_columns = int_columns | {i + "_char_len" for i in str_columns}
        categ_columns = categ_columns | binary_columns
        
        for categ_col in categ_columns:
            original[categ_col] = original[categ_col].astype(str)
            synthetic[categ_col] = synthetic[categ_col].astype(str)
            
        return original, synthetic, float_columns, int_columns, categ_columns

    @abstractmethod
    def report(self, **kwargs):
        """
        Generate the report for certain test
        """
        pass


class Report:
    """
    Singleton metaclass for registration all needed reporters
    """

    __reporters: List[Reporter] = []

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Report, cls).__new__(cls)
        return cls.instance

    @classmethod
    def register_reporter(cls, reporter: Reporter):
        """
        Register all needed reporters
        """
        cls.__reporters.append(reporter)

    @classmethod
    def generate_report(cls):
        """
        Generate all needed reports
        """
        for reporter in cls.__reporters:
            reporter.report()


class AccuracyReporter(Reporter):
    """
    Reporter for running accuracy test
    """

    def report(self):
        """
        Run the report
        """
        (
            original,
            synthetic,
            float_columns,
            int_columns,
            categ_columns,
        ) = self.preprocess_data()
        accuracy_test = AccuracyTest(original, synthetic, self.paths)
        accuracy_test.report(
            cont_columns=list(float_columns | int_columns),
            categ_columns=list(categ_columns)
        )
        logger.info(
            f"Corresponding plot pickle files regarding to accuracy test were saved "
            f"to folder 'model_artifacts/tmp_store/{self.table_name}/draws/'."
        )
