from abc import abstractmethod
from typing import List, Dict

import pandas as pd
import numpy as np
from loguru import logger

from syngen.ml.pipeline import (
    get_nan_labels,
    nan_labels_to_float
)
from syngen.ml.metrics import AccuracyTest, SampleAccuracyTest
from syngen.ml.metrics.utils import text_to_continuous
from syngen.ml.data_loaders import DataLoader
from syngen.ml.pipeline import fetch_dataset


class Reporter:
    """
    Abstract class for reporters
    """

    def __init__(self, metadata: Dict[str, str], paths: Dict[str, str]):
        self.metadata = metadata
        self.table_name = metadata["table_name"]
        self.paths = paths

    def extract_report_data(self):
        original, schema = DataLoader(self.paths["original_data_path"]).load_data()
        synthetic, schema = DataLoader(self.paths["synthetic_data_path"]).load_data()
        return original, synthetic

    @staticmethod
    def convert_data_types(
            df: pd.DataFrame, binary_columns: List, str_columns: List, date_columns: List,
            int_columns: List, float_columns: List, categ_columns: List):
        """
        Synchronize identified data types in data pipeline mechanism with data types in columns of dataframe
        :param df: dataframe
        :param binary_columns: list of columns with data type - "binary"
        :param str_columns: list of columns with data type - "string"
        :param date_columns: list of columns identified as date
        :param int_columns: list of columns with data type - "integer"
        :param float_columns: list of columns with data type - "float"
        :param categ_columns: list of categorical columns
        :return:
        """
        for column in df.columns:
            if column in [*binary_columns, *str_columns, *date_columns, *categ_columns]:
                df[column] = df[column].astype("object")
            elif column in int_columns:
                df[column] = df[column].astype("int")
            elif column in float_columns:
                df[column] = df[column].astype("float")
        return df

    def preprocess_data(self):
        """
        Preprocess original and synthetic data.
        Return original data, synthetic data, float columns, integer columns, categorical columns
        """
        original, synthetic = self.extract_report_data()
        missing_columns = set(original) - set(synthetic)
        for col in missing_columns:
            synthetic[col] = np.nan
        columns_nan_labels = get_nan_labels(original)
        original = nan_labels_to_float(original, columns_nan_labels)
        synthetic = nan_labels_to_float(synthetic, columns_nan_labels)
        dataset = fetch_dataset(self.paths["dataset_pickle_path"])
        types = (
            dataset.str_columns, dataset.date_columns,
            dataset.int_columns, dataset.float_columns,
            dataset.binary_columns, dataset.categ_columns
        )
        str_columns, date_columns, int_columns, float_columns, binary_columns, categ_columns = types
        for date_col in date_columns:
            original[date_col] = list(
                map(lambda d: pd.Timestamp(d).value, original[date_col])
            )
            synthetic[date_col] = list(
                map(lambda d: pd.Timestamp(d).value, synthetic[date_col])
            )

        int_columns = date_columns | int_columns

        original = self.convert_data_types(
            original, binary_columns, str_columns, date_columns,
            int_columns, float_columns, categ_columns
        )

        synthetic = self.convert_data_types(
            synthetic, binary_columns, str_columns, date_columns,
            int_columns, float_columns, categ_columns
        )
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


class SampleAccuracyReporter(Reporter):
    """
    Reporter for running accuracy test
    """

    def extract_report_data(self):
        original, schema = DataLoader(self.paths["source_path"]).load_data()
        sampled, schema = DataLoader(self.paths["input_data_path"]).load_data()
        return original, sampled

    def report(self):
        """
        Run the report
        """
        (
            original,
            sampled,
            float_columns,
            int_columns,
            categ_columns,
        ) = self.preprocess_data()
        accuracy_test = SampleAccuracyTest(original, sampled, self.paths)
        accuracy_test.report(
            cont_columns=list(float_columns | int_columns),
            categ_columns=list(categ_columns)
        )
        logger.info(
            f"Corresponding plot pickle files regarding to sampled data accuracy test were saved "
            f"to folder 'model_artifacts/tmp_store/{self.table_name}/draws/'."
        )
