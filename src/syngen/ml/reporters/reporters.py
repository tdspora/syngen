from abc import abstractmethod
from typing import Dict
import itertools
from collections import defaultdict

import numpy as np
from loguru import logger

from syngen.ml.utils import (
    get_nan_labels,
    nan_labels_to_float,
    fetch_dataset,
    datetime_to_timestamp,
)
from syngen.ml.metrics import AccuracyTest, SampleAccuracyTest
from syngen.ml.data_loaders import DataLoader
from syngen.ml.metrics.utils import text_to_continuous
from syngen.ml.mlflow_tracker import MlflowTracker


class Reporter:
    """
    Abstract class for reporters
    """

    def __init__(self, table_name: str, paths: Dict[str, str], config: Dict[str, str]):
        self.table_name = table_name
        self.paths = paths
        self.config = config

    def _extract_report_data(self):
        original, schema = DataLoader(self.paths["original_data_path"]).load_data()
        synthetic, schema = DataLoader(self.paths["path_to_merged_infer"]).load_data()
        return original, synthetic

    def fetch_data_types(self):
        dataset = fetch_dataset(self.paths["dataset_pickle_path"])
        types = (
            dataset.str_columns,
            dataset.date_columns,
            dataset.int_columns,
            dataset.float_columns,
            dataset.binary_columns,
            dataset.categ_columns,
            dataset.long_text_columns,
        )
        return types

    def preprocess_data(self):
        """
        Preprocess original and synthetic data.
        Return original data, synthetic data, float columns, integer columns, categorical columns
        """
        original, synthetic = self._extract_report_data()
        missing_columns = set(original) - set(synthetic)
        for col in missing_columns:
            synthetic[col] = np.nan
        columns_nan_labels = get_nan_labels(original)
        original = nan_labels_to_float(original, columns_nan_labels)
        synthetic = nan_labels_to_float(synthetic, columns_nan_labels)
        types = self.fetch_data_types()
        (
            str_columns,
            date_columns,
            int_columns,
            float_columns,
            binary_columns,
            categ_columns,
            long_text_columns,
        ) = types
        original = original[[col for col in original.columns if col in set().union(*types)]]
        synthetic = synthetic[[col for col in synthetic.columns if col in set().union(*types)]]
        for date_col in date_columns:
            original[date_col] = list(map(lambda d: datetime_to_timestamp(d), original[date_col]))
            synthetic[date_col] = list(
                map(lambda d: datetime_to_timestamp(d), synthetic[date_col])
            )

        int_columns = date_columns | int_columns
        text_columns = str_columns | long_text_columns
        original = text_to_continuous(original, text_columns).drop(text_columns, axis=1)
        synthetic = text_to_continuous(synthetic, text_columns).drop(text_columns, axis=1)

        for col in [i + "_word_count" for i in text_columns]:
            if original[col].nunique() < 50:  # ToDo check if we need this
                categ_columns = categ_columns | {col}
            else:
                int_columns = int_columns | {col}
        int_columns = int_columns | {i + "_char_len" for i in text_columns}

        categ_columns = categ_columns | binary_columns

        for categ_col in categ_columns:
            original[categ_col] = original[categ_col].astype(str)
            synthetic[categ_col] = synthetic[categ_col].astype(str)
        return (
            original,
            synthetic,
            float_columns,
            int_columns,
            categ_columns,
            date_columns,
        )

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

    _reporters: Dict[str, Reporter] = {}

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Report, cls).__new__(cls)
        return cls.instance

    @classmethod
    def register_reporter(cls, table: str, reporter: Reporter):
        """
        Register all needed reporters
        """
        list_of_reporters = cls._reporters.get(table, [])
        list_of_reporters.append(reporter)
        cls._reporters[table] = list_of_reporters

    @classmethod
    def clear_report(cls):
        """
        Delete unnecessary reporters
        """
        cls._reporters.clear()

    @classmethod
    def _group_reporters(cls):
        """
        Group reporters by table name
        """
        list_of_reporters = itertools.chain.from_iterable(cls._reporters.values())
        sorted_reporters = sorted(list_of_reporters, key=lambda r: r.table_name)
        grouped_reporters = defaultdict(list)

        for reporter in sorted_reporters:
            grouped_reporters[reporter.table_name].append(reporter)

        return grouped_reporters

    @classmethod
    def generate_report(cls):
        """
        Generate all needed reports
        """
        grouped_reporters = cls._group_reporters()

        for table_name, reporters in grouped_reporters.items():
            MlflowTracker.reset_status(active_status=True)
            MlflowTracker().start_run(
                run_name=f"{table_name} | INFER",
                tags={"process": "infer", "table_name": table_name}
            )
            for reporter in reporters:
                reporter.report()
                logger.info(
                    f"The {reporter.__class__.report_type} report "
                    f"of the table - '{reporter.table_name}' has been generated"
                )
            MlflowTracker().end_run()

    @property
    def reporters(self) -> Dict[str, Reporter]:
        return self._reporters


class AccuracyReporter(Reporter):
    """
    Reporter for running accuracy test
    """

    report_type = "accuracy"

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
            date_columns,
        ) = self.preprocess_data()
        accuracy_test = AccuracyTest(original, synthetic, self.paths, self.table_name, self.config)
        accuracy_test.report(
            cont_columns=list(float_columns | int_columns),
            categ_columns=list(categ_columns),
            date_columns=list(date_columns),
        )
        logger.info(
            f"Corresponding plot pickle files regarding to accuracy test were saved "
            f"to folder '{self.paths['draws_path']}'."
        )


class SampleAccuracyReporter(Reporter):
    """
    Reporter for running accuracy test
    """

    report_type = "sample"

    def _extract_report_data(self):
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
            date_columns,
        ) = self.preprocess_data()
        accuracy_test = SampleAccuracyTest(
            original, sampled, self.paths, self.table_name, self.config
        )
        accuracy_test.report(
            cont_columns=list(float_columns | int_columns),
            categ_columns=list(categ_columns),
            date_columns=list(date_columns),
        )
        logger.info(
            f"Corresponding plot pickle files regarding to sampled data accuracy test were saved "
            f"to folder {self.paths['draws_path']}."
        )
