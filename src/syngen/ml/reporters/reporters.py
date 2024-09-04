from abc import abstractmethod
from typing import Dict, Optional, Callable
import itertools
from collections import defaultdict

import pandas as pd
import numpy as np
from loguru import logger

from syngen.ml.utils import (
    nan_labels_to_float,
    fetch_config,
    datetime_to_timestamp,
)
from syngen.ml.metrics import AccuracyTest, SampleAccuracyTest
from syngen.ml.data_loaders import DataLoader, DataFrameFetcher
from syngen.ml.metrics.utils import text_to_continuous
from syngen.ml.mlflow_tracker import MlflowTracker
from syngen.ml.utils import ProgressBarHandler


class Reporter:
    """
    Abstract class for reporters
    """

    def __init__(
        self,
        table_name: str,
        paths: Dict[str, str],
        config: Dict[str, str],
        loader: Optional[Callable[[str], pd.DataFrame]] = None
    ):
        self.table_name = table_name
        self.paths = paths
        self.config = config
        self.loader = loader
        self.dataset = None
        self.columns_nan_labels = dict()

    def _extract_report_data(self):
        if self.loader:
            original, schema = DataFrameFetcher(
                loader=self.loader,
                table_name=self.table_name
            ).fetch_data()
        else:
            original, schema = DataLoader(self.paths["original_data_path"]).load_data()
        synthetic, schema = DataLoader(self.paths["path_to_merged_infer"]).load_data()
        return original, synthetic

    def fetch_data_types(self):
        self.dataset = fetch_config(self.paths["dataset_pickle_path"])
        self.columns_nan_labels = self.dataset.nan_labels_dict
        types = (
            self.dataset.str_columns,
            self.dataset.date_columns,
            self.dataset.int_columns,
            self.dataset.float_columns,
            self.dataset.binary_columns,
            self.dataset.categ_columns,
            self.dataset.long_text_columns,
            self.dataset.email_columns,
        )

        # eliminate keys columns from the report
        keys_columns = (
            set(self.dataset.pk_columns) |
            set(self.dataset.fk_columns) |
            set(self.dataset.uq_columns)
        )
        types = tuple(columns - keys_columns for columns in types)

        return types

    @staticmethod
    def convert_to_timestamp(df, col_name, date_format, na_values):
        """
        Convert date column to timestamp
        """
        return [
            datetime_to_timestamp(d, date_format)
            if d not in na_values else np.NaN for d in df[col_name]
        ]

    def preprocess_data(self):
        """
        Preprocess original and synthetic data.
        Return original data, synthetic data, float columns, integer columns, categorical columns
        without keys columns
        """
        types = self.fetch_data_types()
        original, synthetic = self._extract_report_data()
        missing_columns = set(original) - set(synthetic)
        for col in missing_columns:
            synthetic[col] = np.nan
        exclude_columns = self.dataset.uuid_columns
        original = nan_labels_to_float(original, self.columns_nan_labels, exclude_columns)
        synthetic = nan_labels_to_float(synthetic, self.columns_nan_labels, exclude_columns)
        (
            str_columns,
            date_columns,
            int_columns,
            float_columns,
            binary_columns,
            categ_columns,
            long_text_columns,
            email_columns,
        ) = types

        original = original[[col for col in original.columns if col in set().union(*types)]]
        synthetic = synthetic[[col for col in synthetic.columns if col in set().union(*types)]]
        na_values = self.dataset.format.get("na_values", [])
        for date_col, date_format in self.dataset.date_mapping.items():
            original[date_col] = self.convert_to_timestamp(
                original, date_col, date_format, na_values
            )
            synthetic[date_col] = self.convert_to_timestamp(
                synthetic, date_col, date_format, na_values
            )

        int_columns = date_columns | int_columns
        text_columns = str_columns | long_text_columns | email_columns
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
    def generate_report(cls, type_of_process: str):
        """
        Generate all needed reports
        """
        grouped_reporters = cls._group_reporters()

        for table_name, reporters in grouped_reporters.items():
            MlflowTracker().start_run(
                run_name=f"{table_name}-REPORT",
                tags={"table_name": table_name, "process": "report"},
            )
            delta = 0.25 / len(reporters)
            for reporter in reporters:
                message = (f"The calculation of {reporter.__class__.report_type} metrics "
                           f"for the table - '{reporter.table_name}' has started")
                ProgressBarHandler().set_progress(delta=delta, message=message)
                reporter.report()
                if reporter.config["print_report"]:
                    message = (f"The {reporter.__class__.report_type} report of the table - "
                               f"'{reporter.table_name}' has been generated")
                    logger.info(
                        f"The {reporter.__class__.report_type} report of the table - "
                        f"'{reporter.table_name}' has been generated"
                    )
                    ProgressBarHandler().set_progress(
                        progress=ProgressBarHandler().progress + delta,
                        delta=delta,
                        message=message
                    )

                if (
                        not reporter.config["print_report"]
                        and reporter.config.get("get_infer_metrics") is not None
                ):
                    logger.info(
                        f"The metrics for the table - '{reporter.table_name}' have been evaluated"
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
        accuracy_test = AccuracyTest(
            original,
            synthetic,
            self.paths,
            self.table_name,
            self.config,
        )
        accuracy_test.report(
            cont_columns=list(float_columns | int_columns),
            categ_columns=list(categ_columns),
            date_columns=list(date_columns),
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
            original,
            sampled,
            self.paths,
            self.table_name,
            self.config,
        )
        accuracy_test.report(
            cont_columns=list(float_columns | int_columns),
            categ_columns=list(categ_columns),
            date_columns=list(date_columns),
        )
