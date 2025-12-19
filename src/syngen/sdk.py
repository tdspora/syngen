from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
from typing import Dict, List, Literal, Optional, Set, Union

import pandas as pd
from slugify import slugify
from loguru import logger

from syngen.infer import launch_infer
from syngen.ml.context import get_context, global_context
from syngen.ml.data_loaders import DataEncryptor, DataLoader, MetadataLoader
from syngen.ml.reporters import AccuracyReporter, Report, SampleAccuracyReporter
from syngen.ml.utils import fetch_config, fetch_env_variables, get_reports, setup_log_process
from syngen.ml.validation_schema import ReportTypes, ValidationSchema
from syngen.train import launch_train, validate_required_parameters


class BaseDataIO(ABC):
    """
    Base class for DataIO and AdvancedDataIO to handle common initialization logic
    """
    def __init__(self, path: str, fernet_key: Optional[str] = None, **kwargs):
        self.path = path
        self.fernet_key = fernet_key
        self.format_settings = kwargs
        self.metadata = self._create_metadata()
        self._validate_metadata()
        global_context(metadata=kwargs)
        self._create_data_loader()

    @abstractmethod
    def _create_metadata(self):
        pass

    @abstractmethod
    def _validate_metadata(self):
        pass

    @abstractmethod
    def _create_data_loader(self):
        pass


class DataIO(BaseDataIO):
    """
    SDK class for loading and saving data with optional encryption and format settings.
    """
    def __init__(self, path: str, fernet_key: Optional[str] = None, **kwargs):
        super().__init__(path, fernet_key, **kwargs)

    def _create_data_loader(self):
        self.data_loader = DataLoader(
            path=self.path,
            table_name="table",
            metadata=self.metadata
        )

    def _create_metadata(self) -> Dict:
        """
        Create the metadata dictionary for data loading and saving
        """
        return {
            "table": {
                "train_settings": {
                    "source": self.path
                },
                "encryption": {
                    "fernet_key": (
                        fetch_env_variables({"fernet_key": self.fernet_key}).get("fernet_key")
                    )
                },
                "format": self.format_settings
            }
        }

    def _validate_metadata(self):
        ValidationSchema(
            metadata=self.metadata,
            validation_source=True,
            process="train"
        ).validate_schema()

    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Load data from the specified path with optional parameters
        """
        df, _ = self.data_loader.load_data(**kwargs)
        return df

    def load_schema(self):
        """
        Load the original schema of the data
        """
        return self.data_loader.original_schema

    def save_data(
        self,
        df: pd.DataFrame,
        **kwargs
    ):
        """
        Save data to the specified path with optional parameters
        """
        self.data_loader.save_data(
            df,
            schema=kwargs.get("schema"),
            format=get_context().get_config()
        )


@dataclass
class Syngen:
    """
    SDK class for training, inference, and generation of reports.

    Attributes:
        list_of_tables (List[str]): The list of table names.
        metadata_path (Optional[str]): The path to the metadata file.
        table_name (Optional[str]): The name of the table.
        source (Optional[str]): The source of the data.
        execution_artifacts (Dict): The dictionary to store the information about paths
        to execution artifacts.
        report_types (ReportTypes): The object containing the information
        about available report types.
    """
    list_of_tables: List[str] = field(default_factory=list)
    metadata_path: Optional[str] = None
    table_name: Optional[str] = None
    source: Optional[str] = None
    execution_artifacts: Dict = field(default_factory=dict)
    report_types: object = field(init=False)

    def __post_init__(self):
        validate_required_parameters(
            metadata_path=self.metadata_path,
            source=self.source,
            table_name=self.table_name
        )
        self.list_of_tables = (
            [
                table_name
                for table_name in MetadataLoader(path=self.metadata_path).load_data().keys()
                if table_name != "global"
            ]
            if self.metadata_path is not None
            else [self.table_name]
        )
        self.report_types = ReportTypes()

    def _set_execution_artifacts(
        self,
        type_of_process: str,
        table_name: Optional[str] = None,
        reports: List[str] = list()
    ):
        self.execution_artifacts = dict()

        if type_of_process == "report" and reports:
            self._set_report_artifacts(table_name, reports)
        else:
            self._set_process_artifacts(type_of_process)

    def _set_report_artifacts(self, table_name: str, reports: List[str]):
        """
        Set execution artifacts for report generation
        """
        self.execution_artifacts[table_name] = {"generated_reports": {}}
        for report in reports:
            if report == "sample":
                reports = self._get_train_reports(table_name)
                self.execution_artifacts[table_name]["generated_reports"]["sample_report"] = (
                    reports.get("sample_report")
                )

            if report in self.report_types.full_list_of_infer_report_types:
                reports = self._get_infer_reports(table_name)
                self.execution_artifacts[table_name]["generated_reports"][f"{report}_report"] = (
                    reports.get(f"{report}_report")
                )

    def _set_process_artifacts(self, type_of_process: str):
        """
        Set execution artifacts for 'train' or 'infer' processes
        """
        for table_name in self.list_of_tables:
            if type_of_process == "train":
                self.execution_artifacts[table_name] = self._get_train_artifacts(table_name)
            elif type_of_process == "infer":
                self.execution_artifacts[table_name] = self._get_infer_artifacts(table_name)

    def _get_train_reports(self, table_name: str) -> Dict:
        """
        Get generated reports generated during a training process for a certain table
        """
        train_config = self._get_config(table_name, type_of_process="train")
        return train_config.paths["generated_reports"]

    @staticmethod
    def _get_config(table_name: str, type_of_process: str):
        """
        Get configuration for a certain table and process type
        """
        if type_of_process == "train":
            path_to_config = (
                f"model_artifacts/resources/{slugify(table_name)}/"
                "vae/checkpoints/train_config.pkl"
            )
        elif type_of_process == "infer":
            path_to_config = (
                f"model_artifacts/tmp_store/{slugify(table_name)}/infer_config.pkl"
            )
        else:
            raise ValueError(f"Unsupported process type: {type_of_process}")

        config = fetch_config(config_pickle_path=path_to_config)
        return config

    def _get_infer_reports(self, table_name: str) -> Dict:
        """
        Get the list of generated reports generated during an inference process for a certain table
        """
        infer_config = self._get_config(table_name, type_of_process="infer")
        type_of_process = infer_config.type_of_process
        if type_of_process == "train":
            train_config = self._get_config(table_name, type_of_process="train")
            return train_config.paths["generated_reports"]
        return infer_config.paths["generated_reports"]

    def _get_train_artifacts(self, table_name: str) -> Dict:
        """
        Get execution artifacts for a training process for a certain table
        """
        train_config = self._get_config(table_name, type_of_process="train")
        return {
            "losses_path": train_config.paths["losses_path"],
            "path_to_input_data": train_config.paths["input_data_path"],
            "generated_reports": train_config.paths["generated_reports"],
        }

    def _get_infer_artifacts(self, table_name: str) -> Dict:
        """
        Get execution artifacts for an inference process for a certain table
        """
        infer_config = self._get_config(table_name, type_of_process="infer")
        return {
            "path_to_input_data": infer_config.paths["input_data_path"],
            "path_to_generated_data": infer_config.paths["path_to_merged_infer"],
            "generated_reports": infer_config.paths["generated_reports"],
        }

    def train(
        self,
        epochs: int = 10,
        drop_null: bool = False,
        row_limit: Optional[int] = None,
        reports: Union[str, List[str]] = "none",
        log_level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        batch_size: int = 32,
        fernet_key: Optional[str] = None
    ):
        launch_train(
            metadata_path=self.metadata_path,
            source=self.source,
            table_name=self.table_name,
            epochs=epochs,
            drop_null=drop_null,
            row_limit=row_limit,
            reports=reports,
            log_level=log_level,
            batch_size=batch_size,
            fernet_key=fernet_key
        )
        self._set_execution_artifacts(type_of_process="train")

    def infer(
        self,
        size: int = 100,
        run_parallel: bool = False,
        batch_size: int = None,
        random_seed: Optional[int] = None,
        reports: Union[str, List[str]] = "none",
        log_level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        fernet_key: Optional[str] = None,
    ):
        launch_infer(
            metadata_path=self.metadata_path,
            size=size,
            table_name=self.table_name,
            run_parallel=run_parallel,
            batch_size=batch_size,
            reports=reports,
            random_seed=random_seed,
            log_level=log_level,
            fernet_key=fernet_key
        )
        self._set_execution_artifacts(type_of_process="infer")

    @staticmethod
    def _validate_artifacts(
        table_name: str,
        fernet_key: Optional[str],
        completed_processes: Set[str]
    ):
        errors: List[str] = []

        slug = slugify(table_name)

        path_to_input_data = (
            f"model_artifacts/tmp_store/{slug}/"
            f"input_data_{slug}.{'dat' if fernet_key is not None else 'pkl'}"
        )
        if not os.path.exists(path_to_input_data):
            errors.append(
                (
                    f"The input data file wasn't found for the table '{table_name}' "
                    f"in the path - {path_to_input_data}."
                )
            )

        # Type-specific validations
        path_to_train_config = (
            f"model_artifacts/resources/{slug}/vae/checkpoints/train_config.pkl"
        )
        path_to_train_success_file = f"model_artifacts/resources/{slug}/train_message.success"
        path_to_infer_config = f"model_artifacts/tmp_store/{slug}/infer_config.pkl"
        path_to_infer_success_file = f"model_artifacts/tmp_store/{slug}/infer_message.success"
        checks_of_artifacts = {
            "train": [
                (
                    path_to_train_config,
                    (
                        f"The training configuration wasn't found for the table '{table_name}' "
                        f"in the path - {path_to_train_config}."
                    ),
                ),
                (
                    path_to_train_success_file,
                    (
                        f"The training success file wasn't found for the table '{table_name}' "
                        f"in the path - {path_to_train_success_file}."
                    ),
                ),
            ],
            "infer": [
                (
                    path_to_infer_config,
                    (
                        f"The inference configuration wasn't found for the table '{table_name}' "
                        f"in the path - {path_to_infer_config}"
                    ),
                ),
                (
                    path_to_infer_success_file,
                    (
                        f"The inference success file wasn't found for the table '{table_name}' "
                        f"in the path - {path_to_infer_success_file}."
                    ),
                ),
            ],
        }
        for process in completed_processes:
            for path, message in checks_of_artifacts.get(process, []):
                if not os.path.exists(path):
                    errors.append(message)

        if errors:
            raise FileNotFoundError("\n".join(errors))

    def __get_sample_reporter(
        self,
        table_name: str,
        fernet_key: Optional[str]
    ) -> SampleAccuracyReporter:
        """
        Return a SampleAccuracyReporter instance for the specified table
        """
        train_config = self._get_config(table_name, type_of_process="train")
        train_config.metadata[table_name]["encryption"]["fernet_key"] = fernet_key
        return SampleAccuracyReporter(
            table_name=table_name,
            paths=train_config.paths,
            config=train_config.to_dict(),
            metadata=train_config.metadata,
        )

    def __get_accuracy_reporter(
        self,
        table_name: str,
        report: str,
        fernet_key: Optional[str]
    ) -> AccuracyReporter:
        infer_config = self._get_config(table_name, type_of_process="infer")
        infer_config.reports = [report]
        infer_config.metadata[table_name]["encryption"]["fernet_key"] = fernet_key
        return AccuracyReporter(
            table_name=table_name,
            paths=infer_config.paths,
            config=infer_config.to_dict(),
            metadata=infer_config.metadata,
            loader=None,
            type_of_process=infer_config.type_of_process
        )

    def _register_reporter(self, table_name: str, report: str, fernet_key: Optional[str]) -> None:
        """
        Register a specific type of report
        """
        if report in ["accuracy", "metrics_only"]:
            accuracy_reporter = self.__get_accuracy_reporter(table_name, report, fernet_key)
            Report().register_reporter(table=table_name, reporter=accuracy_reporter)
        elif report == "sample":
            sample_reporter = self.__get_sample_reporter(table_name, fernet_key)
            Report().register_reporter(table=table_name, reporter=sample_reporter)

    def generate_quality_reports(
        self,
        table_name: str,
        reports: Union[List[str], str],
        log_level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        fernet_key: Optional[str] = None
    ) -> None:
        """
        Generate reports for a table using existing artifacts to
        evaluate the quality of the input data or the generated data

        Parameters
        ----------
        table_name: str
            Table name to generate the report for
            (same as used in the training or in the inference processes).
        reports: Union[List[str], str]
            List of report types to generate.
        fernet_key: Optional[str]
            the name of the environment variable kept the value of the Fernet key
            for decrypting the input of the original data, if applicable.
        """
        setup_log_process(
            type_of_process="report",
            log_level=log_level,
            table_name=table_name,
            metadata_path=None,
        )

        reports = get_reports(reports, self.report_types, "train")

        if reports:
            if fernet_key is not None:
                fernet_key = fetch_env_variables({"fernet_key": fernet_key}).get("fernet_key")
                DataEncryptor.validate_fernet_key(fernet_key)
            self._validate_artifacts(
                table_name=table_name,
                fernet_key=fernet_key,
                completed_processes={
                    "infer"
                    if report in self.report_types.infer_report_types
                    else "train"
                    for report in reports
                }
            )

        for report in reports:
            self._register_reporter(table_name, report, fernet_key)

        if reports:
            Report().generate_report()
            Report().clear_report()
        else:
            self.report_types.train_report_types.extend(["none", "all"])
            list_of_available_reports = (
                ", ".join(f"'{rt}'" for rt in self.report_types.train_report_types)
            )
            logger.warning(
                "No reports to generate. Please specify the report type "
                f"from {list_of_available_reports}."
            )

        self._set_execution_artifacts(
            type_of_process="report",
            table_name=table_name,
            reports=reports
        )
