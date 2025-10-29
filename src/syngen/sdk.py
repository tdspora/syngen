from typing import Optional, Dict, List, Union, Set
import os
import pandas as pd
from loguru import logger
from dataclasses import dataclass, field

from slugify import slugify
from syngen.ml.data_loaders import DataLoader, DataEncryptor, MetadataLoader
from syngen.train import launch_train
from syngen.infer import launch_infer
from syngen.ml.utils import fetch_config, fetch_env_variables
from syngen.ml.reporters import (
    Report,
    AccuracyReporter,
    SampleAccuracyReporter,
)
from syngen.ml.validation_schema import ValidationSchema, ReportTypes
from syngen.ml.context import global_context, get_context
from syngen.ml.utils import get_reports


class DataIO:
    """
    SDK class for loading and saving data with optional encryption and format settings.
    """
    def __init__(self, path: str, fernet_key: Optional[str] = None, **kwargs):
        self.path = path
        self.fernet_key = fernet_key
        self.metadata = self._create_metadata(**kwargs)
        self._validate_metadata(self.metadata)
        global_context(metadata=kwargs)
        self.data_loader = DataLoader(
            path=self.path,
            table_name="table",
            metadata=self.metadata
        )

    def _create_metadata(self, **kwargs) -> Dict:
        """
        Create metadata dictionary for data loading and saving
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
                "format": kwargs
            }
        }

    @staticmethod
    def _validate_metadata(metadata: Dict):
        ValidationSchema(
            metadata=metadata,
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
    SDK class for training, inference, and the generation of reports
    """
    list_of_tables: List = field(default_factory=list)
    metadata_path: Optional[str] = None
    table_name: Optional[str] = None
    source: Optional[str] = None
    execution_artifacts: dict = None

    def __post_init__(self):
        self.list_of_tables = (
            [self.table_name]
            if self.table_name is not None
            else [
                table_name
                for table_name in MetadataLoader(path=self.metadata_path).load_data().keys()
                if table_name != "global"
            ]
        )

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

        if "sample" in reports:
            reports = self._get_reports_from_train_config(table_name)
            self.execution_artifacts[table_name]["generated_reports"]["sample_report"] = (
                reports.get("sample_report")
            )

        if "accuracy" in reports:
            reports = self._get_reports_from_infer_config(table_name)
            self.execution_artifacts[table_name]["generated_reports"]["accuracy_report"] = (
                reports.get("accuracy_report")
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

    @staticmethod
    def _get_reports_from_train_config(table_name: str) -> Dict:
        """
        Get generated reports fetched from the training configuration
        """
        path_to_train_config = (
            f"model_artifacts/resources/{slugify(table_name)}/"
            "vae/checkpoints/train_config.pkl"
        )
        train_config = fetch_config(config_pickle_path=path_to_train_config)
        return train_config.paths["generated_reports"]

    @staticmethod
    def _get_reports_from_infer_config(table_name: str) -> Dict:
        """
        Get generated reports fetched from the inference configuration
        """
        path_to_infer_config = f"model_artifacts/tmp_store/{slugify(table_name)}/infer_config.pkl"
        infer_config = fetch_config(config_pickle_path=path_to_infer_config)
        return infer_config.paths["generated_reports"]

    @staticmethod
    def _get_train_artifacts(table_name: str) -> Dict:
        """
        Get execution artifacts for a training process for a certain table
        """
        path_to_train_config = (
            f"model_artifacts/resources/{slugify(table_name)}/"
            "vae/checkpoints/train_config.pkl"
        )
        train_config = fetch_config(config_pickle_path=path_to_train_config)
        return {
            "losses_path": train_config.paths["losses_path"],
            "path_to_input_data": train_config.paths["input_data_path"],
            "generated_reports": train_config.paths["generated_reports"],
        }

    @staticmethod
    def _get_infer_artifacts(table_name: str) -> Dict:
        """
        Get execution artifacts for an inference process for a certain table
        """
        path_to_infer_config = (
            f"model_artifacts/tmp_store/{slugify(table_name)}/infer_config.pkl"
        )
        infer_config = fetch_config(config_pickle_path=path_to_infer_config)
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
        log_level: str = "INFO",
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
        log_level: str = "INFO",
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
        reports: Set[str]
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
            "sample": [
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
            "accuracy": [
                (
                    path_to_infer_config,
                    (
                        f"The inference configuration wasn't found for table '{table_name}' "
                        f"in the path - {path_to_infer_config}"
                    ),
                ),
                (
                    path_to_infer_success_file,
                    (
                        f"The inference success file wasn't found for table '{table_name}' "
                        f"in the path - {path_to_infer_success_file}."
                    ),
                ),
            ],
        }
        for report in reports:
            for path, message in checks_of_artifacts.get(report, []):
                if not os.path.exists(path):
                    errors.append(message)

        if errors:
            raise FileNotFoundError("\n".join(errors))

    @staticmethod
    def _get_sample_reporter(table_name: str, fernet_key: Optional[str]) -> SampleAccuracyReporter:
        """
        Return a SampleAccuracyReporter instance for the specified table
        """
        path_to_train_config = (
            f"model_artifacts/resources/{slugify(table_name)}/vae/checkpoints/train_config.pkl"
        )
        train_config = fetch_config(config_pickle_path=path_to_train_config)
        train_config.paths["generated_reports"] = {}
        train_config.metadata[table_name]["encryption"]["fernet_key"] = fernet_key
        return SampleAccuracyReporter(
            table_name=table_name,
            paths=train_config.paths,
            config=train_config.to_dict(),
            metadata=train_config.metadata,
        )

    @staticmethod
    def _get_accuracy_reporter(
        table_name: str,
        report: str,
        fernet_key: Optional[str]
    ) -> AccuracyReporter:
        path_to_infer_config = f"model_artifacts/tmp_store/{slugify(table_name)}/infer_config.pkl"
        infer_config = fetch_config(config_pickle_path=path_to_infer_config)
        infer_config.reports = [report]
        infer_config.paths["generated_reports"] = {}
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
            accuracy_reporter = self._get_accuracy_reporter(table_name, report, fernet_key)
            Report().register_reporter(table=table_name, reporter=accuracy_reporter)
        elif report == "sample":
            sample_reporter = self._get_sample_reporter(table_name, fernet_key)
            Report().register_reporter(table=table_name, reporter=sample_reporter)

    def generate_reports(
        self,
        table_name: str,
        reports: Union[List[str], str],
        fernet_key: Optional[str] = None
    ) -> None:
        """
        Generate reports for a table using existing artifacts.

        Parameters
        ----------
        table_name: str
            Table name to generate the report for
            (same as used in the training or in the inference processes)
        reports: Union[List[str], str]
            List of report types to generate.
            Supported values are: "accuracy", "metrics_only", "sample", "none", "all"
        fernet_key: Optional[str]
            the name of the environment variable kept the value of the Fernet key
            for decrypting the input of the original data, if applicable.
        """
        reports = get_reports(reports, ReportTypes(), "train")

        if reports:
            if fernet_key is not None:
                fernet_key = fetch_env_variables({"fernet_key": fernet_key}).get("fernet_key")
                DataEncryptor.validate_fernet_key(fernet_key)
            self._validate_artifacts(
                table_name=table_name,
                fernet_key=fernet_key,
                reports={
                    "accuracy"
                    if report in ["accuracy", "metrics_only"]
                    else report
                    for report in reports
                }
            )

        for report in reports:
            self._register_reporter(table_name, report, fernet_key)

        if reports:
            Report().generate_report()
            Report().clear_report()
        else:
            logger.warning(
                "No reports to generate. Please specify the report type "
                "from 'accuracy', 'metrics_only' or 'sample'."
            )

        self._set_execution_artifacts(
            type_of_process="report",
            table_name=table_name,
            reports=reports
        )
