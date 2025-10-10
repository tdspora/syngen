from typing import Optional, List, Union, Literal
import os
import pandas as pd
from dataclasses import dataclass
from loguru import logger

from slugify import slugify
from marshmallow import ValidationError
from syngen.ml.data_loaders import DataLoader, DataEncryptor
from syngen.train import launch_train
from syngen.infer import launch_infer
from syngen.ml.utils import fetch_config, fetch_env_variables, get_reports
from syngen.ml.reporters import (
    Report,
    AccuracyReporter,
    SampleAccuracyReporter,
)


@dataclass
class DataIO:
    path: str
    fernet_key: Optional[str] = None

    def load_data(self) -> pd.DataFrame:
        data_loader = DataLoader(
            path=self.path,
            table_name="table",
            metadata={
                "table": {
                    "encryption": {
                        "fernet_key": (
                            fetch_env_variables({"fernet_key": self.fernet_key}).get("fernet_key")
                        ),
                    }
                }
            }
        )
        df, _ = data_loader.load_data()
        return df

    def save_data(self, df: pd.DataFrame):
        data_loader = DataLoader(
            path=self.path,
            table_name="table",
            metadata={
                "table": {
                    "encryption": {
                        "fernet_key": (
                            fetch_env_variables({"fernet_key": self.fernet_key}).get("fernet_key")
                        ),
                    }
                }
            }
        )
        data_loader.save_data(df)


class Syngen:

    @staticmethod
    def train(
        metadata_path: Optional[str] = None,
        table_name: Optional[str] = None,
        source: Optional[str] = None,
        epochs: int = 10,
        drop_null: bool = False,
        row_limit: Optional[int] = None,
        reports: Union[str, List[str]] = "none",
        log_level: str = "INFO",
        batch_size: int = 32,
        fernet_key: Optional[str] = None
    ):
        launch_train(
            metadata_path=metadata_path,
            source=source,
            table_name=table_name,
            epochs=epochs,
            drop_null=drop_null,
            row_limit=row_limit,
            reports=reports,
            log_level=log_level,
            batch_size=batch_size,
            fernet_key=fernet_key
        )

    @staticmethod
    def infer(
        metadata_path: Optional[str] = None,
        size: int = 100,
        table_name: Optional[str] = None,
        run_parallel: bool = False,
        batch_size: int = 32,
        random_seed: Optional[int] = None,
        reports: Union[str, List[str]] = "none",
        log_level: str = "INFO",
        fernet_key: Optional[str] = None,
    ):
        launch_infer(
            metadata_path=metadata_path,
            size=size,
            table_name=table_name,
            run_parallel=run_parallel,
            batch_size=batch_size,
            reports=reports,
            random_seed=random_seed,
            log_level=log_level,
            fernet_key=fernet_key
        )

    @staticmethod
    def _validate_artifacts(
        table_name: str,
        fernet_key: Optional[str],
        reports: List[str]
    ):
        errors: List[str] = []

        slug = slugify(table_name)

        reports = set(reports)

        if fernet_key is not None:
            try:
                DataEncryptor.validate_fernet_key(fernet_key)
            except ValueError as error:
                errors.append(str(error))

        path_to_input_data = (
            f"model_artifacts/tmp_store/{slug}/"
            f"input_data_{slug}.{'dat' if fernet_key is not None else 'pkl'}"
        )

        path_to_train_config = (
            f"model_artifacts/resources/{slug}/vae/checkpoints/train_config.pkl"
        )
        path_to_train_success_file = f"model_artifacts/resources/{slug}/train_message.success"
        path_to_infer_config = f"model_artifacts/tmp_store/{slug}/infer_config.pkl"
        path_to_infer_success_file = f"model_artifacts/tmp_store/{slug}/infer_message.success"
        path_to_model_dataset = (
            f"model_artifacts/resources/{slug}/vae/checkpoints/model_dataset.pkl"
        )
        common_check_of_artifacts = [
            (
                path_to_input_data,
                (
                    f"The input data file wasn't found for the table '{table_name}' "
                    f"in the path - '{path_to_input_data}'."
                )
            ),
            (
                path_to_model_dataset,
                (
                    f"The model dataset file wasn't found for the table '{table_name}' "
                    f"in the path - '{path_to_model_dataset}'."
                ),
            ),
        ]
        checks_of_artifacts = {
            "sample": [
                (
                    path_to_train_config,
                    (
                        f"The training configuration wasn't found for the table '{table_name}' "
                        f"in the path - '{path_to_train_config}'."
                    ),
                ),
                (
                    path_to_train_success_file,
                    (
                        f"The training success file wasn't found for the table '{table_name}' "
                        f"in the path - '{path_to_train_success_file}'."
                    ),
                ),
            ],
            "accuracy": [
                (
                    path_to_infer_config,
                    (
                        f"The inference configuration wasn't found for table '{table_name}' "
                        f"in the path - '{path_to_infer_config}'"
                    ),
                ),
                (
                    path_to_infer_success_file,
                    (
                        f"The inference success file wasn't found for table '{table_name}' "
                        f"in the path - '{path_to_infer_success_file}'."
                    ),
                ),
            ],
        }
        for path, message in common_check_of_artifacts:
            if not os.path.exists(path):
                errors.append(message)

        type_of_reports = set(
            [
                "accuracy"
                if report in ["accuracy", "metrics_only"]
                else "sample"
                for report in reports
            ]
        )
        for report in type_of_reports:
            for path, message in checks_of_artifacts.get(report, []):
                if not os.path.exists(path):
                    errors.append(message)

        if errors:
            process_phrase = (
                "training and inference"
                if ("accuracy" in reports or "metrics_only" in reports)
                else "training"
            )
            errors.append(
                f"Before the generation of reports, please ensure that the {process_phrase} "
                f"process of the table '{table_name}' were completed successfully."
            )
            raise ValidationError("\n".join(errors))

    @staticmethod
    def _get_sample_reporter(table_name: str) -> SampleAccuracyReporter:
        """
        Return a SampleAccuracyReporter instance for the specified table
        """
        path_to_train_config = (
            f"model_artifacts/resources/{slugify(table_name)}/vae/checkpoints/train_config.pkl"
        )
        train_config = fetch_config(config_pickle_path=path_to_train_config)
        return SampleAccuracyReporter(
            table_name=table_name,
            paths=train_config.paths,
            config=train_config.to_dict(),
            metadata=train_config.metadata,
        )

    @staticmethod
    def _get_accuracy_reporter(
        table_name: str,
        report: str = Literal["accuracy", "metrics_only"]
    ) -> AccuracyReporter:
        path_to_infer_config = (
            f"model_artifacts/tmp_store/{slugify(table_name)}/"
            f"infer_config.pkl"
        )
        infer_config = fetch_config(config_pickle_path=path_to_infer_config)
        infer_config.reports = [report]
        return AccuracyReporter(
            table_name=table_name,
            paths=infer_config.paths,
            config=infer_config.to_dict(),
            metadata=infer_config.metadata
        )

    def generate_reports(
        self,
        table_name: str,
        reports: Union[str, List],
        fernet_key: Optional[str] = None
    ) -> None:
        """
        Generate reports for a table using existing artifacts.

        Parameters
        ----------
        table_name: str
            Table name to generate the report for
            (same as used in the training or in the inference process)
        reports: str
            The list of reports to generate.
            Supported values are 'accuracy', 'metrics_only' and 'sample'.
        fernet_key: Optional[str]
            the name of the environment variable kept the value of the Fernet key
            for decrypting the input of the original data, if applicable.
        """
        reports = get_reports(reports, type_of_process="train")
        if reports:
            self._validate_artifacts(table_name, fernet_key, reports)
        else:
            logger.warning(
                "No reports to generate. Please specify the report type "
                "from 'accuracy', 'metrics_only' or 'sample'."
            )
            return

        for report in reports:
            if report in ["accuracy", "metrics_only"]:
                accuracy_reporter = self._get_accuracy_reporter(table_name, report)
                Report().register_reporter(table=table_name, reporter=accuracy_reporter)
            elif report == "sample":
                sample_reporter = self._get_sample_reporter(table_name)
                Report().register_reporter(table=table_name, reporter=sample_reporter)
            else:
                raise ValueError(
                    f"Invalid report type - '{report}'. "
                    "Use 'accuracy', 'metrics_only' or 'sample'."
                )

        Report().generate_report()


if __name__ == "__main__":
    Syngen().generate_reports(
        table_name="bgb_batchlogs",
        reports=["sample"]
    )
