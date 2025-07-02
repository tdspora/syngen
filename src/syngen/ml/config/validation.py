from typing import Dict, List, Literal
import os
from dataclasses import dataclass, field
import json
from collections import defaultdict
from cryptography.fernet import InvalidToken

from slugify import slugify
from loguru import logger
from syngen.ml.data_loaders import MetadataLoader, DataLoader, DataEncryptor
from syngen.ml.validation_schema import ValidationSchema, ReportTypes
from syngen.ml.utils import ValidationError, fetch_config


@dataclass
class Validator:
    """
    Class for validating the metadata
    """

    metadata: Dict
    metadata_path: str
    type_of_process: Literal["train", "infer"]
    validation_source: bool = True
    type_of_fk_keys = ["FK"]
    infer_report_types: List[str] = field(
        default_factory=lambda: ReportTypes().infer_report_types
    )
    merged_metadata: Dict = field(default_factory=dict)
    mapping: Dict = field(default_factory=dict)
    existed_columns_mapping: Dict = field(default_factory=dict)
    errors = defaultdict(defaultdict)

    def _define_mapping(self):
        """
        Define the mapping contained the information of the foreign keys
        defined in the metadata
        """
        for table_name, table_metadata in self.metadata.items():
            if table_name == "global":
                continue
            metadata_keys = (
                table_metadata.get("keys")
                if "keys" in table_metadata and table_metadata.get("keys") is not None
                else {}
            )
            for key_name, key_data in metadata_keys.items():
                if key_data["type"] not in self.type_of_fk_keys:
                    continue

                self.mapping[key_name] = {
                    "parent_table": key_data["references"]["table"],
                    "parent_columns": key_data["references"]["columns"],
                }

    def _check_conditions(self, metadata: Dict) -> bool:
        """
        Check conditions whether to launch validation or not
        """
        reports = metadata.get("train_settings", {}).get("reports", [])
        return (
            self.type_of_process == "infer"
            or (
                self.type_of_process == "train" and
                any([item in self.infer_report_types for item in reports])
            )
        )

    def _validate_metadata(self, table_name: str):
        """
        Validate the metadata
        """
        metadata_of_the_table = self.metadata[table_name]
        table_keys = metadata_of_the_table.get("keys", {})
        for key, config in table_keys.items():
            if config["type"] not in self.type_of_fk_keys:
                continue
            self._validate_referential_integrity(
                fk_name=key,
                fk_config=config,
                parent_config=self.merged_metadata[self.mapping[key]["parent_table"]],
            )
            parent_table = self.mapping[key]["parent_table"]
            if parent_table not in self.metadata:
                if self._check_conditions(metadata_of_the_table):
                    self._check_completion_of_training(parent_table)
                    self._check_existence_of_generated_data(parent_table)
                elif self.type_of_process == "train":
                    self._check_completion_of_training(parent_table)
            else:
                continue

    def _validate_schema(self):
        """
        Validate the schema of the metadata
        """
        ValidationSchema(
            metadata=self.metadata,
            metadata_path=self.metadata_path,
            validation_source=self.validation_source,
            process=self.type_of_process
        ).validate_schema()

    def _validate_referential_integrity(self, fk_name: str, fk_config: Dict, parent_config: Dict):
        """
        Validate whether the columns related to the primary key are the same as
        the referenced columns of the foreign key
        """
        keys = parent_config.get("keys", {})
        if not keys or all(config["type"] not in ["PK", "UQ"] for config in keys.values()):
            message = (
                "The information about columns of the primary or unique key "
                f"associated with the columns of the '{fk_config['type']}' key - '{fk_name}' "
                "wasn't found in the metadata of the parent table"
            )
            self.errors["validate referential integrity"][fk_name] = message
            return

        result = any(
            [
                config["columns"] == fk_config["references"]["columns"]
                for config in parent_config.get("keys", {}).values()
                if config["type"] in ["PK", "UQ"]
            ]
        )
        if result is False:
            message = (
                f"The columns of primary or unique key associated with the columns of "
                f"the '{fk_config['type']}' key - '{fk_name}' aren't the same"
            )
            self.errors["validate referential integrity"][fk_name] = message

    def _check_existence_of_generated_data(self, parent_table: str):
        """
        Check if the generated data of the certain parent table exists.
        The generated data is created after the successful execution of the inference process
        of the certain table.
        """
        destination = (
            self.merged_metadata[parent_table].get("infer_settings", {}).get("destination")
        )
        if destination is None:
            destination = (
                f"model_artifacts/tmp_store/{slugify(parent_table)}/"
                f"merged_infer_{slugify(parent_table)}.csv"
            )
        if not DataLoader(path=destination).has_existed_path:
            message = (
                f"The generated data of the table - '{parent_table}' hasn't been generated. "
                f"Please, generate the data related to the table '{parent_table}' first"
            )
            self.errors["check existence of the generated data"][parent_table] = message

    def _check_existence_of_source(self, table_name: str):
        """
        Check if the source of the certain table exists
        """
        path_to_source = self.merged_metadata[table_name]["train_settings"]["source"]
        if not DataLoader(path=path_to_source).has_existed_path:
            message = (
                f"It seems that the path to the source of the table - '{table_name}' "
                f"isn't correct. Please, check the path to the source of the table - "
                f"'{table_name}'"
            )
            self.errors["check existence of the source"][table_name] = message

    def _check_existence_of_destination(self, table_name: str):
        """
        Check if the destination of the certain table exists
        """
        destination = self.merged_metadata[table_name].get("infer_settings", {}).get("destination")
        if destination is None:
            logger.warning(
                f"As the destination path wasn't specified for the table - "
                f"'{table_name}', the synthetic data will be stored "
                f"at the default path - './model_artifacts/tmp_store/{slugify(table_name)}/"
                f"merged_infer_{slugify(table_name)}.csv'"
            )
        if destination is not None and not DataLoader(path=destination).has_existed_destination:
            message = (
                f"It seems that the directory path for storing the generated data of table "
                f"'{table_name}' isn't correct. Please, verify the destination path"
            )
            self.errors["check existence of the destination"][table_name] = message

    def _check_completion_of_training(self, table_name: str):
        """
        Check if the training process of a specific table has been completed.

        Args:
        table_name (str): The name of the table to check.

        Raises:
        FileNotFoundError: If the success file does not exist.
        ValueError: If the content of the success file does not indicate success.
        """
        path_to_success_file = (
            f"model_artifacts/resources/{slugify(table_name)}/train_message.success"
        )
        error_message = (
            f"The training of the table - '{table_name}' hasn't been completed. "
            "Please, retrain the table."
        )

        content = str()
        if os.path.exists(path_to_success_file):
            with open(path_to_success_file, "r") as file:
                content = file.read().strip()

        if content != "SUCCESS":
            self.errors["check completion of the training process"][table_name] = error_message

    def _check_merged_metadata(self, parent_table: str):
        if parent_table not in self.merged_metadata:
            message = (
                f"The metadata of the parent table - '{parent_table}' hasn't been found. "
                f"Please, check whether the information of the parent table - '{parent_table}' "
                f"exists in the current metadata file or in the metadata files stored in "
                f"'model_artifacts/metadata' directory"
            )
            logger.error(message)
            raise ValidationError(message)

    def _merge_metadata(self):
        """
        Find the parent metadata contained the parent table
        in the metadata files stored in 'model_artifacts/metadata' directory,
        and merge it with the metadata of the child table
        """
        self.merged_metadata = self.metadata.copy()
        for key_name, config in self.mapping.items():
            parent_table = config.get("parent_table")
            if parent_table in self.metadata:
                continue
            path_to_metadata_storage = "model_artifacts/metadata"
            for file in os.listdir(path_to_metadata_storage):
                path_to_metadata_file = os.path.join(path_to_metadata_storage, file)
                metadata = MetadataLoader(path=path_to_metadata_file).load_data()
                if parent_table not in metadata:
                    continue
                # Validate the schema of the metadata of the parent table
                self._validate_schema()

                metadata.pop("global")
                self.merged_metadata.update(metadata)

                logger.info(
                    f"The metadata located in the path - '{path_to_metadata_storage}' "
                    f"has been merged with the current metadata as it contains "
                    f"the information of the parent table - '{parent_table}'"
                )
            self._check_merged_metadata(parent_table)

    def _check_existence_of_key_columns(self, table_name):
        """
        Check whether the key columns exist in the table
        """
        metadata_of_table = self.merged_metadata[table_name]
        existed_columns = self.existed_columns_mapping[table_name]
        for key, config_of_key in metadata_of_table.get("keys", {}).items():
            if all([column in existed_columns for column in config_of_key["columns"]]):
                continue
            else:
                non_existed_columns = [
                    f"{col!r}"
                    for col in set(config_of_key["columns"]).difference(set(existed_columns))
                ]
                message = (
                    f"The columns of the {config_of_key['type']} key '{key}' - "
                    f"{', '.join(non_existed_columns)} "
                    f"don't exist in the table - '{table_name}'"
                )
                self.errors["check existence of the key columns in 'columns'"][key] = message

    def _check_existence_of_referenced_columns(self, table_name: str):
        """
        Check whether the key columns exist in the referenced table
        """
        metadata_of_table = self.merged_metadata[table_name]
        for key, config_of_key in metadata_of_table.get("keys", {}).items():
            if config_of_key["type"] in self.type_of_fk_keys:
                referenced_table = config_of_key["references"]["table"]
                referenced_columns = config_of_key["references"]["columns"]
                existed_columns = self.existed_columns_mapping.get(referenced_table, {})
                non_existed_columns = [
                    f"{col!r}"
                    for col in referenced_columns
                    if col not in existed_columns
                ]
                if non_existed_columns:
                    message = (
                        f"The 'references.columns' of the {config_of_key['type']} '{key}' - "
                        f"{', '.join(non_existed_columns)} "
                        f"don't exist in the referenced table - '{referenced_table}'"
                    )
                    self.errors[
                        "check existence of the key columns in 'references.columns'"
                    ][key] = message

    def _fetch_existed_columns(self, table_name: str) -> List[str]:
        """
        Fetch the list of the columns of the source table
        """
        metadata_of_table = self.merged_metadata[table_name]
        format_settings = metadata_of_table.get("format", {})
        path_to_source = self._fetch_path_to_source(table_name)
        data_loader = DataLoader(path=path_to_source)
        return data_loader.get_columns(**format_settings)

    def _gather_existed_columns(self, table_name: str):
        """
        Fetch the list of the existed columns of the table
        and put it into the mapping of existed columns
        """
        existed_columns = self._fetch_existed_columns(table_name)
        self.existed_columns_mapping[table_name] = existed_columns

    def _preprocess_metadata(self):
        """
        Preprocess the metadata, set the metadata and the merged metadata
        """
        self._define_mapping()
        self._merge_metadata()

    def _fetch_path_to_source(self, table_name):
        """
        Fetch the path to the source of the certain table
        """
        return self.merged_metadata[table_name]["train_settings"]["source"]

    def _validate_fernet_key(self, table_name: str, fernet_key: str):
        """
        Validate the structure of the fernet key
        """
        try:
            DataEncryptor.validate_fernet_key(fernet_key)
        except ValueError as e:
            self.errors["validate structure of fernet key"][table_name] = str(e)

    def _check_access_to_input_data(self, table_name: str):
        """
        Check if the input data is accessible for the inference process
        """
        path_to_train_config = (
            f"model_artifacts/resources/{slugify(table_name)}/vae/checkpoints/train_config.pkl"
        )
        path_to_input_data = fetch_config(path_to_train_config).paths["input_data_path"]
        log_message = (
            "Access to a sample of the original data is required "
            "to generate reports during the inference process"
        )
        try:
            data_loader = DataLoader(
                path=path_to_input_data,
                table_name=table_name,
                metadata=self.merged_metadata,
                sensitive=True
            )
            data_loader.get_columns()
        except InvalidToken:
            self.errors["check access to input data"][table_name] = (
                f"{log_message}. The provided Fernet key is invalid. "
                "The attempt to decrypt the sample of the original data has been failed. "
                "Please, provide the valid Fernet key."
            )
        except Exception as e:
            self.errors["check access to input data"][table_name] = f"{log_message}. {str(e)}"

    def _check_conditions_of_existence_of_input_data(self, table_name: str) -> bool:
        """
        Check if the input data is accessible for the inference process
        """
        return (
            not self.errors.get("validate structure of fernet key", {}).get(table_name)
            and not self.errors.get("check completion of the training process", {}).get(table_name)
        )

    def _launch_validation(self):
        """
        Launch the validation process
        """
        if self.type_of_process == "train" and self.validation_source:
            for table_name in self.merged_metadata.keys():
                self._gather_existed_columns(table_name)

        for table_name, table_metadata in self.metadata.items():
            fernet_key = table_metadata.get("encryption", {}).get("fernet_key")
            reports = table_metadata.get("infer_settings", {}).get("reports", [])

            if fernet_key is not None and (
                self.type_of_process == "train" or (
                    self.type_of_process == "infer" and reports
                )
            ):
                logger.warning(
                    f"Encryption and decryption are enabled for the table '{table_name}' "
                    "as a Fernet key is provided"
                )
                self._validate_fernet_key(table_name, fernet_key)

            if fernet_key is not None and self.type_of_process == "infer" and not reports:
                logger.warning(
                    f"The Fernet key is provided for the table '{table_name}' "
                    f"hasn't been validated because it will not be used"
                )

            if self.type_of_process == "train" and self.validation_source:
                self._check_existence_of_source(table_name)
                self._check_existence_of_key_columns(table_name)
                self._check_existence_of_referenced_columns(table_name)

            elif self.type_of_process == "infer":
                self._check_completion_of_training(table_name)
                self._check_existence_of_destination(table_name)
                if reports and self._check_conditions_of_existence_of_input_data(table_name):
                    self._check_access_to_input_data(table_name)

        for table_name in self.metadata.keys():
            self._validate_metadata(table_name)

    def _collect_errors(self):
        """
        Collect the errors found during the validation process
        """
        error_logs = []
        for section, errors_details in self.errors.items():
            error_log = f'"{section}": {json.dumps(errors_details, indent=4)}'
            error_logs.append(error_log)
        if self.errors:
            message = (
                f"The validation of the metadata has been failed. "
                f"The error(s) found in - \n{', '.join(error_logs)}"
            )
            logger.error(message)
            raise ValidationError(message)
        logger.info("The validation of the metadata has been passed successfully")

    def run(self):
        """
        Run the validation process
        """
        self._preprocess_metadata()
        self._launch_validation()
        self._collect_errors()
