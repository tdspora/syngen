from typing import Dict, List
import os
from dataclasses import dataclass, field
import json
from collections import defaultdict

from marshmallow import ValidationError
from slugify import slugify
from loguru import logger
from syngen.ml.data_loaders import MetadataLoader, DataLoader
from syngen.ml.validation_schema import ValidationSchema


@dataclass
class Validator:
    """
    Class for validating the metadata
    """

    metadata: Dict
    metadata_path: str
    type_of_process: str
    type_of_fk_keys = ["FK"]
    merged_metadata: Dict = field(default_factory=dict)
    mapping: Dict = field(default_factory=dict)
    errors = defaultdict(defaultdict)

    def _launch_validation_of_schema(self, metadata: Dict, metadata_path: str):
        """
        Launch the validation of the schema of the metadata
        """
        return ValidationSchema(
            metadata=self.metadata, metadata_path=self.metadata_path
        ).validate_schema()

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

    def _validate_metadata(self, table_name: str):
        """
        Validate the metadata
        """
        metadata_of_the_table = self.metadata[table_name]
        table_keys = metadata_of_the_table.get("keys", {})
        print_report = metadata_of_the_table.get("train_settings", {}).get("print_report", False)
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
                if self.type_of_process == "infer" or (
                    self.type_of_process == "train" and print_report is True
                ):
                    self._check_existence_of_success_file(parent_table)
                    self._check_existence_of_generated_data(parent_table)
                elif self.type_of_process == "train":
                    self._check_existence_of_success_file(parent_table)
            else:
                continue

    def _validate_referential_integrity(self, fk_name: str, fk_config: Dict, parent_config: Dict):
        """
        Validate whether the columns related to the primary key are the same as
        the referenced columns of the foreign key
        """
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
                f"the {fk_config['type']} - '{fk_name}' aren't the same"
            )
            self.errors["validate referential integrity"][fk_name] = message

    def _check_existence_of_success_file(self, parent_table: str):
        """
        Check if the success file of the certain parent table exists.
        The success file is created after the successful execution of the training process
        of the certain table.
        """
        if not os.path.exists(
            f"model_artifacts/resources/{slugify(parent_table)}/message.success"
        ):
            message = (
                f"The table - '{parent_table}' hasn't been trained completely. "
                f"Please, retrain this table first"
            )
            self.errors["check existence of the success file"][parent_table] = message

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
        if not DataLoader(destination).has_existed_path:
            message = (
                f"The generated data of the table - '{parent_table}' hasn't been generated. "
                f"Please, generate the data related to the table '{parent_table}' first"
            )
            self.errors["check existence of the generated data"][parent_table] = message

    def _check_existence_of_source(self, table_name: str) -> bool:
        """
        Check if the source of the certain table exists
        """
        if not DataLoader(
            self.merged_metadata[table_name]["train_settings"]["source"]
        ).has_existed_path:
            message = (
                f"It seems that the path to the source of the table - '{table_name}' "
                f"isn't correct. Please, check the path to the source of the table - "
                f"'{table_name}'"
            )
            self.errors["check existence of the source"][table_name] = message
            return False
        return True

    def _check_existence_of_destination(self, table_name: str):
        """
        Check if the destination of the certain table exists
        """
        destination = self.merged_metadata[table_name].get("infer_settings", {}).get("destination")
        if destination is not None and not DataLoader(destination).has_existed_destination:
            message = (
                f"It seems that the directory path for storing the generated data of table "
                f"'{table_name}' isn't correct. Please, verify the destination path"
            )
            self.errors["check existence of the destination"][table_name] = message

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
                metadata = MetadataLoader(path_to_metadata_file).load_data()
                if parent_table not in metadata:
                    continue
                self._launch_validation_of_schema(
                    metadata=metadata, metadata_path=path_to_metadata_file
                )
                self.merged_metadata.update(metadata)
                logger.info(
                    f"The metadata located in the path - '{path_to_metadata_storage}' "
                    f"has been merged with the current metadata as it contains "
                    f"the information of the parent table - '{parent_table}'"
                )
            self._check_merged_metadata(parent_table)

    def _check_existence_of_columns(
        self, metadata_of_table, table_name: str, existed_columns: List[str]
    ):
        """
        Check if the columns of the certain key exist in the source table
        """
        for key, config_of_key in metadata_of_table.get("keys", {}).items():
            if all([column in existed_columns for column in config_of_key["columns"]]):
                continue
            else:
                non_existed_columns = [
                    f"{col!r}"
                    for col in set(config_of_key["columns"]).difference(set(existed_columns))
                ]
                message = (
                    f"The columns of the {config_of_key['type']} '{key}' - "
                    f"{', '.join(non_existed_columns)} "
                    f"don't exist in the source of the table - '{table_name}'"
                )
                self.errors["check existence of the key columns in 'columns'"][key] = message

    def _check_references_columns(self, table_name, metadata_of_table):
        """
        Check if the columns of the certain key exist in the referenced table
        """
        for key, config_of_key in metadata_of_table.get("keys", {}).items():
            if config_of_key["type"] in self.type_of_fk_keys:
                referenced_columns = config_of_key["references"]["columns"]
                referenced_table = config_of_key["references"]["table"]
                existed_columns = self._fetch_existed_columns(
                    self.merged_metadata[referenced_table]
                )
                if all([column in existed_columns for column in referenced_columns]):
                    continue
                else:
                    non_existed_columns = [
                        f"{col!r}"
                        for col in set(referenced_columns).difference(set(existed_columns))
                    ]
                    message = (
                        f"The 'referenced.columns' of the {config_of_key['type']} '{key}' - "
                        f"{', '.join(non_existed_columns)} "
                        f"don't exist in the referenced table - '{table_name}'"
                    )
                    self.errors["check existence of the key columns in 'referenced.columns'"][
                        key
                    ] = message

    @staticmethod
    def _fetch_existed_columns(metadata_of_table) -> List[str]:
        """
        Fetch the list of the columns of the source table
        """
        return DataLoader(metadata_of_table["train_settings"]["source"]).get_columns()

    def _check_key_columns(self, table_name: str):
        """
        Fetch the list of the columns of the source table and
        check whether the columns of the certain key exist in the source table
        """
        metadata_of_table = self.merged_metadata[table_name]
        existed_columns = self._fetch_existed_columns(metadata_of_table)
        self._check_existence_of_columns(metadata_of_table, table_name, existed_columns)
        self._check_references_columns(table_name, metadata_of_table)

    def run(self):
        """
        Run the validation process
        """
        self._launch_validation_of_schema(metadata=self.metadata, metadata_path=self.metadata_path)
        self._define_mapping()
        self._merge_metadata()
        self.merged_metadata.pop("global", None)
        self.metadata.pop("global", None)
        for table_name in self.merged_metadata.keys():
            if self.type_of_process == "train":
                if self._check_existence_of_source(table_name):
                    self._check_key_columns(table_name)
            elif self.type_of_process == "infer":
                self._check_existence_of_destination(table_name)
        for table_name in self.metadata.keys():
            self._validate_metadata(table_name)
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
