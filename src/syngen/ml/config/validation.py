from typing import Dict
import os
from dataclasses import dataclass, field
import json
from collections import defaultdict

from marshmallow import ValidationError
from slugify import slugify
from loguru import logger
from syngen.ml.data_loaders import MetadataLoader
from syngen.ml.validation_schema import ValidationSchema


@dataclass
class Validator:

    """
    Class for validating the metadata
    """
    metadata: Dict
    metadata_path: str
    type_of_process: str
    merged_metadata: Dict = field(default_factory=dict)
    mapping: Dict = field(default_factory=dict)
    errors = defaultdict(defaultdict)

    def _define_mapping(self):
        """
        Define the mapping contained the information of the foreign keys
        defined in the metadata
        """
        for table_name, table_metadata in self.metadata.items():
            if table_name == "global":
                continue
            metadata_keys = table_metadata.get("keys") \
                if "keys" in table_metadata and table_metadata.get("keys") is not None \
                else {}
            for key_name, key_data in metadata_keys.items():
                if key_data["type"] != "FK":
                    continue

                self.mapping[key_name] = {
                    "child_table": table_name,
                    "child_columns": key_data["columns"],
                    "parent_table": key_data["references"]["table"],
                    "parent_columns": key_data["references"]["columns"]
                }

    def _validate_metadata(self, table_name: str):
        """
        Validate the metadata
        """
        metadata_of_the_table = self.metadata[table_name]
        table_keys = metadata_of_the_table.get("keys", {})
        print_report = metadata_of_the_table.get("train_settings", {}).get("print_report", False)
        for key, config in table_keys.items():
            if config["type"] != "FK":
                continue
            self._validate_referential_integrity(
                fk_name=key, fk_config=config, parent_config=self.merged_metadata[self.mapping[key]["parent_table"]]
            )
            parent_table = self.mapping[key]["parent_table"]
            if parent_table not in self.metadata:
                if self.type_of_process == "infer" or (self.type_of_process == "train" and print_report is True):
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
        result = any([config["columns"] == fk_config["references"]["columns"]
                      for config in parent_config.get("keys", {}).values()
                      if config["type"] in ["PK", "UQ"]])
        if result is False:
            message = f"The primary key columns associated with the columns of " \
                      f"the foreign key - '{fk_name}' is not the same"
            self.errors["validate referential integrity"][fk_name] = message

    def _check_existence_of_success_file(self, parent_table: str):
        """
        Check if the success file of the certain parent table exists.
        The success file is created after the successful execution of the training process of the certain table.
        """
        if not os.path.exists(f"model_artifacts/resources/{slugify(parent_table)}/message.success"):
            message = f"The table - '{parent_table}' hasn't been trained completely. Please, retrain this table first"
            self.errors["check existence of the success file"][parent_table] = message

    def _check_existence_of_generated_data(self, parent_table: str):
        """
        Check if the generated data of the certain parent table exists.
        The generated data is created after the successful execution of the inference process of the certain table.
        """
        destination = self.merged_metadata[parent_table].get("infer_settings", {}).get("destination")
        if destination is None:
            destination = f"model_artifacts/tmp_store/{slugify(parent_table)}/merged_infer_{slugify(parent_table)}.csv"
        if not os.path.exists(destination):
            message = f"The generated data of the table - '{parent_table}' hasn't been generated. " \
                      f"Please, generate the data related to the table '{parent_table}' first"
            self.errors["check existence of the generated data"][parent_table] = message

    def _check_existence_of_source(self, table_name: str):
        """
        Check if the source of the certain table exists
        """
        if not os.path.exists(self.merged_metadata[table_name]["train_settings"]["source"]):
            message = f"It seems that the path to the source of the table - '{table_name}' isn't correct. " \
                      f"Please, check the path to the source of the table - '{table_name}'"
            self.errors["check existence of the source"][table_name] = message

    def _check_existence_of_destination(self, table_name: str):
        """
        Check if the destination of the certain table exists
        """
        destination = self.merged_metadata[table_name].get("infer_settings", {}).get("destination")
        if destination is not None and not os.path.exists(os.path.dirname(destination)):
            message = f"It seems that the directory path for storing the generated data of table '{table_name}' " \
                      f"isn't correct. Please, verify the destination path"
            self.errors["check existence of the destination"][table_name] = message

    def _check_merged_metadata(self, parent_table: str):
        if parent_table not in self.merged_metadata:
            message = f"The metadata of the parent table - '{parent_table}' hasn't been found. Please, check " \
                      f"whether the information of the parent table - '{parent_table}' exists in the current " \
                      f"metadata file or in the metadata files stored in 'model_artifacts/metadata' directory"
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
                ValidationSchema(metadata=metadata, metadata_path=path_to_metadata_file).validate_schema()
                self.merged_metadata.update(metadata)
                logger.info(f"The metadata located in the path - '{path_to_metadata_storage}' has been merged "
                            f"with the current metadata as it contains the information of the parent table - "
                            f"'{parent_table}'")
            self._check_merged_metadata(parent_table)

    def run(self):
        """
        Run the validation process
        """
        ValidationSchema(metadata=self.metadata, metadata_path=self.metadata_path).validate_schema()
        self._define_mapping()
        self._merge_metadata()
        self.merged_metadata.pop("global", None)
        self.metadata.pop("global", None)
        for table_name in self.merged_metadata.keys():
            if self.type_of_process == "train":
                self._check_existence_of_source(table_name)
            elif self.type_of_process == "infer":
                self._check_existence_of_destination(table_name)
        for table_name in self.metadata.keys():
            self._validate_metadata(table_name)
        error_logs = []
        for section, errors_details in self.errors.items():
            error_log = f"\"{section}\": {json.dumps(errors_details, indent=4)}"
            error_logs.append(error_log)
        if self.errors:
            message = f"The validation of the metadata has been failed. " \
                      f"The error(s) found in - \n{', '.join(error_logs)}"
            logger.error(message)
            raise ValidationError(message)
        logger.info("The validation of the metadata has been passed successfully")