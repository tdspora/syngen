from typing import Dict
import os
from dataclasses import dataclass, field

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
    type_of_process: str
    merged_metadata: Dict = field(default_factory=dict)
    mapping: Dict = field(default_factory=dict)

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
        metadata_of_the_table = self.merged_metadata[table_name]
        table_keys = metadata_of_the_table.get("keys", {})
        print_report = metadata_of_the_table.get("train_settings", {}).get("print_report", False)
        result = True
        for key, config in table_keys.items():
            if config["type"] != "FK":
                continue
            check_referential_integrity = self._validate_referential_integrity(
                fk_name=key, fk_config=config, parent_config=self.merged_metadata[self.mapping[key]["parent_table"]]
            )
            parent_table = self.mapping[key]["parent_table"]
            if parent_table in self.metadata:
                result = check_referential_integrity
            elif parent_table not in self.metadata:
                if self.type_of_process == "infer" or (self.type_of_process == "train" and print_report is True):
                    result = check_referential_integrity \
                             and self._check_existence_of_success_file(parent_table) \
                             and self._check_existence_of_generated_data(parent_table)
                elif self.type_of_process == "train":
                    result = check_referential_integrity \
                             and self._check_existence_of_success_file(parent_table)
            else:
                continue
        if result is False:
            message = f"The validation of the metadata of the table - '{table_name}' failed"
            logger.error(message)
            raise ValidationError(message)

    @staticmethod
    def _validate_referential_integrity(fk_name: str, fk_config: Dict, parent_config: Dict) -> bool:
        """
        Validate the equality of the number of columns in the primary key and the foreign key
        """
        result = any([config["columns"] == fk_config["references"]["columns"]
                      for config in parent_config.get("keys", {}).values()])
        if result is False:
            logger.error(
                f"The primary key columns associated with the columns of the foreign key - '{fk_name}' is not the same"
            )
        return result

    @staticmethod
    def _check_existence_of_success_file(parent_table: str) -> bool:
        """
        Check if the success file of the certain parent table exists.
        The success file is created after the successful execution of the training process of the certain table.
        """
        path_to_success_file = os.path.exists(f"model_artifacts/resources/{slugify(parent_table)}/message.success")
        if os.path.exists(path_to_success_file):
            return True
        else:
            logger.error(
                f"The table - '{parent_table}' hasn't been trained completely. Please, retrain this table first"
            )
            return False

    def _check_existence_of_generated_data(self, parent_table: str) -> bool:
        """
        Check if the generated data of the certain parent table exists.
        The generated data is created after the successful execution of the inference process of the certain table.
        """
        destination = self.merged_metadata[parent_table].get("infer_settings", {}).get("destination")
        if destination is None:
            destination = f"model_artifacts/tmp_store/{slugify(parent_table)}/merged_infer_{slugify(parent_table)}.csv"
        if os.path.exists(destination):
            return True
        logger.error(f"The generated data of the table - '{parent_table}' hasn't been generated. "
                     f"Please, generate the data related to the table '{parent_table}' first")
        return False

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
                metadata = MetadataLoader(os.path.join(path_to_metadata_storage, file)).load_data()
                if parent_table not in metadata:
                    continue
                self.merged_metadata.update(metadata)
                logger.info(f"The metadata located in the path - '{path_to_metadata_storage}' has been merged "
                            f"with the current metadata as it contains the information of the parent table - "
                            f"'{parent_table}'")

    def run(self):
        """
        Run the validation process
        """
        self._define_mapping()
        self._merge_metadata()
        ValidationSchema(metadata=self.merged_metadata).validate_schema()
        self.merged_metadata.pop("global", None)
        self.metadata.pop("global", None)
        for table_name in self.merged_metadata.keys():
            self._validate_metadata(table_name)
        logger.info("The validation of the metadata has been passed successfully")
