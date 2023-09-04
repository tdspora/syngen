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

    def _check_referential_integrity(self, metadata):
        """
        Check if the references are valid
        """
        for table_name, table_metadata in metadata.items():
            if table_name == "global":
                continue
            metadata_keys = table_metadata.get("keys") \
                if "keys" in table_metadata and table_metadata.get("keys") is not None \
                else {}
            for key_name, key_data in metadata_keys.items():
                if key_data["type"] != "FK":
                    continue

                fk_name = key_name
                fk_config = key_data
                parent_table = key_data["references"]["table"]
                if self._validate_metadata(
                    pk_table=parent_table,
                    fk_name=fk_name,
                    fk_config=fk_config
                ) is False:
                    raise ValidationError(f"The referenced table \"{parent_table}\" is not found")
                return parent_table

    @staticmethod
    def _merge_metadata(metadata, parent_metadata) -> Dict:
        """
        Merge the metadata contained the parent table and
        the metadata contained the child table
        """
        merged_metadata = parent_metadata.copy()
        merged_metadata.update(metadata)
        return merged_metadata

    def _validate_metadata(self, pk_table, fk_name, fk_config):
        """
        Validate the metadata
        """
        pk_table_keys = self.merged_metadata[pk_table].get("keys", {})
        for key, config in pk_table_keys.items():
            if config["type"] in ["PK", "UQ"]:
                if self.type_of_process == "train":
                    return self._validate_pk_columns(pk_name=key, pk_config=config, fk_name=fk_name, fk_config=fk_config) \
                        and self._check_existence_of_success_file(pk_table)

                elif self.type_of_process == "infer":
                    return self._validate_pk_columns(pk_name=key, pk_config=config, fk_name=fk_name, fk_config=fk_config) \
                        and self._check_existence_of_success_file(pk_table) \
                        and self._check_existence_of_generated_data(pk_table)
            else:
                continue

    @staticmethod
    def _validate_pk_columns(pk_name, pk_config, fk_name, fk_config) -> bool:
        """
        Validate the primary key columns
        """
        result = len(pk_config.get("columns", [])) == len(fk_config["references"]["columns"])
        if result is False:
            logger.error(
                f"The number of columns in the primary key - \"{pk_name}\" "
                f"and foreign key - \"{fk_name}\" is different")
        return result

    @staticmethod
    def _check_existence_of_success_file(pk_table: str) -> bool:
        """
        Check if the success file of the certain table exists.
        The success file is created after the successful execution of the training process of the certain table.
        """
        path_to_success_file = os.path.exists(f"model_artifacts/resources/{slugify(pk_table)}/message.success")
        if os.path.exists(path_to_success_file):
            return True
        logger.error(f"The table \"{pk_table}\" hasn't been trained completely. Please, retrain this table first")
        return False

    def _check_existence_of_generated_data(self, pk_table: str) -> bool:
        destination = self.merged_metadata[pk_table].get("infer_settings", {}).get("destination")
        if destination is None:
            destination = f"model_artifacts/tmp_store/{slugify(pk_table)}/merged_infer_{slugify(pk_table)}.csv"
        if os.path.exists(destination):
            return True
        return False

    @staticmethod
    def _find_parent_metadata(parent_table) -> Dict:
        """
        Find the parent metadata contained the parent table
        in the metadata files stored in 'model_artifacts/metadata' directory
        """
        path_to_metadata_storage = "model_artifacts/metadata"
        for file in os.listdir(path_to_metadata_storage):
            metadata = MetadataLoader(os.path.join(path_to_metadata_storage, file)).load_data()
            if parent_table not in metadata:
                continue
            else:
                metadata.pop("global", None)
                return metadata
        return {}

    def run(self):
        """
        Run the validation process
        """
        parent_table = self._check_referential_integrity(metadata=self.metadata)
        parent_metadata = self._find_parent_metadata(parent_table=parent_table)
        self.merged_metadata = self._merge_metadata(metadata=self.metadata, parent_metadata=parent_metadata)
        ValidationSchema(metadata=self.merged_metadata).validate_schema()
