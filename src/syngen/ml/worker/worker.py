from typing import Dict, List, Optional, Any, Callable, Literal, Tuple
import os
import shutil
from itertools import product

import pandas as pd
from attrs import define, field
from copy import deepcopy
from loguru import logger
from slugify import slugify

from syngen.ml.data_loaders import MetadataLoader, DataLoader
from syngen.ml.strategies import TrainStrategy, InferStrategy
from syngen.ml.reporters import Report
from syngen.ml.config import Validator
from syngen.ml.mlflow_tracker import MlflowTrackerFactory
from syngen.ml.context.context import global_context
from syngen.ml.utils import ProgressBarHandler
from syngen.ml.mlflow_tracker import MlflowTracker
from syngen.ml.processors import PreprocessHandler, PostprocessHandler
from syngen.ml.validation_schema import ValidationSchema


@define
class Worker:
    """
    Class for preparing training and infer settings, metadata for training and infer process
    """

    table_name: Optional[str] = field(kw_only=True)
    metadata_path: Optional[str] = field(kw_only=True)
    settings: Dict = field(kw_only=True)
    log_level: str = field(kw_only=True)
    encryption_settings: Dict = field(kw_only=True)
    type_of_process: Literal["train", "infer"] = field(kw_only=True)
    metadata: Optional[Dict] = None
    loader: Optional[Callable[[str], pd.DataFrame]] = None
    divided: List = field(default=list())
    initial_table_names: List = field(default=list())
    merged_metadata: Dict = field(default=dict())
    validation_source: bool = field(default=False)
    train_stages: List = ["PREPROCESS", "TRAIN", "POSTPROCESS"]
    infer_stages: List = ["INFER", "REPORT"]

    def __attrs_post_init__(self):
        self.validation_source = (
            False if self.loader and self.type_of_process == "train" else True
        )
        os.makedirs("model_artifacts/metadata", exist_ok=True)
        self.metadata = self.__fetch_metadata()
        # The validation of the initial metadata provided by the user
        self.__validate_schema()
        self._update_metadata()
        self._clean_up()
        self.__validate_metadata()
        self.initial_table_names = list(self.merged_metadata.keys())
        self._set_mlflow()

    def _clean_up(self):
        """
        Clean up the directories stored the artifacts
        from the previous run of a training or inference process
        """
        for table in self.metadata.keys():
            if table == "global":
                continue

            slugified_table = slugify(table)

            if self.type_of_process == "train":
                self._clean_training_directories(slugified_table)
            elif self.type_of_process == "infer":
                self._clean_inference_directories(slugified_table)

    def _clean_training_directories(self, table):
        """
        Remove existing artifacts and prepare directories before a training process
        """
        resources_path = f"model_artifacts/resources/{table}/"
        tmp_store_path = f"model_artifacts/tmp_store/{table}/"

        self._remove_existed_artifact(resources_path)
        self._remove_existed_artifact(tmp_store_path)
        self._prepare_dirs(table)

    def _clean_inference_directories(self, table):
        """
        Remove existing artifacts before an inference process
        """
        path_to_reports = f"model_artifacts/tmp_store/{table}/reports"
        default_path_to_merged_infer = (
            f"model_artifacts/tmp_store/{table}/merged_infer_{table}.csv"
        )
        success_file_path = f"model_artifacts/tmp_store/{table}/infer_message.success"

        self._remove_existed_artifact(path_to_reports)
        self._remove_existed_artifact(default_path_to_merged_infer)
        self._remove_existed_artifact(success_file_path)

    @staticmethod
    def _remove_existed_artifact(path_to_artifact: str):
        """
        Remove the existed artifact from the previous training process
        """
        if os.path.exists(path_to_artifact):
            if os.path.isfile(path_to_artifact):
                os.remove(path_to_artifact)
            else:
                shutil.rmtree(path_to_artifact)
            logger.info(f"The artifacts located in the path - '{path_to_artifact}' was removed")

    @staticmethod
    def _prepare_dirs(table: str):
        """
        Create main directories for saving original, synthetic data and model artifacts
        """
        resources_path = f"model_artifacts/resources/{table}/"
        tmp_store_path = f"model_artifacts/tmp_store/{table}/"
        state_path = (
            f"model_artifacts/resources/{table}/vae/checkpoints"
        )
        flatten_config_path = "model_artifacts/system_store/flatten_configs/"
        losses_path = "model_artifacts/system_store/losses/"
        os.makedirs(resources_path, exist_ok=True)
        os.makedirs(tmp_store_path, exist_ok=True)
        os.makedirs(state_path, exist_ok=True)
        os.makedirs(flatten_config_path, exist_ok=True)
        os.makedirs(losses_path, exist_ok=True)

    def __validate_schema(self):
        """
        Validate the schema of the metadata file
        """
        ValidationSchema(
            metadata=self.metadata,
            metadata_path=self.metadata_path,
            validation_source=self.validation_source,
            process=self.type_of_process
        ).validate_schema()

    def __validate_metadata(self):
        """
        Validate the metadata, set the merged metadata
        """
        validator = Validator(
            metadata=self.metadata,
            metadata_path=self.metadata_path,
            type_of_process=self.type_of_process,
            validation_source=self.validation_source
        )
        validator.run()
        self.merged_metadata = validator.merged_metadata

    def __preprocess_data(self, table_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess the data before a training process
        """
        return PreprocessHandler(
            metadata=self.metadata,
            metadata_path=self.metadata_path,
            table_name=table_name,
            loader=self.loader
        ).run()

    def __postprocess_data(self):
        """
        Postprocess the data after an inference process
        """
        PostprocessHandler(
            metadata=self.metadata,
            metadata_path=self.metadata_path,
            table_name=self.table_name
        ).run()

    def _set_mlflow(self):
        """
        Set the mlflow experiment name and the mlflow run
        """
        MlflowTrackerFactory.create_tracker(
            table_name=self.table_name,
            metadata_path=self.metadata_path
        )

    def _update_metadata_for_table(self):
        """
        Update the metadata for training or inference process if a metadata file wasn't provided
        """
        if self.type_of_process == "train":
            train_settings = self.metadata[self.table_name]["train_settings"]
            train_settings.update(self.settings)
        elif self.type_of_process == "infer":
            infer_settings = self.metadata[self.table_name]["infer_settings"]
            infer_settings.update(self.settings)

    @staticmethod
    def _update_table_settings(
        table_settings: Dict[str, Any], settings_to_update: Dict[str, Any]
    ) -> None:
        """
        Update the table settings with the provided settings that are not already defined
        """
        for setting, value in settings_to_update.items():
            if setting not in table_settings:
                table_settings[setting] = value

    def _update_metadata_for_tables(self):
        """
        Update the metadata for training or inference process if a metadata file was provided
        """
        global_train_settings = self.metadata.get("global", {}).get("train_settings", {})
        global_infer_settings = self.metadata.get("global", {}).get("infer_settings", {})
        global_encryption_settings = self.metadata.get("global", {}).get("encryption", {})

        process_settings_map = {
            "train": ("train_settings", global_train_settings),
            "infer": ("infer_settings", global_infer_settings),
        }

        process_info = process_settings_map.get(self.type_of_process)
        settings_key, global_process_settings = process_info

        for table, table_metadata in self.metadata.items():
            if table == "global":
                continue

            table_process_settings = table_metadata.setdefault(settings_key, {})
            table_encryption_settings = table_metadata.setdefault("encryption", {})

            self._update_table_settings(table_process_settings, global_process_settings)
            self._update_table_settings(table_process_settings, self.settings)
            self._update_table_settings(table_encryption_settings, global_encryption_settings)
            self._update_table_settings(table_encryption_settings, self.encryption_settings)

    def _update_metadata(self) -> None:
        if self.metadata_path:
            self._update_metadata_for_tables()
            self.metadata.pop("global", None)
        if self.table_name:
            self._update_metadata_for_table()

    def __fetch_metadata(self) -> Dict:
        """
        Fetch the metadata for training or infer process
        """
        if self.metadata_path:
            metadata = MetadataLoader(path=self.metadata_path).load_data()
            return metadata
        if self.table_name:
            source = self.settings.get("source")
            metadata = {
                self.table_name: {
                    "train_settings": {
                        "source": source,
                    },
                    "infer_settings": {},
                    "encryption": {"fernet_key": self.encryption_settings.get("fernet_key")},
                    "keys": {},
                    "format": {}
                }
            }
            return metadata

    @staticmethod
    def _get_tables_without_keys(config_of_tables: Dict) -> List[str]:
        """
        Return the list of tables without the information about keys
        """
        tables_without_keys = []
        for table in config_of_tables.keys():
            if not config_of_tables[table].get("keys"):
                tables_without_keys.append(table)
        return tables_without_keys

    @staticmethod
    def _get_tables(config: Dict, key_type: str):
        """
        Return the list of related tables regarding the type of key -
        'primary key', 'foreign key', 'unique key'
        :param config: configuration of related tables declared in metadata.yaml file
        :param key_type:
        type of key either 'primary key' ('PK'), 'foreign key' ('FK'), 'unique key' ('UQ')
        """
        try:
            tables = [
                table_name
                for table_name, config in config.items()
                for key in config["keys"]
                if config["keys"][key]["type"] == key_type
            ]
            return list(dict.fromkeys(tables))
        except KeyError:
            raise KeyError(
                "The structure of metadata file seems to be invalid. "
                "Please check the presence of 'configuration', 'tables', "
                "'keys', 'type' fields in metadata file"
            )

    def _prepare_metadata_for_process(self, **kwargs):
        """
        Return the list of tables for training or infer process, the configuration of tables

        type_of_process can be "train", "infer" or "all" for the Enterprise version
        """
        config_of_tables = deepcopy(self.merged_metadata)
        if kwargs.get("type_of_process") in ("infer", "all"):
            config_of_tables = self._split_pk_fk_metadata(
                config_of_tables, list(config_of_tables.keys())
            )
        tables_without_keys = self._get_tables_without_keys(config_of_tables)
        pk_tables = self._get_tables(config_of_tables, "PK")
        uq_tables = self._get_tables(config_of_tables, "UQ")
        fk_tables = self._get_tables(config_of_tables, "FK")
        chain_of_tables = list(
            dict.fromkeys([*tables_without_keys, *pk_tables, *uq_tables, *fk_tables])
        )

        config_of_tables = {
            **self.metadata,
            **{k: v for k, v in config_of_tables.items() if k not in self.metadata},
        }
        chain_of_tables = [i for i in chain_of_tables for j in self.metadata if j in i]
        return chain_of_tables, config_of_tables

    @staticmethod
    def _split_keys(keys):
        """
        Split the keys into primary keys and unique keys, and foreign keys
        """
        pk_uq_keys = {k: v for k, v in keys.items() if v["type"] in ["PK", "UQ"]}
        fk_keys = {k: v for k, v in keys.items() if v["type"] == "FK"}
        return pk_uq_keys, fk_keys

    @staticmethod
    def _get_meta_copy(original_meta, new_keys):
        """
        Get updated metadata copy
        """
        meta_copy = original_meta.copy()
        meta_copy["keys"] = new_keys
        return meta_copy

    def _split_pk_fk_metadata(self, config, tables):
        """
        Split the metadata of tables into primary key and foreign key metadata
        """

        for table in tables:
            keys = config[table].get("keys", {})
            types_of_keys = {k_type["type"] for k_type in keys.values()}

            if "PK" in types_of_keys and "FK" in types_of_keys:
                self.divided.append(f"{table}_pk")
                self.divided.append(f"{table}_fk")

                pk_uq_keys, fk_keys = self._split_keys(keys)

                config[f"{table}_pk"] = self._get_meta_copy(config[table], pk_uq_keys)
                config[f"{table}_fk"] = self._get_meta_copy(config[table], fk_keys)

                config.pop(table)

        return config

    @staticmethod
    def _should_generate_data(
        config_of_tables: Dict,
        type_of_process: str
    ):
        """
        Determine whether the synthetic data should be generated
        in order to generate reports based on it

        Parameters:
        -----------
        config_of_tables: Dict
            Dictionary containing configuration for tables
        type_of_process: str
            Type of process ('train' or 'infer')

        Returns:
        --------
        bool
            True if any reports are configured for the specified process type
            except 'sample' report
        """

        list_of_reports = [
            config.get(f"{type_of_process}_settings", {}).get("reports", [])
            for config in config_of_tables.values()
        ]
        list_of_reports = [
            report for sublist in list_of_reports for report in sublist if report != "sample"
        ]
        return any(list_of_reports)

    def _collect_metrics_in_train(
        self,
        tables_for_training: List[str],
        tables_for_inference: List[str],
        generation_of_reports: bool
    ):
        """
        Collect the integral metrics for the training process
        """
        MlflowTracker().start_run(
            run_name="integral_metrics",
            tags={"process": "bottleneck"}
        )
        self._collect_integral_metrics(tables=tables_for_training, type_of_process="train")
        if generation_of_reports:
            self._collect_integral_metrics(tables=tables_for_inference, type_of_process="infer")
        MlflowTracker().end_run()

    def _save_input_data(self, data: pd.DataFrame, table_name: str):
        """
        Save the input data to the predefined path
        """
        fernet_key = self.metadata[table_name].get("encryption", {}).get("fernet_key", None)
        path_to_input_data = (
            f"model_artifacts/tmp_store/{slugify(table_name)}/"
            f"input_data_{slugify(table_name)}.{'dat' if fernet_key else 'pkl'}"
        )
        DataLoader(
            path=path_to_input_data,
            table_name=table_name,
            metadata=self.metadata,
            sensitive=True
        ).save_data(data)

    def _train_table(self, data, schema, table, metadata, delta):
        """"
        Train process for a single table
        """
        config_of_table = metadata[table]
        global_context(config_of_table.get("format", {}))
        train_settings = config_of_table["train_settings"]
        log_message = f"Training process of the table - '{table}' has started"
        logger.info(log_message)
        ProgressBarHandler().set_progress(delta=delta, message=log_message)

        TrainStrategy().run(
            data=data,
            schema=schema,
            metadata=metadata,
            metadata_path=self.metadata_path,
            source=train_settings.get("source"),
            epochs=train_settings["epochs"],
            drop_null=train_settings["drop_null"],
            row_limit=train_settings["row_limit"],
            table_name=table,
            reports=train_settings["reports"],
            batch_size=train_settings["batch_size"],
            loader=self.loader
        )
        self._save_metadata_file()
        ProgressBarHandler().set_progress(
            delta=delta,
            message=f"Training of the table - '{table}' was completed"
        )
        self._save_input_data(data=data, table_name=table)
        self._write_success_file(table_name=table, type_of_process="train")

    def __train_tables(
        self,
        tables_for_training: List,
        tables_for_inference: List,
        metadata_for_training: Dict,
        metadata_for_inference: Dict,
        generation_of_reports: bool
    ):
        """
        Run training process for the list of tables
        :param metadata_for_training:
        the tuple included the list of tables and metadata for training process
        :param metadata_for_inference:
        the tuple included the list of tables and metadata for inference process
        """
        delta = 0.49 / len(tables_for_training)

        for table in tables_for_training:
            data, schema = self.__preprocess_data(table_name=table)
            self._train_table(data, schema, table, metadata_for_training, delta)

        if generation_of_reports:
            self.__infer_tables(
                tables_for_inference,
                metadata_for_inference,
                delta,
                type_of_process="train"
            )

    def _get_surrogate_tables_mapping(self):
        """
        Get the mapping of surrogate tables, which end with "_pk" and "_fk",
        to the initial tables from which they were derived
        """
        return {
            table: [t for t in self.divided if t.startswith(table)]
            for table in self.initial_table_names
            if any(t.startswith(table) for t in self.divided)
        }

    def _find_parent_table(self, table):
        """
        Find the initial table
        from which the surrogate table was derived
        """
        return next(
            (
                parent
                for parent, children in self._get_surrogate_tables_mapping().items()
                if table in children
            ), None
        )

    def _infer_table(self, table, metadata, type_of_process, delta, is_nested=False):
        """
        Infer process for a single table
        """
        config_of_table = metadata[table]
        global_context(config_of_table.get("format", {}))
        log_message = f"Infer process of the table - '{table}' has started"
        logger.info(log_message)
        ProgressBarHandler().set_progress(delta=delta, message=log_message)
        both_keys = table in self.divided
        settings = config_of_table[f"{type_of_process}_settings"]

        MlflowTracker().start_run(
                run_name=f"{table}-INFER",
                tags={"table_name": table, "process": type_of_process},
                nested=is_nested,
        )
        InferStrategy().run(
            destination=settings.get("destination") if type_of_process == "infer" else None,
            metadata=metadata,
            metadata_path=self.metadata_path,
            size=settings.get("size") if type_of_process == "infer" else None,
            table_name=table,
            run_parallel=settings.get("run_parallel") if type_of_process == "infer" else False,
            batch_size=settings.get("batch_size") if type_of_process == "infer" else 1000,
            random_seed=settings.get("random_seed") if type_of_process == "infer" else 1,
            reports=settings["reports"],
            log_level=self.log_level,
            both_keys=both_keys,
            type_of_process=self.type_of_process,
            loader=self.loader
        )
        ProgressBarHandler().set_progress(
            delta=delta,
            message=f"Infer process of the table - '{table}' was completed"
        )
        MlflowTracker().end_run()
        self._write_success_file(table_name=table, type_of_process="infer", both_keys=both_keys)

    def __infer_tables(
        self,
        tables: List,
        config_of_tables: Dict,
        delta: float,
        type_of_process: str
    ):
        """
        Run infer process for the list of tables
        :param tables: the list of tables for infer process
        :param config_of_tables: configuration of tables declared in metadata file
        """

        non_surrogate_tables = [table for table in tables if table not in self.divided]

        for table in non_surrogate_tables:
            self._infer_table(
                table=table,
                metadata=config_of_tables,
                type_of_process=type_of_process,
                delta=delta
            )

        tables_mapping = self._get_surrogate_tables_mapping()
        for table_root in tables_mapping.keys():
            MlflowTracker().start_run(
                run_name=f"{table_root}-INFER",
                tags={"table_name": table_root, "process": type_of_process}
            )
            for table in tables_mapping[table_root]:
                self._infer_table(
                    table=table,
                    metadata=config_of_tables,
                    type_of_process=type_of_process,
                    delta=delta,
                    is_nested=True
                )
            MlflowTracker().end_run()

    def _collect_integral_metrics(self, tables, type_of_process):
        """
        Collect the integral metrics depending on the type of process
        """
        stages = self.train_stages if type_of_process == "train" else self.infer_stages
        for table, stage in product(tables, stages):
            MlflowTracker().collect_metrics(table, stage)

    def _generate_reports(self):
        """
        Generate reports
        """
        Report().generate_report()
        Report().clear_report()

    @staticmethod
    def _write_success_file(
        table_name: str, type_of_process: Literal["train", "infer"], both_keys: bool = False
    ):
        """
        Write a success message to the '.success' file based on the type of the process
        """
        dynamic_name = slugify(table_name)[:-3] if both_keys else slugify(table_name)

        file_paths = {
            "train": f"model_artifacts/resources/{dynamic_name}/train_message.success",
            "infer": f"model_artifacts/tmp_store/{dynamic_name}/infer_message.success",
        }

        with open(file_paths[type_of_process], "w") as success_file:
            success_file.write("SUCCESS")

    def _save_metadata_file(self):
        if self.metadata_path:
            os.makedirs("model_artifacts/metadata", exist_ok=True)
            metadata_file_name = os.path.basename(self.metadata_path)
            MetadataLoader(
                path=f"model_artifacts/metadata/{metadata_file_name}"
            ).save_data(metadata=self.metadata)

    def launch_train(self):
        """
        Launch training process either for a single table or for several tables
        """
        metadata_for_training = self._prepare_metadata_for_process(type_of_process="train")
        metadata_for_inference = self._prepare_metadata_for_process(type_of_process="infer")

        (
            tables_for_training,
            metadata_for_training,
        ) = metadata_for_training
        (
            tables_for_inference,
            metadata_for_inference,
        ) = metadata_for_inference

        generation_of_reports = self._should_generate_data(
            metadata_for_training,
            "train"
        )

        self.__train_tables(
            tables_for_training,
            tables_for_inference,
            metadata_for_training,
            metadata_for_inference,
            generation_of_reports
        )

        self._generate_reports()
        self._collect_metrics_in_train(
            tables_for_training,
            tables_for_inference,
            generation_of_reports
        )

    def _collect_metrics_in_infer(self, tables):
        """
        Collect the integral metrics for the inference process
        """
        MlflowTracker().start_run(
            run_name="integral_metrics",
            tags={"process": "bottleneck"}
        )
        self._collect_integral_metrics(tables, type_of_process="infer")
        MlflowTracker().end_run()

    def launch_infer(self):
        """
        Launch infer process either for a single table or for several tables
        """
        tables, config_of_tables = self._prepare_metadata_for_process(type_of_process="infer")

        generation_of_reports = self._should_generate_data(
            config_of_tables,
            "infer"
        )
        delta = 0.25 / len(tables) if generation_of_reports else 0.5 / len(tables)

        self.__infer_tables(tables, config_of_tables, delta, type_of_process="infer")
        self._generate_reports()
        self.__postprocess_data()
        self._collect_metrics_in_infer(tables)
