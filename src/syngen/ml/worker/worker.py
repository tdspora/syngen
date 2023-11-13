from typing import Dict, List, Optional, Tuple, Any
from attrs import define, field
from copy import deepcopy
from loguru import logger
from slugify import slugify
import os

from syngen.ml.data_loaders import MetadataLoader
from syngen.ml.strategies import TrainStrategy, InferStrategy
from syngen.ml.reporters import Report
from syngen.ml.config import Validator
from syngen.ml.mlflow_tracker import MlflowTrackerFactory
from syngen.ml.context.context import global_context


@define
class Worker:
    """
    Class for preparing training and infer settings, metadata for training and infer process
    """

    table_name: Optional[str] = field(kw_only=True)
    metadata_path: Optional[str] = field(kw_only=True)
    settings: Dict = field(kw_only=True)
    log_level: str = field(kw_only=True)
    type_of_process: str = field(kw_only=True)
    train_strategy = TrainStrategy()
    infer_strategy = InferStrategy()
    metadata: Optional[Dict] = None
    divided: List = field(default=list())
    merged_metadata: Dict = field(default=dict())

    def __attrs_post_init__(self):
        os.makedirs("model_artifacts/metadata", exist_ok=True)
        self.metadata = self.__fetch_metadata()
        self._update_metadata()
        self.__validate_metadata()
        self._set_mlflow()

    def __validate_metadata(self):
        """
        Validate the metadata, set the merged metadata
        """
        validator = Validator(
            metadata=self.metadata,
            metadata_path=self.metadata_path,
            type_of_process=self.type_of_process,
        )
        validator.run()
        self.merged_metadata = validator.merged_metadata

    def _set_mlflow(self):
        """
        Set the mlflow experiment name and the mlflow run
        """
        MlflowTrackerFactory.create_tracker(
            table_name=self.table_name,
            metadata_path=self.metadata_path,
            type_of_process=self.type_of_process
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

        for table, table_metadata in self.metadata.items():
            if table == "global":
                continue
            elif self.type_of_process == "train":
                settings_key = "train_settings"
                global_settings = global_train_settings
            elif self.type_of_process == "infer":
                settings_key = "infer_settings"
                global_settings = global_infer_settings
            else:
                continue

            table_settings = table_metadata[settings_key]

            self._update_table_settings(table_settings, global_settings)
            self._update_table_settings(table_settings, self.settings)

    def _update_metadata(self) -> None:
        if self.metadata_path:
            self._update_metadata_for_tables()
        if self.table_name:
            self._update_metadata_for_table()

    def __fetch_metadata(self) -> Dict:
        """
        Fetch the metadata for training or infer process
        """
        if self.metadata_path:
            metadata = MetadataLoader(self.metadata_path).load_data()
            return metadata
        if self.table_name:
            source = self.settings.get("source", "absent")
            # Set a metadata for training or infer process if a metadata file wasn't provided
            metadata = {
                self.table_name: {
                    "train_settings": {
                        "source": source,
                    },
                    "infer_settings": {},
                    "keys": {},
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

    def _split_pk_fk_metadata(self, config, tables):
        for table in tables:
            types_of_keys = [value["type"] for key, value in config[table].get("keys", {}).items()]
            if "PK" in types_of_keys and "FK" in types_of_keys:
                self.divided += [table + "_pk", table + "_fk"]
                pk_uq_part = {
                    key: value
                    for key, value in config[table]["keys"].items()
                    if value["type"] in ["PK", "UQ"]
                }
                fk_part = {
                    key: value
                    for key, value in config[table]["keys"].items()
                    if value["type"] == "FK"
                }

                # Do this to create a new object instead of a reference
                as_pk_meta = {k: v for k, v in config[table].items()}
                as_fk_meta = {k: v for k, v in config[table].items()}

                as_pk_meta["keys"] = pk_uq_part
                as_fk_meta["keys"] = fk_part

                config[table + "_pk"] = as_pk_meta
                config[table + "_fk"] = as_fk_meta
                config.pop(table)
        return config

    def __train_tables(
        self,
        metadata_for_training: Tuple[List, Dict],
        metadata_for_inference: Tuple[List, Dict],
    ):
        """
        Run training process for the list of tables
        :param metadata_for_training:
        the tuple included the list of tables and metadata for training process
        :param metadata_for_inference:
        the tuple included the list of tables and metadata for inference process
        """
        (
            chain_for_tables_for_training,
            config_of_metadata_for_training,
        ) = metadata_for_training
        (
            chain_for_tables_for_inference,
            config_of_metadata_for_inference,
        ) = metadata_for_inference

        for table in chain_for_tables_for_training:
            config_of_table = config_of_metadata_for_training[table]
            global_context(config_of_table.get("format", {}))
            train_settings = config_of_table["train_settings"]
            logger.info(f"Training process of the table - {table} has started.")

            self.train_strategy.run(
                metadata=config_of_metadata_for_training,
                source=train_settings["source"],
                epochs=train_settings["epochs"],
                drop_null=train_settings["drop_null"],
                row_limit=train_settings["row_limit"],
                table_name=table,
                metadata_path=self.metadata_path,
                print_report=train_settings["print_report"],
                batch_size=train_settings["batch_size"],
            )
            self._write_success_message(slugify(table))
            self._save_metadata_file()
        generation_of_reports = any(
            [
                config.get("train_settings", {}).get("print_report", False)
                for table, config in config_of_metadata_for_training.items()
            ]
        )
        if generation_of_reports:
            for table in chain_for_tables_for_inference:
                config_of_table = config_of_metadata_for_inference[table]
                global_context(config_of_table.get("format", {}))
                train_settings = config_of_table["train_settings"]
                print_report = train_settings.get("print_report")
                both_keys = table in self.divided

                logger.info(f"Infer process of the table - {table} has started")

                self.infer_strategy.run(
                    destination=None,
                    metadata=config_of_metadata_for_inference,
                    size=None,
                    table_name=table,
                    metadata_path=self.metadata_path,
                    run_parallel=False,
                    batch_size=1000,
                    random_seed=1,
                    print_report=print_report,
                    log_level=self.log_level,
                    both_keys=both_keys,
                    type=self.type_of_process,
                )

    def __infer_tables(self, tables: List, config_of_tables: Dict):
        """
        Run infer process for the list of tables
        :param tables: the list of tables for infer process
        :param config_of_tables: configuration of tables declared in metadata file
        """
        for table in tables:
            config_of_table = config_of_tables[table]
            global_context(config_of_table.get("format", {}))
            logger.info(f"Infer process of the table - {table} has started")
            both_keys = table in self.divided
            infer_settings = config_of_table["infer_settings"]

            self.infer_strategy.run(
                destination=infer_settings.get("destination"),
                metadata=config_of_tables,
                size=infer_settings["size"],
                table_name=table,
                metadata_path=self.metadata_path,
                run_parallel=infer_settings["run_parallel"],
                batch_size=infer_settings["batch_size"],
                random_seed=infer_settings["random_seed"],
                print_report=infer_settings["print_report"],
                log_level=self.log_level,
                both_keys=both_keys,
                type=self.type_of_process,
            )

    @staticmethod
    def _generate_reports():
        """
        Generate reports
        """
        Report().generate_report()
        Report().clear_report()

    @staticmethod
    def _write_success_message(table_name: str):
        """
        Write success message to the '.success' file
        """
        with open(f"model_artifacts/resources/{table_name}/message.success", "w") as f:
            f.write("SUCCESS")

    def _save_metadata_file(self):
        if self.metadata_path:
            os.makedirs("model_artifacts/metadata", exist_ok=True)
            metadata_file_name = os.path.basename(self.metadata_path)
            MetadataLoader(self.metadata_path).save_data(
                f"model_artifacts/metadata/{metadata_file_name}", self.metadata
            )

    def launch_train(self):
        """
        Launch training process either for a single table or for several tables
        """
        metadata_for_training = self._prepare_metadata_for_process()
        metadata_for_inference = self._prepare_metadata_for_process(type_of_process="infer")
        self.__train_tables(metadata_for_training, metadata_for_inference)
        self._generate_reports()

    def launch_infer(self):
        """
        Launch infer process either for a single table or for several tables
        """
        chain_of_tables, config_of_tables = self._prepare_metadata_for_process(
            type_of_process="infer"
        )
        self.__infer_tables(chain_of_tables, config_of_tables)
        self._generate_reports()
