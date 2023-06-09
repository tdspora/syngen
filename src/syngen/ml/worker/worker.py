from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy

from syngen.ml.data_loaders import MetadataLoader
from syngen.ml.strategies import TrainStrategy, InferStrategy
from syngen.ml.reporters import Report
from syngen.ml.custom_logger import custom_logger

@dataclass
class Worker:
    """
    Class for preparing training and infer settings, metadata for training and infer process
    """
    table_name: Optional[str]
    metadata_path: Optional[str]
    settings: Dict
    log_level: str
    type: str
    train_strategy = TrainStrategy()
    infer_strategy = InferStrategy()
    metadata = None
    divided = []

    def __post_init__(self):
        self.metadata = self.__fetch_metadata()


    def _update_metadata_for_table(self, metadata: Dict) -> Dict:
        """
        Update the metadata for training or inference process if a metadata file wasn't provided
        """
        if "source" in self.settings.keys():
            del self.settings["source"]
        if self.type == "train":
            train_settings = metadata[self.table_name]["train_settings"]
            train_settings.update(self.settings)
        elif self.type == "infer":
            infer_settings = metadata[self.table_name]["infer_settings"]
            infer_settings.update(self.settings)
        return metadata

    def _update_metadata_for_tables(self, metadata: Dict) -> Dict:
        """
        Update the metadata for training or inference process if a metadata file was provided
        """
        if "source" in self.settings.keys():
            del self.settings["source"]
        if self.type == "train":
            for table in metadata.keys():
                train_settings = metadata[table]["train_settings"]
                train_settings.update({
                    setting: value for setting, value in self.settings.items()
                    if setting not in train_settings
                })
            return metadata
        elif self.type == "infer":
            for key in metadata.keys():
                infer_settings = metadata[key]["infer_settings"]
                infer_settings.update({
                    setting: value for setting, value in self.settings.items()
                    if setting not in infer_settings
                })
            return metadata

    def __fetch_metadata(self) -> Dict[str, str]:
        """
        Fetch the metadata for training or infer process
        """
        metadata = MetadataLoader(self.metadata_path).load_data() if self.metadata_path else None
        source = self.settings.get("source")
        # Set a metadata for training or infer process if a metadata file wasn't provided
        if self.table_name:
            metadata = {
                self.table_name: {
                    "train_settings": {},
                    "infer_settings": {},
                    "keys": {},
                    "source": source
                }
            }
            metadata = self._update_metadata_for_table(metadata)
            return metadata
        # Update a metadata for training or infer process if a metadata file was provided
        if self.metadata_path:
            metadata = self._update_metadata_for_tables(metadata)
            return metadata
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
        :param key_type: type of key either 'primary key' ('PK'), 'foreign key' ('FK'), 'unique key' ('UQ')
        """
        try:
            tables = [table_name for table_name, config in config.items()
                      for key in config["keys"]
                      if config["keys"][key]["type"] == key_type]
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
        config_of_tables = deepcopy(self.metadata)
        tables_without_keys = self._get_tables_without_keys(config_of_tables)
        if kwargs.get("type_of_process") in ("infer", "all"):
            config_of_tables = self._split_pk_fk_metadata(config_of_tables, list(config_of_tables.keys()))
        pk_tables = self._get_tables(config_of_tables, "PK")
        uq_tables = self._get_tables(config_of_tables, "UQ")
        fk_tables = self._get_tables(config_of_tables, "FK")
        chain_of_tables = list(dict.fromkeys([*tables_without_keys, *pk_tables, *uq_tables, *fk_tables]))

        return chain_of_tables, config_of_tables

    def _split_pk_fk_metadata(self, config, tables):
        for table in tables:
            types_of_keys = [value["type"] for key, value in config[table]["keys"].items()]
            if "PK" in types_of_keys and "FK" in types_of_keys:
                self.divided += [table+"_pk", table+"_fk"]
                pk_uq_part = {key: value for key, value in config[table]["keys"].items() if value["type"] in ["PK", "UQ"]}
                fk_part = {key: value for key, value in config[table]["keys"].items() if value["type"] == "FK"}

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
            metadata_for_inference: Tuple[List, Dict]
    ):
        """
        Run training process for the list of tables
        :param metadata_for_training: the tuple included the list of tables and metadata for training process
        :param metadata_for_inference: the tuple included the list of tables and metadata for inference process
        """
        chain_for_tables_for_training, config_of_metadata_for_training = metadata_for_training
        chain_for_tables_for_inference, config_of_metadata_for_inference = metadata_for_inference

        for table in chain_for_tables_for_training:
            config_of_table = config_of_metadata_for_training[table]
            source = config_of_table["source"]
            train_settings = config_of_table["train_settings"]
            custom_logger.info(f"Training process of the table - {table} has started.")

            self.train_strategy.run(
                metadata=self.metadata,
                source=source,
                epochs=train_settings["epochs"],
                drop_null=train_settings["drop_null"],
                row_limit=train_settings["row_limit"],
                table_name=table,
                metadata_path=self.metadata_path,
                print_report=train_settings["print_report"],
                batch_size=train_settings["batch_size"]
            )
        generation_of_reports = any(
            [
                config.get("train_settings", {}).get("print_report", False)
                for table, config in config_of_metadata_for_training.items()
            ]
        )
        self.metadata = config_of_metadata_for_inference
        if generation_of_reports:
            for table in chain_for_tables_for_inference:
                config_of_table = config_of_metadata_for_inference[table]
                train_settings = config_of_table["train_settings"]
                print_report = train_settings.get("print_report")
                both_keys = table in self.divided

                custom_logger.info(f"Infer process of the table - {table} has started")

                self.infer_strategy.run(
                    metadata=self.metadata,
                    size=None,
                    table_name=table,
                    metadata_path=self.metadata_path,
                    run_parallel=False,
                    batch_size=1000,
                    random_seed=1,
                    print_report=print_report,
                    log_level=self.log_level,
                    both_keys=both_keys
                )

    def __infer_tables(self, tables: List, config_of_tables: Dict):
        """
        Run infer process for the list of tables
        :param tables: the list of tables for infer process
        :param config_of_tables: configuration of tables declared in metadata file
        """
        for table in tables:
            custom_logger.info(f"Infer process of the table - {table} has started")
            both_keys = table in self.divided
            config_of_table = config_of_tables[table]
            infer_settings = config_of_table["infer_settings"]

            self.infer_strategy.run(
                metadata=self.metadata,
                size=infer_settings["size"],
                table_name=table,
                metadata_path=self.metadata_path,
                run_parallel=infer_settings["run_parallel"],
                batch_size=infer_settings["batch_size"],
                random_seed=infer_settings["random_seed"],
                print_report=infer_settings["print_report"],
                log_level=self.log_level,
                both_keys=both_keys
            )

    @staticmethod
    def _generate_reports():
        """
        Generate reports
        """
        Report().generate_report()
        Report().clear_report()

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
        chain_of_tables, config_of_tables = self._prepare_metadata_for_process(type_of_process="infer")
        self.metadata = config_of_tables
        self.__infer_tables(chain_of_tables, config_of_tables)
        self._generate_reports()
