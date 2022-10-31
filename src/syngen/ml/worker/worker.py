from typing import Dict, List, Optional
from dataclasses import dataclass

from loguru import logger

from syngen.ml.data_loaders import MetadataLoader
from syngen.ml.interface import TrainInterface, InferInterface


@dataclass
class Worker:
    """
    Class for preparing training and infer settings, metadata for training and infer process
    """
    table_name: str
    metadata_path: Optional[str]
    settings: Dict
    train_interface = TrainInterface()
    infer_interface = InferInterface()
    metadata_loader = MetadataLoader()
    metadata = None

    def __post_init__(self):
        self.metadata = self.metadata_loader.load_data(self.metadata_path) if self.metadata_path else None

    def __parse_train_settings(self, config: Dict):
        """
        Parse the settings for training process
        :param config: settings for training process declared in metadata.yml file
        """
        epochs = config.get("train_settings", {}).get("epochs", self.settings.get("epochs"))
        drop_null = config.get("train_settings", {}).get("drop_null", self.settings.get("drop_null"))
        row_limit = config.get("train_settings", {}).get("row_limit", self.settings.get("row_limit"))
        batch_size = config.get("train_settings", {}).setdefault("batch_size", 32)
        return {
            "table_name": self.table_name,
            "epochs": epochs,
            "drop_null": drop_null,
            "row_limit": row_limit,
            "batch_size": batch_size
        }

    def __parse_infer_settings(self, config: Dict):
        """
        Parse the settings for infer process
        :param config: settings for infer process declared in metadata.yml file
        """
        size = config.get("infer_settings", {}).get("size", self.settings.get("size"))
        if size is None:
            raise AttributeError(
                "The size is mandatory parameter. "
                "It seems that the information of size for infer process is absent. "
                "Please provide the information of size either through parameter in CLI command "
                "or in size parameter in metadata file."
            )
        run_parallel = config.get("infer_settings", {}).get("run_parallel", self.settings.get("run_parallel"))
        random_seed = config.get("infer_settings", {}).get("random_seed", self.settings.get("random_seed"))
        print_report = config.get("infer_settings", {}).get("print_report", self.settings.get("print_report"))
        batch_size = config.get("infer_settings", {}).get("batch_size", self.settings.get("batch_size"))
        return {
            "size": size,
            "run_parallel": run_parallel,
            "random_seed": random_seed,
            "print_report": print_report,
            "batch_size": batch_size
        }

    @staticmethod
    def _get_tables(config: Dict, key_type: str):
        """
        Return the list of related tables regarding to the type of key - 'primary key', 'foreign key'
        :param config: configuration of related tables declared in metadata.yml file
        :param key_type: type of key either 'primary key' ('PK') or 'foreign key' ('FK')
        """
        try:
            tbls = [table_name for table_name, config in config.items()
                    for key in config["keys"]
                    if config["keys"][key]["type"] == key_type]
            return list(dict.fromkeys(tbls))
        except KeyError:
            raise KeyError(
                "The structure of metadata file seems to be invalid. "
                "Please check the presence of 'configuration', 'tables', "
                "'keys', 'type' fields in metadata file"
            )

    def _prepare_metadata_for_process(self):
        """
        Return the list of related tables for training or infer process,
        configuration of related tables
        """
        config_of_tables = self.metadata
        pk_tables = self._get_tables(config_of_tables, "PK")
        fk_tables = self._get_tables(config_of_tables, "FK")
        # chain_of_tables = [*pk_tables, *list(set(fk_tables).difference(set(pk_tables)))]
        chain_of_tables = [*pk_tables, *fk_tables]
        return chain_of_tables, config_of_tables

    def __train_chain_of_tables(self, tables: List, config_of_tables: Dict):
        """
        Run training process for the list of related tables
        :param tables: the list of related tables for training process
        :param config_of_tables: configuration of related tables declared in metadata.yml file
        """
        for table in tables:
            config_of_table = config_of_tables[table]
            source = config_of_table.get("source", None)
            if source is None:
                raise AttributeError(
                    f"The source of table - {table} is mandatory parameter. "
                    f"It seems that the information of source for training is absent. "
                    f"Please provide the information of source in metadata file."
                )
            train_settings = self.__parse_train_settings(config_of_table)
            logger.info(f"Training process of the table - {table} has started.")
            self.train_interface.run(
                metadata=self.metadata,
                source=source,
                epochs=train_settings["epochs"],
                drop_null=train_settings["drop_null"],
                row_limit=train_settings["row_limit"],
                table_name=table,
                metadata_path=self.metadata_path,
                batch_size=train_settings["batch_size"]
            )

    def __infer_chain_of_tables(self, tables: List, config_of_tables: Dict):
        """
        Run infer process for the list of related tables
        :param tables: the list of related tables for infer process
        :param config_of_tables: configuration of related tables declared in metadata.yml file
        """
        for table in tables:
            config_of_table = config_of_tables[table]
            infer_settings = self.__parse_infer_settings(config_of_table)
            logger.info(f"Infer process of the table - {table} has started.")
            self.infer_interface.run(
                metadata=self.metadata,
                size=infer_settings["size"],
                table_name=table,
                run_parallel=infer_settings["run_parallel"],
                batch_size=infer_settings["batch_size"],
                metadata_path=self.metadata_path,
                random_seed=infer_settings["random_seed"],
                print_report=infer_settings["print_report"]
            )

    def __train_table(self):
        """
        Run training process for a single table
        :return:
        """
        logger.info(f"Training process of the table - {self.table_name} has started.")
        self.train_interface.run(
            metadata=self.metadata,
            source=self.settings.get("source"),
            epochs=self.settings.get("epochs"),
            drop_null=self.settings.get("drop_null"),
            row_limit=self.settings.get("row_limit"),
            table_name=self.table_name,
            metadata_path=self.metadata_path,
            batch_size=self.settings.get("batch_size")
        )

    def __infer_table(self):
        """
        Run infer process for a single table
        """
        if self.settings.get("size") is None:
            raise AttributeError(
                "The size is mandatory parameter. "
                "It seems that the information of size for infer process is absent. "
                "Please provide the information of size through parameter in CLI command."
            )
        logger.info(f"Infer process of the table - {self.table_name} has started.")
        self.infer_interface.run(
            metadata=self.metadata,
            size=self.settings.get("size"),
            table_name=self.table_name,
            metadata_path=self.metadata_path,
            run_parallel=self.settings.get("run_parallel"),
            batch_size=self.settings.get("batch_size"),
            random_seed=self.settings.get("random_seed"),
            print_report=self.settings.get("print_report")
        )

    def launch_train(self):
        """
        Launch training process either for a single table or for related tables
        """
        if self.metadata_path is not None:
            chain_of_tables, config_of_tables = self._prepare_metadata_for_process()
            self.__train_chain_of_tables(chain_of_tables, config_of_tables)
        if self.table_name is not None:
            self.__train_table()

    def launch_infer(self):
        """
        Launch infer process either for a single table or for related tables
        """
        if self.metadata_path is not None:
            chain_of_tables, config_of_tables = self._prepare_metadata_for_process()
            self.__infer_chain_of_tables(chain_of_tables, config_of_tables)
        if self.table_name is not None:
            self.__infer_table()
