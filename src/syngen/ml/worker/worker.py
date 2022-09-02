from typing import Dict, List, Optional
from loguru import logger

from syngen.ml.data_loaders import MetadataLoader
from syngen.ml.interface import TrainInterface, InferInterface


class Worker:
    """
    Class for preparing training and infer settings, metadata for training and infer process
    """
    def __init__(self, table_name: Optional[str], metadata_path: str, settings: Dict):
        self.table_name = table_name
        self.metadata_path = metadata_path
        self.settings = settings
        self.train_interface = TrainInterface()
        self.infer_interface = InferInterface()
        self.metadata_loader = MetadataLoader()

    def _parse_train_settings(self, config: Dict):
        """
        Parse the settings for training process
        :param config: settings for training process declared in metadata.yml file
        """
        try:
            path = self.settings.get("path")
            epochs = config.get("train_settings", {}).get("epochs", self.settings.get("epochs"))
            dropna = config.get("train_settings", {}).get("dropna", self.settings.get("dropna"))
            row_limit = config.get("train_settings", {}).get("row_limit", self.settings.get("row_limit"))
            batch_size = config.get("train_settings", {}).setdefault("batch_size", 32)
            return {
                "path": path,
                "table_name": self.table_name,
                "epochs": epochs,
                "dropna": dropna,
                "row_limit": row_limit,
                "batch_size": batch_size
            }
        except KeyError:
            logger.info("The values of parameters for training process are set to default values.")

    def _parse_infer_settings(self, config: Dict):
        """
        Parse the settings for infer process
        :param config: settings for infer process declared in metadata.yml file
        """
        try:
            size = config.get("infer_settings", {}).get("size", self.settings.get("size"))
            if size is None:
                raise AttributeError(
                    f"The size is mandatory parameter. "
                    f"It seems that the information of size for infer process is absent. "
                    f"Please provide the information of size either through parameter in CLI command "
                    f"or in size parameter in metadata file."
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
        except KeyError:
            logger.info("The values of parameters for infer process are set to default values.")

    @staticmethod
    def _get_tables(config: Dict, key_type: str):
        """
        Return the list of related tables regarding to the type of key - 'primary key', 'foreign key'
        :param config: configuration of related tables declared in metadata.yml file
        :param key_type: type of key either 'primary key' ('PK') or 'foreign key' ('FK')
        """
        try:
            return [table_name for table_name, config in config["configuration"]["tables"].items()
                    for key in config["keys"]
                    if config["keys"][key]["type"] == key_type]
        except KeyError:
            raise KeyError(
                "The structure of metadata file seems to be invalid. "
                "Please check the presence of 'configuration', 'tables', "
                "'keys', 'type' fields in metadata file"
            )

    @staticmethod
    def _parse_tables_config(config: Dict):
        """
        Return the configuration of related tables
        :param config: configuration of related tables declared in metadata.yml file
        """
        try:
            return config["configuration"]["tables"]
        except KeyError:
            raise KeyError(
                "The information of tables seems to be absent in metadata file. "
                "Please provide the information of tables."
            )

    def _prepare_metadata_for_process(self):
        """
        Return the list of related tables for training or infer process,
        configuration of related tables
        """
        metadata = self.metadata_loader.load_data(self.metadata_path)
        config_of_tables = self._parse_tables_config(metadata)
        pk_tables = self._get_tables(metadata, "PK")
        fk_tables = self._get_tables(metadata, "FK")
        # chain_of_tables = [*pk_tables, *list(set(fk_tables).difference(set(pk_tables)))]
        chain_of_tables = [*pk_tables, *fk_tables]
        return chain_of_tables, config_of_tables

    def _train_chain_of_tables(self, tables: List, config_of_tables: Dict):
        """
        Run training process for the list of related tables
        :param tables: the list of related tables for training process
        :param config_of_tables: configuration of related tables declared in metadata.yml file
        """
        for table in tables:
            config_of_table = config_of_tables[table]
            path = config_of_table.get("path", None)
            if path is None:
                raise AttributeError(
                    f"The path of table - {table} is mandatory parameter. "
                    f"It seems that the information of path for training is absent. "
                    f"Please provide the information of path in metadata file."
                )
            train_settings = self._parse_train_settings(config_of_table)
            self.train_interface.run(
                path=path,
                epochs=train_settings["epochs"],
                dropna=train_settings["dropna"],
                row_limit=train_settings["row_limit"],
                table_name=table,
                metadata_path=self.metadata_path,
                batch_size=train_settings["batch_size"]
            )

    def _infer_chain_of_tables(self, tables: List, config_of_tables: Dict):
        """
        Run infer process for the list of related tables
        :param tables: the list of related tables for infer process
        :param config_of_tables: configuration of related tables declared in metadata.yml file
        """
        for table in tables:
            config_of_table = config_of_tables[table]
            infer_settings = self._parse_infer_settings(config_of_table)
            self.infer_interface.run(
                size=infer_settings["size"],
                table_name=table,
                run_parallel=infer_settings["run_parallel"],
                batch_size=infer_settings["batch_size"],
                metadata_path=self.metadata_path,
                random_seed=infer_settings["random_seed"],
                print_report=infer_settings["print_report"]
            )

    def _train_table(self):
        """
        Run training process for a single table
        :return:
        """
        self.train_interface.run(
            path=self.settings.get("path"),
            epochs=self.settings.get("epochs"),
            dropna=self.settings.get("dropna"),
            row_limit=self.settings.get("row_limit"),
            table_name=self.table_name,
            metadata_path=self.metadata_path,
            batch_size=self.settings.get("batch_size")
        )

    def _infer_table(self):
        """
        Run infer process for a single table
        """
        if self.settings.get("size") is None:
            raise AttributeError(
                f"The size is mandatory parameter. "
                f"It seems that the information of size for infer process is absent. "
                f"Please provide the information of size through parameter in CLI command."
            )
        self.infer_interface.run(
            size=self.settings.get("size"),
            table_name=self.settings.get("table_name"),
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
            self._train_chain_of_tables(chain_of_tables, config_of_tables)
        if self.table_name is not None:
            self._train_table()

    def launch_infer(self):
        """
        Launch infer process either for a single table or for related tables
        """
        if self.metadata_path:
            chain_of_tables, config_of_tables = self._prepare_metadata_for_process()
            self._infer_chain_of_tables(chain_of_tables, config_of_tables)
        if self.table_name is not None:
            self._infer_table()
