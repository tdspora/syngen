from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy

from loguru import logger

from syngen.ml.data_loaders import MetadataLoader
from syngen.ml.strategies import TrainStrategy, InferStrategy
from syngen.ml.reporters import Report


@dataclass
class Worker:
    """
    Class for preparing training and infer settings, metadata for training and infer process
    """
    table_name: str
    metadata_path: Optional[str]
    settings: Dict
    train_strategy = TrainStrategy()
    infer_strategy = InferStrategy()
    metadata = None
    divided = []

    def __post_init__(self):
        self.metadata_loader = MetadataLoader(self.metadata_path)
        self.metadata = self.metadata_loader.load_data() if self.metadata_path else None

    def _extract_setting(self, params, setting):
        """
        Extract the value of the certain setting
        """
        return params.get(setting) if params.get(setting) is not None else self.settings.get(setting)

    def __parse_train_settings(self, config: Dict):
        """
        Parse the settings for training process
        :param config: settings for training process declared in metadata.yaml file
        """
        train_settings = config.get("train_settings", {})

        epochs = self._extract_setting(train_settings, setting="epochs")
        drop_null = self._extract_setting(train_settings, setting="drop_null")
        row_limit = self._extract_setting(train_settings, setting="row_limit")
        print_report = self._extract_setting(train_settings, setting="print_report")
        batch_size = self._extract_setting(train_settings, setting="batch_size")

        return {
            "table_name": self.table_name,
            "epochs": epochs,
            "drop_null": drop_null,
            "row_limit": row_limit,
            "print_report": print_report,
            "batch_size": batch_size
        }

    def __parse_infer_settings(self, config: Dict):
        """
        Parse the settings for infer process
        :param config: settings for infer process declared in metadata.yaml file
        """
        infer_settings = config.get("infer_settings", {})

        size = self._extract_setting(infer_settings, "size")
        run_parallel = self._extract_setting(infer_settings, "run_parallel")
        random_seed = self._extract_setting(infer_settings, "random_seed")
        batch_size = self._extract_setting(infer_settings, "batch_size")
        print_report = self._extract_setting(infer_settings, "print_report")

        return {
            "size": size,
            "run_parallel": run_parallel,
            "random_seed": random_seed,
            "batch_size": batch_size,
            "print_report": print_report
        }

    @staticmethod
    def _get_tables(config: Dict, key_type: str):
        """
        Return the list of related tables regarding to the type of key - 'primary key', 'foreign key'
        :param config: configuration of related tables declared in metadata.yaml file
        :param key_type: type of key either 'primary key' ('PK') or 'foreign key' ('FK')
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
        Return the list of related tables for training or infer process,
        configuration of related tables

        type_of_process can be "train", "infer" or "all" for the Enterprise version
        """
        config_of_tables = deepcopy(self.metadata)
        table_names = list(config_of_tables.keys())
        if len(table_names) == 1 and "keys" not in config_of_tables[table_names[0]]:
            # case with one table without any keys
            chain_of_tables = table_names
        else:
            if kwargs.get("type_of_process") in ("infer", "all"):
                config_of_tables = self._split_pk_fk_metadata(config_of_tables, list(config_of_tables.keys()))
            pk_tables = self._get_tables(config_of_tables, "PK")
            fk_tables = self._get_tables(config_of_tables, "FK")
            chain_of_tables = list(dict.fromkeys([*pk_tables, *fk_tables]))

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

    def __train_chain_of_tables_with_generation(
            self,
            metadata_for_training: Tuple[List, Dict],
            metadata_for_inference: Tuple[List, Dict]
    ):
        """
        Run training process for the list of related tables
        :param metadata_for_training: the tuple included the list of related tables and metadata for training process
        :param metadata_for_inference: the tuple included the list of related tables and metadata for inference process
        """
        chain_for_tables_for_training, config_of_metadata_for_training = metadata_for_training
        chain_for_tables_for_inference, config_of_metadata_for_inference = metadata_for_inference

        for table in chain_for_tables_for_training:
            config_of_table = config_of_metadata_for_training[table]
            source = config_of_table.get("source", None)
            if source is None:
                raise AttributeError(
                    f"The source of table - {table} is mandatory parameter. "
                    f"It seems that the information of source for training is absent. "
                    f"Please provide the information of source in metadata file."
                )
            train_settings = self.__parse_train_settings(config_of_table)
            self.__train_table(
                source=source,
                epochs=train_settings.get("epochs"),
                drop_null=train_settings.get("drop_null"),
                row_limit=train_settings.get("row_limit"),
                table_name=table,
                print_report=train_settings.get("print_report"),
                batch_size=train_settings.get("batch_size")
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
                train_settings = self.__parse_train_settings(config_of_table)
                print_report = train_settings.get("print_report")
                both_keys = table in self.divided

                self.__infer_table(
                    table_name=table,
                    run_parallel=False,
                    batch_size=1000,
                    random_seed=1,
                    print_report=print_report,
                    both_keys=both_keys
                )

    def __infer_chain_of_tables(self, tables: List, config_of_tables: Dict):
        """
        Run infer process for the list of related tables
        :param tables: the list of related tables for infer process
        :param config_of_tables: configuration of related tables declared in metadata.yaml file
        """
        for table in tables:
            both_keys = table in self.divided
            config_of_table = config_of_tables[table]
            infer_settings = self.__parse_infer_settings(config_of_table)
            self.__infer_table(
                size=infer_settings.get("size"),
                table_name=table,
                run_parallel=infer_settings.get("run_parallel"),
                batch_size=infer_settings.get("batch_size"),
                random_seed=infer_settings.get("random_seed"),
                print_report=infer_settings.get("print_report"),
                both_keys=both_keys
            )

    def __train_table_with_generation(self):
        """
        Run training process for a single table
        """
        self.__train_table()

        if self.settings.get("print_report"):
            self.__infer_table(
                run_parallel=False,
                batch_size=1000,
                random_seed=1
            )

    def __train_table(self, **kwargs):
        table = self.table_name if self.table_name else kwargs.get("table_name")
        source = self._extract_setting(kwargs, setting="source")
        epochs = self._extract_setting(kwargs, setting="epochs")
        drop_null = self._extract_setting(kwargs, setting="drop_null")
        row_limit = self._extract_setting(kwargs, setting="row_limit")
        print_report = self._extract_setting(kwargs, setting="print_report")
        batch_size = self._extract_setting(kwargs, setting="batch_size")

        logger.info(f"Training process of the table - {table} has started.")

        self.train_strategy.run(
            metadata=self.metadata,
            source=source,
            epochs=epochs,
            drop_null=drop_null,
            row_limit=row_limit,
            table_name=table,
            metadata_path=self.metadata_path,
            print_report=print_report,
            batch_size=batch_size
        )

    def __infer_table(self, **kwargs):
        """
        Run infer process for a single table
        """
        table = self.table_name if self.table_name is not None else kwargs.get("table_name")
        size = self._extract_setting(kwargs, setting="size")
        run_parallel = self._extract_setting(kwargs, setting="run_parallel")
        batch_size = self._extract_setting(kwargs, setting="batch_size")
        random_seed = self._extract_setting(kwargs, setting="random_seed")
        both_keys = table in self.divided
        print_report = self._extract_setting(kwargs, setting="print_report")

        logger.info(f"Infer process of the table - {table} has started")

        self.infer_strategy.run(
            metadata=self.metadata,
            size=size,
            table_name=table,
            metadata_path=self.metadata_path,
            run_parallel=run_parallel,
            batch_size=batch_size,
            random_seed=random_seed,
            print_report=print_report,
            both_keys=both_keys
        )

    @staticmethod
    def _generate_reports():
        """
        Generate reports
        :return:
        """
        Report().generate_report()
        Report().clear_report()

    def launch_train(self):
        """
        Launch training process either for a single table or for related tables
        """
        if self.metadata_path is not None:
            metadata_for_training = self._prepare_metadata_for_process()
            metadata_for_inference = self._prepare_metadata_for_process(type_of_process="infer")
            self.__train_chain_of_tables_with_generation(metadata_for_training, metadata_for_inference)
            self._generate_reports()
        elif self.table_name is not None:
            self.__train_table_with_generation()
            self._generate_reports()

    def launch_infer(self):
        """
        Launch infer process either for a single table or for related tables
        """
        if self.metadata_path is not None:
            chain_of_tables, config_of_tables = self._prepare_metadata_for_process(type_of_process="infer")
            self.metadata = config_of_tables
            self.__infer_chain_of_tables(chain_of_tables, config_of_tables)
            self._generate_reports()
        if self.table_name is not None:
            self.__infer_table()
            self._generate_reports()
