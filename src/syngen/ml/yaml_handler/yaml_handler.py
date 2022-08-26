from typing import Dict
from loguru import logger

from syngen.ml.config import TrainConfig, InferConfig
from syngen.train import train
from syngen.infer import infer
from syngen.ml.data_loaders import YAMLLoader


class YAMLHandler:
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        self.yaml_loader = YAMLLoader()

    @staticmethod
    def parse_train_settings(config: Dict):
        try:
            epochs = config["train_settings"].get("epochs", 10)
            keys_mode = config["train_settings"].get("keys_mode", False)
            dropna = config["train_settings"].get("dropna", False)
            row_limit = config["train_settings"].get("row_limit", None)
            batch_size = config["train_settings"].get("batch_size", 32)
            return {
                "epochs": epochs,
                "keys_mode": keys_mode,
                "dropna": dropna,
                "row_limit": row_limit,
                "batch_size": batch_size
            }
        except KeyError:
            logger.info("The values of parameters for training process are set to default values.")

    @staticmethod
    def parse_infer_settings(config: Dict):
        try:
            size = config["infer_settings"].get("size", 0)
            run_parallel = config["infer_settings"].get("run_parallel", False)
            keys_mode = config["infer_settings"].get("keys_mode", None)
            random_seed = config["infer_settings"].get("random_seed", None)
            print_report = config["infer_settings"].get("print_report", False)
            batch_size = config["train_settings"].get("batch_size", None)
            return {
                "size": size,
                "run_parallel": run_parallel,
                "keys_mode": keys_mode,
                "random_seed": random_seed,
                "print_report": print_report,
                "batch_size": batch_size
            }
        except KeyError:
            logger.info("The values of parameters for infer process are set to default values.")

    @staticmethod
    def parse_relationship(config: Dict):
        try:
            return config["configuration"]["relationship"].split(" -> ")
        except KeyError:
            raise KeyError(
                "The information of relationship seems to be absent in metadata file. "
                "Please provide the information of relationship."
            )

    @staticmethod
    def parse_tables_config(config: Dict):
        try:
            return config["configuration"]["tables"]
        except KeyError:
            raise KeyError(
                "The information of tables seems to be absent in metadata file. "
                "Please provide the information of tables."
            )

    def run(self):
        config = self.yaml_loader.load_data(self.metadata_path)
        train_settings = self.parse_train_settings(config)
        infer_settings = self.parse_infer_settings(config)
        chain_of_tables = self.parse_relationship(config)
        tables = self.parse_tables_config(config)
        for table in chain_of_tables:
            path = tables[table]["path"]
            train_config = TrainConfig(
                path=path,
                keys_mode=train_settings["keys_mode"],
                epochs=train_settings["epochs"],
                dropna=train_settings["dropna"],
                row_limit=train_settings["row_limit"],
                table_name=table,
                metadata_path=self.metadata_path,
                batch_size=train_settings["batch_size"]
            )

            train(train_config)

            infer_config = InferConfig(
                size=infer_settings["size"],
                table_name=table,
                run_parallel=infer_settings["run_parallel"],
                batch_size=infer_settings["batch_size"],
                keys_mode=infer_settings["keys_mode"],
                metadata_path=self.metadata_path,
                random_seed=infer_settings["random_seed"],
                print_report=infer_settings["print_report"]
            )

            infer(infer_config)








