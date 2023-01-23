from dataclasses import dataclass, field
from typing import Optional, Dict
import os

from loguru import logger
import pandas as pd

from syngen.ml.data_loaders import DataLoader


@dataclass
class TrainConfig:
    """
    The configuration class to set up the work of train process
    """
    source: Optional[str]
    epochs: int
    drop_null: bool
    row_limit: Optional[int]
    table_name: Optional[str]
    metadata_path: Optional[str]
    print_report: bool
    batch_size: int
    paths: Dict = field(default_factory=dict)
    row_subset: int = field(default=None)
    schema: Dict = field(default=None)

    def __post_init__(self):
        self.paths = self.set_paths()
        self._prepare_dirs()
        data, self.schema = self.extract_data()
        self.prepare_data(data)

    def to_dict(self) -> Dict:
        """
        Return the values of the settings of training process
        """
        return {
            "epochs": self.epochs,
            "drop_null": self.drop_null,
            "row_subset": self.row_subset,
            "batch_size": self.batch_size
        }

    def _prepare_dirs(self):
        os.makedirs(self.paths["model_artifacts_path"], exist_ok=True)
        tmp_store_path = self.paths["tmp_store_path"]
        os.makedirs(tmp_store_path, exist_ok=True)

    def extract_data(self):
        data, schema = DataLoader(self.source).load_data()
        # remove completely empty columns
        data = data.dropna(how="all", axis=1)
        if schema is not None:
            schema["fields"] = {
                column: data_type for column, data_type in schema.get("fields", {}).items() if column in data.columns
            }
        return data, schema

    def _set_options(self, data) -> pd.DataFrame:
        if self.drop_null:
            if not data.dropna().empty:
                data = data.dropna()
            else:
                logger.warning("The specified 'drop_null' argument results in the empty dataframe, "
                               "so it will be ignored")

        if self.row_limit:
            self.row_subset = min(self.row_limit, len(data))

            data = data.sample(n=self.row_subset)
            if len(data) < 100:
                logger.warning("The input table is too small to provide any meaningful results. "
                               "Please consider 1) disable drop_null argument, 2) provide bigger table")
            elif len(data) < 500:
                logger.warning(
                    f"The amount of data is {len(data)} rows. It seems that it isn't enough to supply "
                    f"high-quality results. To improve the quality of generated data please consider any of the steps: "
                    f"1) provide a bigger table, 2) disable drop_null argument")

        logger.info(f"The subset of rows was set to {len(data)}.")

        self.row_subset = len(data)
        return data

    def set_options(self, data) -> pd.DataFrame:
        return self._set_options(data)

    def prepare_data(self, data):
        data = self.set_options(data)

        data_columns = set(data.columns)
        dropped_cols = set(data.columns) - data_columns
        if len(dropped_cols) > 0:
            logger.info(f"Empty columns {dropped_cols} were removed")
        data.to_csv(self.paths["input_data_path"], index=False)

    def set_paths(self) -> Dict:
        """
        Create the paths which used in training process
        """
        return {
            "model_artifacts_path": "model_artifacts/",
            "tmp_store_path": f"model_artifacts/tmp_store/{self.table_name}",
            "source_path": self.source,
            "draws_path": f"model_artifacts/tmp_store/{self.table_name}/draws",
            "input_data_path": f"model_artifacts/tmp_store/{self.table_name}/input_data_{self.table_name}.csv",
            "state_path": f"model_artifacts/resources/{self.table_name}/vae/checkpoints",
            "results_path": f"model_artifacts/tmp_store/{self.table_name}/metrics_{self.table_name}.csv",
            "bad_columns_path": f"model_artifacts/tmp_store/{self.table_name}/bad_columns_{self.table_name}.csv",
            "dataset_pickle_path": f"model_artifacts/resources/{self.table_name}/vae/checkpoints/model_dataset.pkl",
            "fk_kde_path": f"model_artifacts/resources/{self.table_name}/vae/checkpoints/fk_kde.pkl",
            "original_data_path": f"model_artifacts/tmp_store/{self.table_name}/input_data_{self.table_name}.csv",
            "synthetic_data_path": f"model_artifacts/tmp_store/{self.table_name}/merged_infer_{self.table_name}.csv",
        }


@dataclass
class InferConfig:
    """
    The configuration class to set up the work of infer process
    """
    size: Optional[int]
    table_name: Optional[str]
    run_parallel: bool
    batch_size: Optional[int]
    metadata_path: Optional[str]
    random_seed: Optional[int]
    print_report: bool
    both_keys: bool

    def to_dict(self) -> Dict:
        """
        Return the values of the settings of inference process
        :return:
        """
        return {
            "size": self.size,
            "run_parallel": self.run_parallel,
            "batch_size": self.batch_size,
            "random_seed": self.random_seed
        }

    def set_paths(self) -> Dict:
        """
        Create the paths which used in inference process
        """
        dynamic_name = self.table_name[:-3] if self.both_keys else self.table_name
        return {
            "original_data_path": f"model_artifacts/tmp_store/{dynamic_name}/input_data_{dynamic_name}.csv",
            "synthetic_data_path": f"model_artifacts/tmp_store/{dynamic_name}/merged_infer_{dynamic_name}.csv",
            "draws_path": f"model_artifacts/tmp_store/{dynamic_name}/draws",
            "input_data_path": f"model_artifacts/tmp_store/{dynamic_name}/input_data_{dynamic_name}.csv",
            "path_to_merged_infer": f"model_artifacts/tmp_store/{dynamic_name}/merged_infer_{dynamic_name}.csv",
            "state_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints",
            "tmp_store_path": f"model_artifacts/tmp_store/{dynamic_name}",
            "vae_resources_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/",
            "dataset_pickle_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/model_dataset.pkl",
            "fk_kde_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/fk_kde.pkl"
        }
