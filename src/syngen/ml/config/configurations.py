from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Set
import os

from loguru import logger
import pandas as pd

from syngen.ml.data_loaders import DataLoader
from syngen.ml.utils import slugify_attribute


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
    paths: Dict = field(init=False)
    row_subset: int = field(init=False)
    schema: Dict = field(init=False)
    slugify_table_name: str = field(init=False)

    def __post_init__(self):
        self.paths = self._set_paths()
        self._prepare_dirs()
        data, self.schema = self._extract_data()
        self._prepare_data(data)
        self._set_batch_size()

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

    def _set_batch_size(self):
        """
        Set up "batch_size" for training process
        """
        self.batch_size = min(self.batch_size, self.row_subset)

    def _prepare_dirs(self):
        """
        Create main directories for saving original, synthetic data and model artifacts
        """
        os.makedirs(self.paths["model_artifacts_path"], exist_ok=True)
        tmp_store_path = self.paths["tmp_store_path"]
        os.makedirs(tmp_store_path, exist_ok=True)

    def _load_source(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Return dataframe and schema of original data
        """
        return DataLoader(self.source).load_data()

    @staticmethod
    def _remove_empty_columns(data: pd.DataFrame) -> Tuple[pd.DataFrame, Set]:
        """
        Remove completely empty columns from dataframe
        """
        data_columns = set(data.columns)
        data = data.dropna(how="all", axis=1)

        dropped_cols = data_columns - set(data.columns)
        if len(dropped_cols) > 0:
            logger.info(f"Empty columns - {', '.join(dropped_cols)} were removed")
        return data, dropped_cols

    @staticmethod
    def _mark_removed_columns(data: pd.DataFrame, schema: Optional[Dict], dropped_columns: Set) -> Dict:
        """
        Mark removed columns in the schema
        """
        if schema.get("format") == "CSV" and dropped_columns:
            schema["fields"] = {
                column: "removed"
                for column in dropped_columns
            }
        else:
            for column, data_type in schema.get("fields", {}).items():
                if column not in data.columns:
                    schema["fields"][column] = "removed"
                continue
        return schema

    def _extract_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Extract data and schema necessary for training process
        """
        data, schema = self._load_source()
        data, dropped_columns = self._remove_empty_columns(data)
        schema = self._mark_removed_columns(data, schema, dropped_columns)
        return data, schema

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data and set the parameter "row_subset" for training process
        """
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

        logger.info(f"The subset of rows was set to {len(data)}")

        self.row_subset = len(data)
        return data

    def _save_input_data(self, data: pd.DataFrame):
        DataLoader(self.paths["input_data_path"]).save_data(self.paths["input_data_path"], data)

    def _prepare_data(self, data: pd.DataFrame):
        """
        Preprocess and save data necessary for training process
        """
        data = self._preprocess_data(data)
        self._save_input_data(data)

    @slugify_attribute(table_name="slugify_table_name")
    def _set_paths(self) -> Dict:
        """
        Create the paths which used in training process
        """

        return {
            "model_artifacts_path": "model_artifacts/",
            "tmp_store_path": f"model_artifacts/tmp_store/{self.slugify_table_name}",
            "source_path": self.source,
            "draws_path": f"model_artifacts/tmp_store/{self.slugify_table_name}/draws",
            "input_data_path":
                f"model_artifacts/tmp_store/{self.slugify_table_name}/input_data_{self.slugify_table_name}.pkl",
            "state_path": f"model_artifacts/resources/{self.slugify_table_name}/vae/checkpoints",
            "dataset_pickle_path":
                f"model_artifacts/resources/{self.slugify_table_name}/vae/checkpoints/model_dataset.pkl",
            "fk_kde_path": f"model_artifacts/resources/{self.slugify_table_name}/vae/checkpoints/",
            "original_data_path":
                f"model_artifacts/tmp_store/{self.slugify_table_name}/input_data_{self.slugify_table_name}.pkl",
            "synthetic_data_path":
                f"model_artifacts/tmp_store/{self.slugify_table_name}/merged_infer_{self.slugify_table_name}.csv",
            "no_ml_state_path": f"model_artifacts/resources/{self.slugify_table_name}/no_ml/checkpoints/"
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
    slugify_table_name: str = field(init=False)

    def __post_init__(self):
        self.paths = self._set_paths()
        self._set_up_reporting()
        self._set_up_size()
        self._set_up_batch_size()

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

    def _set_up_reporting(self):
        """
        Check whether it is possible to generate the report
        """
        if self.print_report and not DataLoader(self.paths["input_data_path"]).has_existed_path:
            self.print_report = False
            logger.warning(
                f"It seems that the path to original data of the table - {self.table_name} doesn't exist. "
                f"In this case, the accuracy report of the table - {self.table_name} won't be generated. "
                f"The parameter '--print_report' of the table - {self.table_name} will be set to False\n")

    def _set_up_size(self):
        """
        Set up "size" of generated data
        """
        if self.size is None and DataLoader(self.paths["input_data_path"]).has_existed_path:
            data, schema = DataLoader(self.paths["input_data_path"]).load_data()
            self.size = len(data)

    def _set_up_batch_size(self):
        """
        Set up "batch_size" of inference process
        """
        self.batch_size = min(self.batch_size, self.size) if self.batch_size is not None else self.size

    @slugify_attribute(table_name="slugify_table_name")
    def _set_paths(self) -> Dict:
        """
        Create the paths which used in inference process
        """
        dynamic_name = self.slugify_table_name[:-3] if self.both_keys else self.slugify_table_name
        return {
            "original_data_path": f"model_artifacts/tmp_store/{dynamic_name}/input_data_{dynamic_name}.pkl",
            "synthetic_data_path": f"model_artifacts/tmp_store/{dynamic_name}/merged_infer_{dynamic_name}.csv",
            "draws_path": f"model_artifacts/tmp_store/{dynamic_name}/draws",
            "input_data_path": f"model_artifacts/tmp_store/{dynamic_name}/input_data_{dynamic_name}.pkl",
            "path_to_merged_infer": f"model_artifacts/tmp_store/{dynamic_name}/merged_infer_{dynamic_name}.csv",
            "state_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints",
            "tmp_store_path": f"model_artifacts/tmp_store/{dynamic_name}",
            "vae_resources_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/",
            "dataset_pickle_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/model_dataset.pkl",
            "fk_kde_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/",
            "path_to_no_ml": f"model_artifacts/resources/{dynamic_name}/no_ml/checkpoints/",
        }
