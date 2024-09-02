from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Set, List, Callable
import os
import shutil
from datetime import datetime

import pandas as pd
from loguru import logger
from slugify import slugify

from syngen.ml.data_loaders import DataLoader, DataFrameFetcher
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
    loader: Optional[Callable[[str], pd.DataFrame]]
    paths: Dict = field(init=False)
    row_subset: int = field(init=False)
    schema: Dict = field(init=False)
    slugify_table_name: str = field(init=False)
    columns: List = field(init=False)
    dropped_columns: Set = field(init=False)

    def __post_init__(self):
        self.paths = self._get_paths()
        self._remove_existed_artifacts()
        self._prepare_dirs()

    def __getstate__(self) -> Dict:
        """
        Return an updated config's instance
        """
        instance = self.__dict__.copy()
        attribute_keys_to_remove = ["loader"]
        for attr_key in attribute_keys_to_remove:
            if attr_key in instance:
                del instance[attr_key]
        return instance

    def preprocess_data(self):
        data, self.schema = self._extract_data()
        self.columns = list(data.columns)
        data = self._remove_empty_columns(data)
        self._mark_removed_columns(data)
        self._prepare_data(data)

    def to_dict(self) -> Dict:
        """
        Return the values of the settings of training process
        """
        return {
            "epochs": self.epochs,
            "drop_null": self.drop_null,
            "row_subset": self.row_subset,
            "batch_size": self.batch_size,
            "print_report": self.print_report
        }

    def _set_batch_size(self):
        """
        Set up "batch_size" for training process
        """
        self.batch_size = min(self.batch_size, self.row_subset)

    def _remove_existed_artifacts(self):
        """
        Remove existed artifacts from previous train process
        """
        if os.path.exists(self.paths["resources_path"]):
            shutil.rmtree(self.paths["resources_path"])
            logger.info(
                f"The artifacts located in the path - '{self.paths['resources_path']}' "
                f"were removed"
            )
        if os.path.exists(self.paths["tmp_store_path"]):
            shutil.rmtree(self.paths["tmp_store_path"])
            logger.info(
                f"The artifacts located in the path - '{self.paths['tmp_store_path']}' "
                f"were removed"
            )

    def _prepare_dirs(self):
        """
        Create main directories for saving original, synthetic data and model artifacts
        """
        os.makedirs(self.paths["model_artifacts_path"], exist_ok=True)
        os.makedirs(self.paths["state_path"], exist_ok=True)
        os.makedirs(self.paths["tmp_store_path"], exist_ok=True)

    def _load_source(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Return dataframe and schema of original data
        """
        if self.loader is not None:
            return DataFrameFetcher(
                loader=self.loader,
                table_name=self.table_name
            ).fetch_data()
        else:
            return DataLoader(self.source).load_data()

    def _remove_empty_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove completely empty columns from dataframe
        """
        data_columns = set(data.columns)
        data = data.dropna(how="all", axis=1)

        self.dropped_columns = data_columns - set(data.columns)
        if len(self.dropped_columns) > 0:
            logger.info(f"Empty columns - {', '.join(self.dropped_columns)} were removed")
        return data

    def _mark_removed_columns(self, data: pd.DataFrame):
        """
        Mark removed columns in the schema
        """
        if self.schema.get("format") == "CSV":
            self.schema["fields"] = dict()
            self.schema["fields"] = {column: "removed" for column in self.dropped_columns}
        else:
            for column, data_type in self.schema.get("fields", {}).items():
                if column not in data.columns:
                    self.schema["fields"][column] = "removed"

    def _check_if_data_is_empty(self, data: pd.DataFrame):
        """
        Check if the provided data is empty
        """
        if data.shape[0] < 1:
            raise ValueError(
                f"The empty table was provided. Unable to train the table - '{self.table_name}'"
            )

    def _extract_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Extract data and schema necessary for training process
        """
        data, schema = self._load_source()
        self._check_if_data_is_empty(data)
        return data, schema

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data and set the parameter "row_subset" for training process
        """
        if self.loader:
            warning_message = (
                "parameter will be ignored because the retrieval of the data "
                "is handled by a callback function"
            )
            if self.drop_null:
                logger.warning(f"The 'drop_null' {warning_message}")
            if self.row_limit is not None:
                logger.warning(f"The 'row_limit' {warning_message}")
        else:
            if self.drop_null:
                if not data.dropna().empty:
                    initial_data = data
                    data = data.dropna()
                    if count_of_dropped_rows := initial_data.shape[0] - data.shape[0]:
                        logger.info(
                            f"As the parameter 'drop_null' set to 'True', "
                            f"{count_of_dropped_rows} rows of the table - '{self.table_name}' "
                            f"that have empty values have been dropped. "
                            f"The count of remained rows is {data.shape[0]}."
                        )
                else:
                    logger.warning(
                        "The specified 'drop_null' argument results in the empty dataframe, "
                        "so it will be ignored"
                    )

            if self.row_limit:
                self.row_subset = min(self.row_limit, len(data))

                data = data.sample(n=self.row_subset)

        if len(data) < 100:
            logger.warning(
                "The input table is too small to provide any meaningful results. "
                "Please consider 1) disable drop_null argument, 2) provide bigger table"
            )
        elif len(data) < 500:
            logger.warning(
                f"The amount of data is {len(data)} rows. It seems that it isn't enough "
                f"to supply high-quality results. To improve the quality of generated data "
                f"please consider any of the steps: 1) provide a bigger table, "
                f"2) disable drop_null argument"
            )

        logger.info(f"The subset of rows was set to {len(data)}")

        self.row_subset = len(data)
        self._set_batch_size()
        return data

    def _save_input_data(self, data: pd.DataFrame):
        DataLoader(self.paths["input_data_path"]).save_data(self.paths["input_data_path"], data)

    def _prepare_data(self, data: pd.DataFrame):
        """
        Preprocess and save the data necessary for the training process
        """
        data = self._preprocess_data(data)
        if not self.loader:
            self._save_input_data(data)

    @slugify_attribute(table_name="slugify_table_name")
    def _get_paths(self) -> Dict:
        """
        Create the paths which used in training process
        """
        losses_file_name = (
            f"losses_{self.table_name}_"
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return {
            "model_artifacts_path": "model_artifacts/",
            "resources_path": f"model_artifacts/resources/{self.slugify_table_name}/",
            "tmp_store_path": f"model_artifacts/tmp_store/{self.slugify_table_name}/",
            "source_path": self.source,
            "reports_path": f"model_artifacts/tmp_store/{self.slugify_table_name}/reports",
            "input_data_path": f"model_artifacts/tmp_store/{self.slugify_table_name}/"
                               f"input_data_{self.slugify_table_name}.pkl",
            "state_path": f"model_artifacts/resources/{self.slugify_table_name}/vae/checkpoints",
            "train_config_pickle_path": f"model_artifacts/resources/{self.slugify_table_name}/vae/"
                                        f"checkpoints/train_config.pkl",
            "dataset_pickle_path": f"model_artifacts/resources/{self.slugify_table_name}/vae/"
                                   f"checkpoints/model_dataset.pkl",
            "fk_kde_path": f"model_artifacts/resources/{self.slugify_table_name}/vae/"
                           f"checkpoints/stat_keys/",
            "original_data_path": f"model_artifacts/tmp_store/{self.slugify_table_name}/"
                                  f"input_data_{self.slugify_table_name}.pkl",
            "path_to_merged_infer": f"model_artifacts/tmp_store/{self.slugify_table_name}/"
                                    f"merged_infer_{self.slugify_table_name}.csv",
            "no_ml_state_path":
                f"model_artifacts/resources/{self.slugify_table_name}/no_ml/checkpoints/",
            "losses_path": f"model_artifacts/tmp_store/losses/{slugify(losses_file_name)}.csv"
        }


@dataclass
class InferConfig:
    """
    The configuration class to set up the work of infer process
    """

    destination: Optional[str]
    size: Optional[int]
    table_name: Optional[str]
    run_parallel: bool
    batch_size: Optional[int]
    metadata_path: Optional[str]
    random_seed: Optional[int]
    print_report: bool
    get_infer_metrics: bool
    both_keys: bool
    log_level: str
    loader: Optional[Callable[[str], pd.DataFrame]]
    slugify_table_name: str = field(init=False)

    def __post_init__(self):
        self.paths = self._get_paths()
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
            "random_seed": self.random_seed,
            "print_report": self.print_report,
            "get_infer_metrics": self.get_infer_metrics,
        }

    def _set_up_reporting(self):
        """
        Check whether it is possible to generate the report
        """
        if (
                (self.print_report or self.get_infer_metrics)
                and (
                    not DataLoader(self.paths["input_data_path"]).has_existed_path
                    and not self.loader
                )
        ):
            message = (
                f"It seems that the path to original data "
                f"of the table - '{self.table_name}' doesn't exist. "
            )
            logger.warning(message)
            if self.print_report:
                self.print_report = False
                log_message = (
                    "As a result, the accuracy report of the table - "
                    f"'{self.table_name}' won't be generated. "
                    "The parameter '--print_report' of the table - "
                    f"'{self.table_name}' has been set to False"
                )
                logger.warning(log_message)
            if self.get_infer_metrics:
                self.get_infer_metrics = False
                log_message = (
                    "As a result, the infer metrics related to the table - "
                    f"'{self.table_name}' won't be fetched. "
                    "The parameter '--get_infer_metrics' of the table - "
                    f"'{self.table_name}' has been set to False"
                )
                logger.warning(log_message)

    def _set_up_size(self):
        """
        Set up "size" of generated data
        """
        if self.size is None:
            data_loader = DataLoader(self.paths["input_data_path"])
            if data_loader.has_existed_path:
                data, schema = data_loader.load_data()
            elif self.loader:
                data, schema = DataFrameFetcher(
                    loader=self.loader,
                    table_name=self.table_name
                ).fetch_data()
            self.size = len(data)

    def _set_up_batch_size(self):
        """
        Set up "batch_size" of inference process
        """
        self.batch_size = (
            min(self.batch_size, self.size) if self.batch_size is not None else self.size
        )

    @slugify_attribute(table_name="slugify_table_name")
    def _get_paths(self) -> Dict:
        """
        Create the paths which used in inference process
        """
        dynamic_name = self.slugify_table_name[:-3] if self.both_keys else self.slugify_table_name
        return {
            "original_data_path":
                f"model_artifacts/tmp_store/{dynamic_name}/input_data_{dynamic_name}.pkl",
            "reports_path": f"model_artifacts/tmp_store/{dynamic_name}/reports",
            "input_data_path":
                f"model_artifacts/tmp_store/{dynamic_name}/input_data_{dynamic_name}.pkl",
            "path_to_merged_infer": self.destination
            if self.destination is not None
            else f"model_artifacts/tmp_store/{dynamic_name}/merged_infer_{dynamic_name}.csv",
            "state_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints",
            "train_config_pickle_path":
                f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/train_config.pkl",
            "tmp_store_path": f"model_artifacts/tmp_store/{dynamic_name}",
            "vae_resources_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/",
            "dataset_pickle_path":
                f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/model_dataset.pkl",
            "fk_kde_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/stat_keys/",
            "path_to_no_ml":
                f"model_artifacts/resources/{dynamic_name}/no_ml/checkpoints/kde_params.pkl",
        }
