from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Set, List, Callable, Literal
import os
from copy import deepcopy
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
    metadata: Dict
    reports: List[str]
    batch_size: int
    loader: Optional[Callable[[str], pd.DataFrame]]
    data: pd.DataFrame = field(init=False)
    initial_data_shape: Tuple[int, int] = field(init=False)
    paths: Dict = field(init=False)
    row_subset: int = field(init=False)
    schema: Dict = field(init=False)
    original_schema: Dict = field(init=False)
    slugify_table_name: str = field(init=False)
    columns: List = field(init=False)
    dropped_columns: Set = field(init=False)

    def __post_init__(self):
        self._set_paths()
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
        self._extract_data()
        self._save_original_schema()
        self.columns = list(self.data.columns)
        self._remove_empty_columns()
        self._mark_removed_columns()
        self._prepare_data()
        self._check_reports()

    def to_dict(self) -> Dict:
        """
        Return the values of the settings of training process
        """
        return {
            "epochs": self.epochs,
            "drop_null": self.drop_null,
            "row_subset": self.row_subset,
            "batch_size": self.batch_size,
            "reports": self.reports
        }

    def _set_batch_size(self):
        """
        Set up "batch_size" for training process
        """
        self.batch_size = min(self.batch_size, self.row_subset)

    def _check_sample_report(self):
        """
        Check whether it is necessary to generate a certain report
        """
        if "sample" in self.reports and self.initial_data_shape[0] == self.row_subset:
            logger.warning(
                "The generation of the sample report is unnecessary and won't be produced "
                "as the source data and sampled data sizes are identical"
            )
            reports = deepcopy(self.reports)
            reports.remove("sample")
            self.reports = reports

    def _check_reports(self):
        """
        Check whether it is necessary to generate a certain report
        """
        self._check_sample_report()

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

    def _fetch_dataframe(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch the dataframe using the callback function
        """
        dataframe_fetcher = DataFrameFetcher(
            loader=self.loader,
            table_name=self.table_name
        )
        self.original_schema = dataframe_fetcher.original_schema
        return dataframe_fetcher.fetch_data()

    def _load_source(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Return dataframe and schema of original data
        """
        if self.loader is not None:
            return self._fetch_dataframe()
        else:
            data_loader = DataLoader(self.source)
            self.original_schema = data_loader.original_schema
            if self.original_schema is not None:
                logger.trace(
                    f"The schema of the table - '{self.table_name}': {self.original_schema}"
                )
            return data_loader.load_data()

    def _remove_empty_columns(self):
        """
        Remove completely empty columns from dataframe
        """
        data_columns = set(self.data.columns)
        self.data = self.data.dropna(how="all", axis=1)

        self.dropped_columns = data_columns - set(self.data.columns)
        if len(self.dropped_columns) > 0:
            logger.info(f"Empty columns - {', '.join(self.dropped_columns)} were removed")

    def _mark_removed_columns(self):
        """
        Mark removed columns in the schema
        """
        if self.schema.get("format") == "CSV":
            self.schema["fields"] = dict()
            self.schema["fields"] = {column: "removed" for column in self.dropped_columns}
        else:
            for column, data_type in self.schema.get("fields", {}).items():
                if column not in self.data.columns:
                    self.schema["fields"][column] = "removed"

    def _check_if_data_is_empty(self):
        """
        Check if the provided data is empty
        """
        if self.data.shape[0] < 1:
            raise ValueError(
                f"The empty table was provided. Unable to train the table - '{self.table_name}'"
            )

    def _extract_data(self):
        """
        Extract data and schema necessary for training process
        """
        self.data, self.schema = self._load_source()
        self.initial_data_shape = self.data.shape
        self._check_if_data_is_empty()

    def _preprocess_data(self):
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
                if not self.data.dropna().empty:
                    initial_data = self.data
                    self.data = self.data.dropna()
                    if count_of_dropped_rows := initial_data.shape[0] - self.data.shape[0]:
                        logger.info(
                            f"As the parameter 'drop_null' set to 'True', "
                            f"{count_of_dropped_rows} rows of the table - '{self.table_name}' "
                            f"that have empty values have been dropped. "
                            f"The count of remained rows is {self.data.shape[0]}."
                        )
                else:
                    logger.warning(
                        "The specified 'drop_null' argument results in the empty dataframe, "
                        "so it will be ignored"
                    )

            if self.row_limit:
                self.row_subset = min(self.row_limit, len(self.data))

                self.data = self.data.sample(n=self.row_subset)

        if len(self.data) < 100:
            logger.warning(
                "The input table is too small to provide any meaningful results. "
                "Please consider 1) disable drop_null argument, 2) provide bigger table"
            )
        elif len(self.data) < 500:
            logger.warning(
                f"The amount of data is {len(self.data)} rows. It seems that it isn't enough "
                f"to supply high-quality results. To improve the quality of generated data "
                f"please consider any of the steps: 1) provide a bigger table, "
                f"2) disable drop_null argument"
            )

        logger.info(f"The subset of rows was set to {len(self.data)}")

        self.row_subset = len(self.data)
        self._set_batch_size()

    def _save_input_data(self):
        """
        Save the subset of the original data
        """
        DataLoader(self.paths["input_data_path"]).save_data(self.data)

    def _save_original_schema(self):
        """
        Save the schema of the original data
        """
        DataLoader(self.paths["original_schema_path"]).save_data(self.original_schema)

    def _prepare_data(self):
        """
        Preprocess and save the data necessary for the training process
        """
        self._preprocess_data()
        if not self.loader:
            self._save_input_data()

    @slugify_attribute(table_name="slugify_table_name")
    def _set_paths(self):
        """
        Create the paths which used in training process
        """
        losses_file_name = (
            f"losses_{self.table_name}_"
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.paths = {
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
            "original_schema_path": f"model_artifacts/tmp_store/{self.slugify_table_name}/"
                                    f"original_schema_{self.slugify_table_name}.pkl",
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
    metadata: Dict
    metadata_path: Optional[str]
    size: Optional[int]
    table_name: Optional[str]
    run_parallel: bool
    batch_size: Optional[int]
    random_seed: Optional[int]
    reports: List[str]
    both_keys: bool
    log_level: str
    loader: Optional[Callable[[str], pd.DataFrame]]
    type_of_process: Literal["train", "infer"]
    slugify_table_name: str = field(init=False)

    def __post_init__(self):
        self._set_paths()
        self._remove_artifacts()
        self._set_infer_parameters()

    def _set_infer_parameters(self):
        self._check_reports()
        self._set_up_size()
        self._set_up_batch_size()

    def _remove_reports(self):
        path_to_reports = self.paths["reports_path"]
        if os.path.exists(path_to_reports):
            shutil.rmtree(path_to_reports)
            logger.info(
                f"The reports generated in the previous run of an inference process "
                f"and located in the path - '{path_to_reports}' were removed"
            )

    def _remove_generated_data(self):
        default_path_to_synth_data = self.paths["default_path_to_merged_infer"]
        if os.path.exists(default_path_to_synth_data):
            os.remove(default_path_to_synth_data)
            logger.info(
                f"The synthetic data generated in the previous run of an inference process and "
                f"located in the path - '{default_path_to_synth_data}' was removed"
            )

    def _remove_artifacts(self):
        """
        Remove artifacts related to the previous generation process
        """
        self._remove_reports()
        self._remove_generated_data()

    def to_dict(self) -> Dict:
        """
        Return the values of the settings of inference process
        """
        return {
            "size": self.size,
            "run_parallel": self.run_parallel,
            "batch_size": self.batch_size,
            "random_seed": self.random_seed,
            "reports": self.reports,
        }

    def _check_required_artifacts(self):
        """
        Check whether required artifacts exists
        """
        if (
                self.reports
                and (
                    DataLoader(self.paths["input_data_path"]).has_existed_path is False
                    or self.loader is not None
                )
        ):
            self.reports = list()
            log_message = (
                f"It seems that the path to the sample of the original data for the table "
                f"'{self.table_name}' at '{self.paths['input_data_path']}' does not exist. "
                f"As a result, no reports for the table '{self.table_name}' will be generated. "
                f"The 'reports' parameter for the table '{self.table_name}' "
                f"has been set to 'none'."
            )
            logger.warning(log_message)

    def _check_reports(self):
        """
        Check whether it is possible to generate reports
        """
        self._check_required_artifacts()

    def _set_up_size(self):
        """
        Set up "size" of generated data
        """
        if self.size is None:
            data_loader = DataLoader(self.paths["input_data_path"])
            data = pd.DataFrame()
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
    def _set_paths(self):
        """
        Create the paths which used in inference process
        """
        dynamic_name = self.slugify_table_name[:-3] if self.both_keys else self.slugify_table_name
        self.paths = {
            "original_data_path":
                f"model_artifacts/tmp_store/{dynamic_name}/input_data_{dynamic_name}.pkl",
            "reports_path": f"model_artifacts/tmp_store/{dynamic_name}/reports",
            "input_data_path":
                f"model_artifacts/tmp_store/{dynamic_name}/input_data_{dynamic_name}.pkl",
            "default_path_to_merged_infer": f"model_artifacts/tmp_store/{dynamic_name}/"
                                            f"merged_infer_{dynamic_name}.csv",
            "path_to_merged_infer": self.destination
            if self.destination is not None
            else f"model_artifacts/tmp_store/{dynamic_name}/merged_infer_{dynamic_name}.csv",
            "state_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints",
            "train_config_pickle_path":
                f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/train_config.pkl",
            "original_schema_path": f"model_artifacts/tmp_store/{self.slugify_table_name}/"
                                    f"original_schema_{self.slugify_table_name}.pkl",
            "tmp_store_path": f"model_artifacts/tmp_store/{dynamic_name}",
            "vae_resources_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/",
            "dataset_pickle_path":
                f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/model_dataset.pkl",
            "fk_kde_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/stat_keys/",
            "path_to_no_ml":
                f"model_artifacts/resources/{dynamic_name}/no_ml/checkpoints/kde_params.pkl",
        }
