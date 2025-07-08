from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Literal
from datetime import datetime

import pandas as pd
from slugify import slugify

from syngen.ml.data_loaders import DataLoader, DataFrameFetcher
from syngen.ml.utils import slugify_attribute, fetch_unique_root, fetch_config


@dataclass
class TrainConfig:
    """
    The configuration class to set up the work of train process
    """
    data: pd.DataFrame
    schema: Dict
    source: Optional[str]
    epochs: int
    drop_null: bool
    row_limit: Optional[int]
    table_name: str
    metadata: Dict
    metadata_path: Optional[str]
    reports: List[str]
    batch_size: int
    loader: Optional[Callable[[str], pd.DataFrame]]
    paths: Dict = field(init=False)
    row_subset: int = field(init=False)
    slugify_table_name: str = field(init=False)

    def __post_init__(self):
        self.row_subset = len(self.data)
        self.batch_size = min(self.batch_size, self.row_subset)
        self._set_paths()

    def __getstate__(self) -> Dict:
        """
        Return an updated config's instance
        """
        instance = self.__dict__.copy()
        attribute_keys_to_remove = ["loader", "data"]
        for attr_key in attribute_keys_to_remove:
            if attr_key in instance:
                del instance[attr_key]
        return instance

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

    @slugify_attribute(table_name="slugify_table_name")
    def _set_paths(self):
        """
        Create the paths which used in training process
        """
        losses_file_name = (
            f"losses_{self.table_name}_"
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        fernet_key = self.metadata[self.table_name].get("encryption", {}).get("fernet_key")
        self.paths = {
            "model_artifacts_path": "model_artifacts/",
            "resources_path": f"model_artifacts/resources/{self.slugify_table_name}/",
            "tmp_store_path": f"model_artifacts/tmp_store/{self.slugify_table_name}/",
            "source_path": self.source,
            "reports_path": f"model_artifacts/tmp_store/{self.slugify_table_name}/reports",
            "input_data_path": f"model_artifacts/tmp_store/{self.slugify_table_name}/"
                               f"input_data_{self.slugify_table_name}."
                               f"{'dat' if fernet_key is not None else 'pkl'}",
            "state_path": f"model_artifacts/resources/{self.slugify_table_name}/vae/checkpoints",
            "train_config_pickle_path": f"model_artifacts/resources/{self.slugify_table_name}/vae/"
                                        f"checkpoints/train_config.pkl",
            "dataset_pickle_path": f"model_artifacts/resources/{self.slugify_table_name}/vae/"
                                   f"checkpoints/model_dataset.pkl",
            "fk_kde_path": f"model_artifacts/resources/{self.slugify_table_name}/vae/"
                           f"checkpoints/stat_keys/",
            "original_schema_path": f"model_artifacts/tmp_store/{self.slugify_table_name}/"
                                    f"original_schema_{self.slugify_table_name}.pkl",
            "path_to_merged_infer": f"model_artifacts/tmp_store/{self.slugify_table_name}/"
                                    f"merged_infer_{self.slugify_table_name}.csv",
            "no_ml_state_path":
                f"model_artifacts/resources/{self.slugify_table_name}/no_ml/checkpoints/",
            "path_to_flatten_metadata":
                f"model_artifacts/system_store/flatten_configs/"
                f"flatten_metadata_{fetch_unique_root(self.table_name, self.metadata_path)}.json",
            "losses_path": f"model_artifacts/system_store/losses/{slugify(losses_file_name)}.csv"
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
    table_name: str
    run_parallel: bool
    batch_size: Optional[int]
    random_seed: Optional[int]
    reports: List[str]
    both_keys: bool
    log_level: str
    loader: Optional[Callable[[str], pd.DataFrame]]
    type_of_process: Literal["train", "infer"]
    slugify_table_name: str = field(init=False)
    paths: Dict = field(init=False)

    def __post_init__(self):
        self.__set_paths()
        self._set_infer_parameters()

    def _set_infer_parameters(self):
        self._set_up_size()
        self._set_up_batch_size()

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

    def _set_up_size(self):
        """
        Set up "size" of generated data
        """
        if self.size is None:
            data_loader = DataLoader(
                path=self.paths["input_data_path"],
                table_name=self.table_name,
                metadata=self.metadata,
                sensitive=True
            )
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

    @property
    def train_config(self):
        """
        Fetch the training configuration
        """
        return fetch_config(self.paths["train_config_pickle_path"])

    def _set_paths(self):
        """
        Create the paths which used in inference process
        """
        self.paths.update({
            "original_schema_path": f"model_artifacts/tmp_store/{self.slugify_table_name}/"
                                    f"original_schema_{self.slugify_table_name}.pkl",
            "path_to_flatten_metadata":
                f"model_artifacts/system_store/flatten_configs/"
                f"flatten_metadata_{fetch_unique_root(self.table_name, self.metadata_path)}.json",
            "input_data_path": self.train_config.paths["input_data_path"]
        })

    @slugify_attribute(table_name="slugify_table_name")
    def __set_paths(self):
        """
        Create the paths which used in inference process
        """
        dynamic_name = (
            self.slugify_table_name[:-3] if self.both_keys else self.slugify_table_name
        )
        self.paths = {
            "reports_path": f"model_artifacts/tmp_store/{dynamic_name}/reports",
            "train_config_pickle_path":
                f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/train_config.pkl",
            "default_path_to_merged_infer": f"model_artifacts/tmp_store/{dynamic_name}/"
                                            f"merged_infer_{dynamic_name}.csv",
            "path_to_merged_infer": (
                self.destination
                if self.destination is not None
                else f"model_artifacts/tmp_store/{dynamic_name}/"
                     f"merged_infer_{dynamic_name}.csv"
            ),
            "state_path": f"model_artifacts/resources/{dynamic_name}/vae/checkpoints",
            "tmp_store_path": f"model_artifacts/tmp_store/{dynamic_name}",
            "vae_resources_path":
                f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/",
            "dataset_pickle_path":
                f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/model_dataset.pkl",
            "fk_kde_path":
                f"model_artifacts/resources/{dynamic_name}/vae/checkpoints/stat_keys/",
            "path_to_no_ml":
                f"model_artifacts/resources/{dynamic_name}/no_ml/checkpoints/kde_params.pkl"
        }

        self._set_paths()
