from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """
    The configuration class to set up the work of train process
    """
    path: str
    keys_mode: Optional[bool]
    epochs: int
    dropna: bool
    row_limit: Optional[int]
    table_name: Optional[str]
    metadata_path: Optional[str]
    batch_size: int

    def set_paths(self):
        """
        Create paths which used in training process
        """
        return {
            "model_artifacts_path": "model_artifacts/",
            "tmp_store_path": f"model_artifacts/tmp_store/{self.table_name}",
            "input_data_path": f"model_artifacts/tmp_store/{self.table_name}/input_data.csv",
            "state_path": f"model_artifacts/resources/{self.table_name}/vae/checkpoints",
            "results_path": f"model_artifacts/tmp_store/{self.table_name}/metrics.csv",
            "bad_columns_path": f"model_artifacts/tmp_store/{self.table_name}/bad_columns.csv",
            "dataset_pickle_path": f"model_artifacts/resources/{self.table_name}/vae/checkpoints/model_dataset.pkl"
        }


@dataclass
class InferConfig:
    """
    The configuration class to set up the work of infer process
    """
    size: int
    run_parallel: bool
    table_name: str
    batch_size: Optional[int]
    keys_mode: Optional[bool]
    metadata_path: Optional[str]
    random_seed: Optional[int]
    print_report: bool

    def set_paths(self):
        """
        Create paths which used in infer process
        """
        return {
            "original_data_path": f"model_artifacts/tmp_store/{self.table_name}/input_data.csv",
            "synthetic_data_path": f"model_artifacts/tmp_store/{self.table_name}/merged_infer.csv",
            "draws_path": f"model_artifacts/tmp_store/{self.table_name}/draws/",
            "path_to_merged_infer": f"model_artifacts/tmp_store/{self.table_name}/merged_infer.csv",
            "state_path": f"model_artifacts/resources/{self.table_name}/vae/checkpoints",
            "tmp_store_path": f"model_artifacts/tmp_store/{self.table_name}",
            "vae_resources_path": f"model_artifacts/resources/{self.table_name}/vae/checkpoints/",
            "dataset_pickle_path": f"model_artifacts/resources/{self.table_name}/vae/checkpoints/model_dataset.pkl",
            "pk_path": f"model_artifacts/resources/{self.table_name}/pk/"
        }

