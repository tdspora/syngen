import attr
from typing import Optional


@attr.s(auto_attribs=True)
class TrainConfig:
    """
    The configuration class to set up the work of train process
    """
    path: str = attr.ib(kw_only=True)
    keys_mode: Optional[bool] = False
    epochs: int = 10
    dropna: bool = False
    row_limit: Optional[int] = None
    table_name: Optional[str] = None
    metadata_path: Optional[str] = None
    batch_size: int = 32

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


@attr.s(auto_attribs=True)
class InferConfig:
    """
    The configuration class to set up the work of infer process
    """
    size: int = attr.ib(kw_only=True)
    table_name: str = attr.ib(kw_only=True)
    run_parallel: bool = True
    batch_size: Optional[int] = None
    keys_mode: Optional[bool] = None
    metadata_path: Optional[str] = None
    random_seed: Optional[int] = None
    print_report: bool = False

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
