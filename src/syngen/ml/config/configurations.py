from attr import define
from typing import Optional


@define(kw_only=True)
class TrainConfig:
    """
    The configuration class to set up the work of train process
    """
    source: Optional[str]
    epochs: int = 10
    drop_null: bool = False
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
            "source_path": self.source,
            "draws_path": f"model_artifacts/tmp_store/{self.table_name}/draws",
            "input_data_path": f"model_artifacts/tmp_store/{self.table_name}/input_data_{self.table_name}.csv",
            "state_path": f"model_artifacts/resources/{self.table_name}/vae/checkpoints",
            "results_path": f"model_artifacts/tmp_store/{self.table_name}/metrics_{self.table_name}.csv",
            "bad_columns_path": f"model_artifacts/tmp_store/{self.table_name}/bad_columns_{self.table_name}.csv",
            "dataset_pickle_path": f"model_artifacts/resources/{self.table_name}/vae/checkpoints/model_dataset.pkl",
            "fk_kde_path": f"model_artifacts/resources/{self.table_name}/vae/checkpoints/fk_kde.pkl",
        }


@define(kw_only=True)
class InferConfig:
    """
    The configuration class to set up the work of infer process
    """
    size: int
    table_name: Optional[str]
    run_parallel: bool = True
    batch_size: Optional[int] = None
    metadata_path: Optional[str] = None
    random_seed: Optional[int] = None
    print_report: bool = False
    both_keys: bool

    def set_paths(self):
        """
        Create paths which used in infer process
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
