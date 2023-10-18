import mlflow
import re
from typing import Optional, Dict, Any


class MlflowTracker:
    """
    A singleton class for tracking the MLflow experiments.
    All methods are derived from the MLflow API, however a check for the active state is added.
    """
    _instance = None

    def __new__(cls, experiment_name=None, is_active=False):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.experiment_name = experiment_name
            cls._instance.is_active = is_active
        return cls._instance

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        if self.is_active:
            mlflow.log_metric(key, value, step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        if self.is_active:
            mlflow.log_artifact(local_path, artifact_path)

    def start_run(
        self,
        run_id: str = None,
        experiment_id: Optional[str] = None,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ):
        if self.is_active:
            mlflow.start_run(run_id, experiment_id, run_name, nested, tags, description)

    def end_run(self):
        if self.is_active:
            mlflow.end_run()

    def set_tags(self, tags: dict):
        if self.is_active:
            mlflow.set_tags(tags)

    def set_tracking_uri(self, uri: str):
        print(uri, self.is_active)
        if self.is_active:
            mlflow.set_tracking_uri(uri)

    def create_experiment(
        self,
        name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ):
        if self.is_active:
            mlflow.create_experiment(name, artifact_location, tags)

    def set_experiment(
            self,
            experiment_name: str = None,
            experiment_id: str = None,
    ):
        """
        Set the experiment for tracking.
        If the experiment name is not provided, the last experiment will be used.
        """
        if self.is_active:
            if re.search("\d{4}-\d+-\d+\s\d+:\d+:\d+", experiment_name):
                metadata_name = experiment_name[:-20]  # strip datetime
            else:
                metadata_name = experiment_name
            last_matching = mlflow.search_experiments(filter_string=f"name LIKE '{metadata_name}%'")[0]
            matching_name = last_matching.name
            mlflow.set_experiment(matching_name, experiment_id)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.is_active:
            mlflow.log_metrics(metrics, step)