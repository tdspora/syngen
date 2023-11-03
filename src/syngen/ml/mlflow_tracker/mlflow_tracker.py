import mlflow
import re
from typing import Optional, Dict, Any
from loguru import logger


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

    @classmethod
    def reset_status(cls, active_status: bool = True):
        cls._instance.is_active = active_status

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        if self.is_active:
            mlflow.log_metric(key, value, step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        if self.is_active:
            mlflow.log_artifact(local_path, artifact_path)

    def log_params(self, params: Dict[str, Any]):
        if self.is_active:
            mlflow.log_params(params)

    def start_run(
        self,
        run_id: str = None,
        experiment_id: Optional[str] = None,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        if self.is_active:
            mlflow.start_run(run_id, experiment_id, run_name, nested, tags, description)

    def end_run(self):
        if self.is_active:
            mlflow.end_run()

    def set_tracking_uri(self, uri: str):
        if self.is_active:
            mlflow.set_tracking_uri(uri)

    def create_experiment(
        self,
        name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
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
            datetime_pattern = r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}$"
            name = re.sub(datetime_pattern, "", experiment_name)
            experiments = mlflow.search_experiments(
                filter_string=f"name LIKE '{name}%'"
            )
            last_matching = experiments[0] if experiments else None
            if not last_matching:
                logger.warning(
                    f"It seems that no experiment with a name starting with - '{name}' was found. "
                    f"A new experiment with the name  - '{experiment_name}' will be created"
                )
            matching_name = last_matching.name if last_matching else experiment_name
            mlflow.set_experiment(matching_name, experiment_id)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.is_active:
            mlflow.log_metrics(metrics, step)
