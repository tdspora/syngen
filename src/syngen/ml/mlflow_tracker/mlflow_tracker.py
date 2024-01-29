from typing import Optional, Dict, Any, Union
import os
import requests

import mlflow
from loguru import logger
from syngen.ml.utils import fetch_unique_root


class MlflowTrackerFactory:
    """
    A factory class for creating the Mlflow tracker
    """

    @staticmethod
    def check_mlflow_server(server_url):
        """
        Check if the MlFlow server is up and running
        """
        if server_url is None:
            logger.warning("MLFlow server URL not provided")
            return False
        try:
            response = requests.get(server_url)
            # If the response was successful, no Exception will be raised
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            logger.warning(
                f"An HTTP error occurred while connecting to the MLFlow server: {http_err}"
            )
        except Exception as err:
            logger.warning(
                f"An unexpected error occurred while connecting to the MLFlow server: {err}"
            )
        else:
            logger.info("MLFlow server is up and running")
            return True

    @classmethod
    def create_tracker(
        cls, table_name: Optional[str], metadata_path: Optional[str], is_active=False
    ):
        """
        Create the Mlflow tracker, and create or set the experiment
        """
        experiment_name = cls.get_mlflow_exp_name(table_name, metadata_path)

        tracker = MlflowTracker(experiment_name, is_active)

        response = cls.check_mlflow_server(os.environ.get("MLFLOW_TRACKING_URI"))

        if response:
            tracker.is_active = True
            tracker.connect_to_server = True
        else:
            tracker.is_active = False
            tracker.connect_to_server = False
            logger.warning(
                "MLFlow server is either unreachable or not set up, "
                "therefore the tracking will not be performed"
            )

        tracker.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        tracker.set_experiment(experiment_name)

    @classmethod
    def get_mlflow_exp_name(
        cls, table_name: Optional[str], metadata_path: Optional[str]
    ) -> str:
        """
        Get the name of the Mlflow experiment
        """
        if os.getenv("MLFLOW_EXPERIMENT_NAME"):
            return os.getenv("MLFLOW_EXPERIMENT_NAME")
        return fetch_unique_root(table_name, metadata_path)


class MlflowTracker:
    """
    A singleton class for tracking the Mlflow experiments.
    All methods are derived from the Mlflow API,
    however a check for the active state is added.
    """

    _instance = None

    def __new__(cls, experiment_name=None, is_active=False):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.experiment_name = experiment_name
            cls._instance.connect_to_server = is_active
            cls._instance.is_active = is_active
        return cls._instance

    @classmethod
    def reset_status(cls, active_status: bool = True):
        if cls._instance.connect_to_server:
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

    def create_experiment(self, name: str, artifact_location: Optional[str] = None):
        if self.is_active:
            mlflow.create_experiment(name, artifact_location)

    def set_experiment(self, experiment_name: str = None, experiment_id: str = None):
        """
        Set the experiment for tracking.
        If the experiment name is not provided, the last experiment will be used.
        """
        if self.is_active:
            experiments = mlflow.search_experiments(
                filter_string=f"name LIKE '{experiment_name}'"
            )
            env_value = os.getenv("MLFLOW_EXPERIMENT_NAME", "")
            last_matching = experiments[0] if experiments else []
            if not last_matching:
                MlflowTracker().create_experiment(
                    experiment_name,
                    artifact_location=os.environ.get(
                        "MLFLOW_ARTIFACTS_DESTINATION", "/mlflow_tracker"
                    ),
                )
                mlflow.set_experiment(experiment_name, experiment_id)

                if not env_value:
                    logger.warning(
                        f"A new experiment with the name based on 'table_name' or "
                        f"'metadata_path' value - '{experiment_name}' will be created"
                    )

                if env_value:
                    logger.info(
                        f"A new experiment with the name - '{experiment_name}' will be created"
                    )

            if last_matching:
                matching_name = last_matching.name
                mlflow.set_experiment(matching_name, experiment_id)
                if env_value:
                    logger.warning(
                        f"The experiment with the same name - '{experiment_name}' already exists. "
                        f"The created runs will be stored in the experiment - '{matching_name}'"
                    )
                if not env_value:
                    logger.warning(
                        f"The experiment with the name similar to 'table_name' or "
                        f"'metadata_path' value - '{experiment_name}' already exists "
                        f"and new runs will be sent there"
                    )

    def log_metrics(
        self, metrics: Dict[str, Union[float, str]], step: Optional[int] = None
    ):
        if self.is_active:
            mlflow.log_metrics(metrics, step)

    def search_run(self, table_name: str, type_of_process: str):
        if self.is_active:
            run = mlflow.search_runs(
                filter_string=f"run_name = '{table_name}-{type_of_process}'"
            )
            return run["run_id"][0] if run.shape[0] > 0 else None
