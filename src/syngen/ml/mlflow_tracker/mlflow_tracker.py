from typing import Optional, Dict, Any, Union
import os
import requests
from statistics import mean, StatisticsError

import mlflow
from loguru import logger
from syngen.ml.utils import fetch_unique_root


HARDWARE_METRICS = [
    "system/cpu_utilization_percentage",
    "system/system_memory_usage_percentage",
    "system/system_memory_usage_megabytes",
    "system/gpu_utilization_percentage",
    "system/gpu_memory_usage_percentage",
    "system/gpu_memory_usage_megabytes"
]


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

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a metric to the current run
        """
        if self.is_active:
            mlflow.log_metric(key, value, step)

    def log_duration(self, run_id, table, stage):
        """
        Log the duration of the run
        """
        try:
            run_info = self.get_run(run_id).info
            duration = (run_info.end_time - run_info.start_time) / 1000
            self.log_metric(key=f'{table}-{stage}-duration', value=duration)
        except AttributeError:
            pass

    def log_hardware_metrics(self, run_id, table, stage):
        """
        Log the hardware metrics of the run
        """
        for metric in HARDWARE_METRICS:
            try:
                mean_value = mean([m.value for m in self.get_metric_history(run_id, metric)])
                self.log_metric(key=f'{table}-{stage}-{metric}', value=mean_value)
            except (TypeError, StatisticsError):
                pass

    def collect_metrics(self, table, stage):
        """
        Collect the duration of the run
        and hardware metrics related to the run
        """
        runs = self.search_runs(table_name=table, type_of_process=stage)
        for run_id in runs:
            self.log_duration(run_id, table, stage)
            self.log_hardware_metrics(run_id, table, stage)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact from the local path to the current run
        """
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
            mlflow.start_run(
                run_id=run_id,
                experiment_id=experiment_id,
                run_name=run_name,
                nested=nested,
                tags=tags,
                description=description
            )

    def get_last_run(self):
        if self.is_active:
            return mlflow.last_active_run()

    def end_run(self):
        if self.is_active:
            mlflow.end_run()

    def set_tracking_uri(self, uri: str):
        """
        Set the tracking URI
        """
        if self.is_active:
            mlflow.set_tracking_uri(uri)

    def create_experiment(self, name: str, artifact_location: Optional[str] = None):
        if self.is_active:
            mlflow.create_experiment(name, artifact_location)

    @staticmethod
    def get_experiment(experiment_name: str):
        """
        Get the last matching of the experiment by name
        """
        experiments = mlflow.search_experiments(
            filter_string=f"name LIKE '{experiment_name}'"
        )
        return experiments[0] if experiments else []

    def set_experiment(self, experiment_name: str = None, experiment_id: str = None):
        """
        Set the experiment for tracking.
        If the experiment name is not provided, the last experiment will be used.
        """
        if self.is_active:
            last_matching = self.get_experiment(experiment_name)
            env_value = os.getenv("MLFLOW_EXPERIMENT_NAME", "")
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

    def search_runs(self, table_name: str, type_of_process: str):
        """
        Get the list of runs related the certain experment
        """
        if self.is_active:
            run = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                filter_string=f"run_name like '{table_name}-{type_of_process}%'"
            )
            try:
                count_runs = 2 if run.at[0, "tags.mlflow.runName"].endswith("-2") else 1
                return run["run_id"][:count_runs].to_list()
            except KeyError:
                return []
        else:
            return []

    def get_run(self, run_id):
        """
        Get the run by the run_id
        """
        if self.is_active:
            return mlflow.get_run(run_id)

    def get_metric_history(self, run_id, metric_key):
        if self.is_active:
            return mlflow.tracking.MlflowClient().get_metric_history(run_id, metric_key)
