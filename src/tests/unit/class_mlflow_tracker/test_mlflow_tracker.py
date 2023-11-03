from unittest.mock import patch
from tests.conftest import SUCCESSFUL_MESSAGE
from syngen.ml.mlflow_tracker import MlflowTracker


def test_log_metric_with_active_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'log_metric' of the class 'MlflowTracker' with the active status"
    )
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.log_metric") as mock_log_metric:
        mlflow_tracker.log_metric("loss", 42, step=1)
        mock_log_metric.assert_called_once_with("loss", 42, 1)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_log_metric_with_inactive_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'log_metric' of the class 'MlflowTracker' with the inactive status"
    )
    MlflowTracker.reset_status(active_status=False)
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.log_metric") as mock_log_metric:
        mlflow_tracker.log_metric("loss", 42, step=1)
        mock_log_metric.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_log_metrics_with_active_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'log_metrics' of the class 'MlflowTracker' with the active status"
    )
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.log_metrics") as mock_log_metric:
        mlflow_tracker.log_metrics({"metric": 42}, 1)
        mock_log_metric.assert_called_once_with({"metric": 42}, 1)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_log_metrics_with_inactive_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'log_metrics' of the class 'MlflowTracker' with the inactive status"
    )
    MlflowTracker.reset_status(active_status=False)
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.log_metrics") as mock_log_metric:
        mlflow_tracker.log_metrics({"metric": 42}, 1)
        mock_log_metric.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_log_artifact_with_active_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'log_artifact' of the class 'MlflowTracker' with the active status"
    )
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.log_artifact") as mock_log_artifact:
        mlflow_tracker.log_artifact("path/to/local/storage", "path/to/artifacts")
        mock_log_artifact.assert_called_once_with("path/to/local/storage", "path/to/artifacts")
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_log_artifact_with_inactive_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'log_artifact' of the class 'MlflowTracker' with the inactive status"
    )
    MlflowTracker.reset_status(active_status=False)
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.log_artifact") as mock_log_artifact:
        mlflow_tracker.log_artifact("path/to/local/storage", "path/to/artifacts")
        mock_log_artifact.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_log_params_with_active_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'log_params' of the class 'MlflowTracker' with the active status"
    )
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.log_params") as mock_log_params:
        params = {"param1": 1, "param2": 2}
        mlflow_tracker.log_params(params)
        mock_log_params.assert_called_once_with(params)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_log_params_with_inactive_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'log_params' of the class 'MlflowTracker' with the inactive status"
    )
    MlflowTracker.reset_status(active_status=False)
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.log_params") as mock_log_params:
        params = {"param1": 1, "param2": 2}
        mlflow_tracker.log_params(params)
        mock_log_params.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_start_run_with_active_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'start_run' of the class 'MlflowTracker' with the active mlflow"
    )
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.start_run") as mock_start_run:
        mlflow_tracker.start_run(run_name="test_run | TRAIN")
        mock_start_run.assert_called_once_with(None, None, "test_run | TRAIN", False, None, None)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_start_run_with_inactive_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'start_run' of the class 'MlflowTracker' with the inactive mlflow"
    )
    MlflowTracker.reset_status(active_status=False)
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.start_run") as mock_start_run:
        mlflow_tracker.start_run(run_name="test_run | TRAIN")
        mock_start_run.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_end_run_with_active_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'end_run' of the class 'MlflowTracker' with the active mlflow"
    )
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.end_run") as mock_end_run:
        mlflow_tracker.end_run()
        mock_end_run.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_end_run_with_inactive_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'end_run' of the class 'MlflowTracker' with the inactive mlflow"
    )
    MlflowTracker.reset_status(active_status=False)
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.end_run") as mock_end_run:
        mlflow_tracker.end_run()
        mock_end_run.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_set_tracking_uri_with_active_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'set_tracking_uri' of the class 'MlflowTracker' "
        "with the active mlflow"
    )
    with patch(
            "syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.set_tracking_uri"
    ) as mock_set_tracking_uri:
        test_uri = "http://localhost"
        mlflow_tracker.set_tracking_uri(test_uri)
        mock_set_tracking_uri.assert_called_once_with(test_uri)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_set_tracking_uri_with_inactive_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'set_tracking_uri' of the class 'MlflowTracker' "
        "with the inactive mlflow"
    )
    MlflowTracker.reset_status(active_status=False)
    with patch(
            "syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.set_tracking_uri"
    ) as mock_set_tracking_uri:
        test_uri = "http://localhost"
        mlflow_tracker.set_tracking_uri(test_uri)
        mock_set_tracking_uri.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_create_experiment_with_active_status(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'create_experiment' of the class 'MlflowTracker' "
        "with the active mlflow"
    )
    with patch(
            "syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.create_experiment"
    ) as mock_create_experiment:
        test_experiment_name = "test_experiment"
        artifact_location = "path/to/artifacts"
        tags = {"tag1": 1, "tag2": 2}
        mlflow_tracker.create_experiment(test_experiment_name, artifact_location, tags)
        mock_create_experiment.assert_called_once_with(
            test_experiment_name,
            artifact_location,
            tags
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_create_experiment_with_inactive_mlflow(mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'create_experiment' of the class 'MlflowTracker' "
        "with the inactive mlflow"
    )
    MlflowTracker.reset_status(active_status=False)
    with patch(
            "syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.create_experiment"
    ) as mock_create_experiment:
        test_experiment_name = "test_experiment"
        artifact_location = "path/to/artifacts"
        tags = {"tag1": 1, "tag2": 2}
        mlflow_tracker.create_experiment(test_experiment_name, artifact_location, tags)
        mock_create_experiment.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.set_experiment")
@patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.search_experiments")
def test_set_experiment_with_valid_name_and_active_mlflow(
        mock_search_experiment, mock_set_experiment, mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'set_experiment' of the class 'MlflowTracker' "
        "with the valid experiment name and the active mlflow"
    )
    test_experiment_name = "test_experiment-2000-00-00 00:00:00"
    mlflow_tracker.set_experiment(test_experiment_name)
    mock_search_experiment.assert_called_once_with(
        filter_string="name LIKE 'test_experiment-%'"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.set_experiment")
@patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.search_experiments", return_value=[])
def test_set_experiment_with_valid_name_and_active_mlflow_(
        mock_search_experiment, mock_set_experiment, mlflow_tracker, rp_logger, caplog):
    rp_logger.info(
        "Test the method 'set_experiment' of the class 'MlflowTracker' "
        "with the valid experiment name and the active mlflow"
    )
    test_experiment_name = "test_experiment-2000-00-00 00:00:00"
    with caplog.at_level(level="WARNING"):
        mlflow_tracker.set_experiment(test_experiment_name)
        mock_search_experiment.assert_called_once_with(
            filter_string="name LIKE 'test_experiment-%'"
        )
        assert ("It seems that no experiment with a name starting "
                "with - 'test_experiment-' was found. A new experiment "
                "with the name  - 'test_experiment-2000-00-00 00:00:00' "
                "will be created") in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.set_experiment")
@patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.search_experiments")
def test_set_experiment_with_invalid_name_and_active_mlflow(
        mock_search_experiment, mock_set_experiment, mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'set_experiment' of the class 'MlflowTracker' "
        "with the invalid experiment name and the active mlflow"
    )
    test_experiment_name = "test_experiment"
    mlflow_tracker.set_experiment(test_experiment_name)
    mock_search_experiment.assert_called_once_with(
        filter_string="name LIKE 'test_experiment%'"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.set_experiment")
@patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.search_experiments")
def test_set_experiment_with_inactive_mlflow(
        mock_search_experiment, mock_set_experiment, mlflow_tracker, rp_logger):
    rp_logger.info(
        "Test the method 'set_experiment' of the class 'MlflowTracker' "
        "with the valid experiment name and the inactive mlflow"
    )
    MlflowTracker.reset_status(active_status=False)
    test_experiment_name = "test_experiment-2000-00-00 00:00:00"
    mlflow_tracker.set_experiment(test_experiment_name)
    mock_search_experiment.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)
