from unittest.mock import patch


def test_log_metric(mlflow_tracker):
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.log_metric") as mock_log_metric:
        mlflow_tracker.log_metric("loss", 42, step=1)
        mock_log_metric.assert_called_once_with("loss", 42, 1)


def test_log_metrics(mlflow_tracker):
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.log_metrics") as mock_log_metric:
        mlflow_tracker.log_metrics({"metric": 42}, 1)
        mock_log_metric.assert_called_once_with({"metric": 42}, 1)


def test_log_artifact(mlflow_tracker):
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.log_artifact") as mock_log_artifact:
        mlflow_tracker.log_artifact("path/to/local/storage", "path/to/artifacts")
        mock_log_artifact.assert_called_once_with("path/to/local/storage", "path/to/artifacts")


def test_log_params(mlflow_tracker):
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.log_params") as mock_log_params:
        params = {"param1": 1, "param2": 2}
        mlflow_tracker.log_params(params)
        mock_log_params.assert_called_once_with(params)


def test_start_run(mlflow_tracker):
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.start_run") as mock_start_run:
        mlflow_tracker.start_run(run_name="test_run | TRAIN")
        mock_start_run.assert_called_once_with(None, None, 'test_run | TRAIN', False, None, None)


def test_end_run(mlflow_tracker):
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.end_run") as mock_end_run:
        mlflow_tracker.end_run()
        mock_end_run.assert_called_once()


def test_set_tags(mlflow_tracker):
    with patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.set_tags") as mock_set_tags:
        tags = {"tag1": 1, "tag2": 2}
        mlflow_tracker.set_tags(tags)
        mock_set_tags.assert_called_once_with(tags)


def test_set_tracking_uri(mlflow_tracker):
    with patch(
            "syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.set_tracking_uri"
    ) as mock_set_tracking_uri:
        test_uri = "http://localhost"
        mlflow_tracker.set_tracking_uri(test_uri)
        mock_set_tracking_uri.assert_called_once_with(test_uri)


def test_create_experiment(mlflow_tracker):
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


@patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.set_experiment")
@patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.search_experiments")
def test_set_experiment_with_proper_name(
        mock_search_experiment, mock_set_experiment, mlflow_tracker):
    test_experiment_name = "test_experiment-2000-00-00 00:00:00"
    mlflow_tracker.set_experiment(test_experiment_name)
    mock_search_experiment.assert_called_once_with(
        filter_string="name LIKE 'test_experiment-%'"
    )


@patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.set_experiment")
@patch("syngen.ml.mlflow_tracker.mlflow_tracker.mlflow.search_experiments")
def test_set_experiment_with_wrong_name(
        mock_search_experiment, mock_set_experiment, mlflow_tracker):
    test_experiment_name = "test_experiment"
    mlflow_tracker.set_experiment(test_experiment_name)
    mock_search_experiment.assert_called_once_with(
        filter_string="name LIKE 'test_experiment%'"
    )
