import pandas as pd
import pytest
from unittest.mock import patch

from syngen.ml.config import TrainConfig, InferConfig
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


@pytest.mark.parametrize(
    "drop_null, row_limit, expected_row_subset, expected_reports",
    [
        (True, None, 801, ["accuracy", "sample"]),
        (False, 100, 100, ["accuracy", "sample"]),
        (False, None, 1000, ["accuracy"])
    ]
)
@patch.object(TrainConfig, "_save_input_data")
def test_init_train_config(
    mock_save_input_data,
    drop_null,
    row_limit,
    expected_row_subset,
    expected_reports,
    rp_logger
):
    rp_logger.info(
        "Test the process of initialization of the instance of the class TrainConfig"
    )
    path_to_source = f"{DIR_NAME}/unit/config/fixtures/data_types_detection_set.csv"
    table_name = "test_table"
    metadata = {
        "test_table": {
            "train_settings": {
                "source": path_to_source
            }
        }
    }
    train_config = TrainConfig(
        source=path_to_source,
        epochs=10,
        drop_null=drop_null,
        row_limit=row_limit,
        table_name=table_name,
        metadata=metadata,
        metadata_path=None,
        reports=expected_reports,
        batch_size=32,
        loader=None
    )
    train_config.preprocess_data()
    assert train_config.source == path_to_source
    assert train_config.epochs == 10
    assert train_config.drop_null == drop_null
    assert train_config.row_limit == row_limit
    assert train_config.table_name == table_name
    assert train_config.metadata == metadata
    assert train_config.metadata_path is None
    assert train_config.reports == expected_reports
    assert train_config.batch_size == 32
    assert train_config.loader is None
    assert train_config.initial_data_shape == (1000, 11)
    assert train_config.row_subset == expected_row_subset
    assert train_config.schema == {"fields": {}, "format": "CSV"}
    assert train_config.original_schema is None
    assert train_config.slugify_table_name == "test-table"
    assert train_config.columns == [
        "id", "first_name", "last_name", "email",
        "gender", "gender_abbr", "gender_abbr_3",
        "age", "price", "date", "comments"
    ]
    assert train_config.dropped_columns == set()
    assert set(train_config.__dict__.keys()) == {
        "source", "epochs", "drop_null", "row_limit", "table_name", "metadata",
        "metadata_path", "reports", "batch_size", "loader", "slugify_table_name",
        "paths", "original_schema", "data", "schema", "initial_data_shape",
        "columns", "dropped_columns", "row_subset"
    }
    assert set(train_config.__getstate__().keys()) == {
        "source", "epochs", "drop_null", "row_limit", "table_name", "metadata",
        "metadata_path", "reports", "batch_size", "slugify_table_name",
        "paths", "original_schema", "schema", "initial_data_shape",
        "columns", "dropped_columns", "row_subset"
    }

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_init_infer_config_with_absent_input_data_in_train_process(rp_logger):
    rp_logger.info(
        "Test the process of initialization of the instance of the class InferConfig "
        "during the training process in case the input data is absent"
    )
    table_name = "test_table"
    path_to_source = "path/to/source.csv"
    metadata = {
        "test_table": {
            "train_settings": {
                "source": path_to_source,
                "reports": ["accuracy"]
            }
        }
    }
    infer_config = InferConfig(
        destination="path/to/destination.csv",
        metadata=metadata,
        metadata_path="path/to/metadata.yaml",
        size=100,
        table_name=table_name,
        run_parallel=False,
        batch_size=100,
        random_seed=None,
        reports=["accuracy"],
        both_keys=True,
        log_level="DEBUG",
        loader=None,
        type_of_process="train"
    )
    assert infer_config.reports == []

    assert set(infer_config.__dict__.keys()) == {
        "destination", "metadata", "metadata_path", "size", "table_name",
        "run_parallel", "batch_size", "random_seed", "reports", "both_keys",
        "log_level", "loader", "type_of_process", "slugify_table_name", "paths"
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_init_infer_config_with_absent_input_data_in_infer_process(rp_logger):
    rp_logger.info(
        "Test the process of initialization of the instance of the class InferConfig "
        "during the inference process in case the input data is absent"
    )
    table_name = "test_table"
    path_to_source = "path/to/source.csv"
    metadata = {
        "test_table": {
            "train_settings": {
                "source": path_to_source
            },
            "infer_settings": {
                "reports": ["accuracy"]
            }
        }
    }
    infer_config = InferConfig(
        destination="path/to/destination.csv",
        metadata=metadata,
        metadata_path="path/to/metadata.yaml",
        size=100,
        table_name=table_name,
        run_parallel=False,
        batch_size=100,
        random_seed=None,
        reports=["accuracy"],
        both_keys=True,
        log_level="DEBUG",
        loader=None,
        type_of_process="infer"
    )
    assert infer_config.reports == []

    assert set(infer_config.__dict__.keys()) == {
        "destination", "metadata", "metadata_path", "size", "table_name",
        "run_parallel", "batch_size", "random_seed", "reports", "both_keys",
        "log_level", "loader", "type_of_process", "slugify_table_name", "paths"
    }

    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.fixture
def test_init_infer_config_with_existed_input_data_in_train_process(mocker, rp_logger):
    rp_logger.info(
        "Test the process of initialization of the instance of the class InferConfig "
        "during the training process in case the input data is present"
    )
    table_name = "test_table"
    path_to_source = "path/to/source.csv"

    metadata = {
        "test_table": {
            "train_settings": {
                "source": path_to_source,
                "reports": ["accuracy"]
            }
        }
    }

    mocker.patch("syngen.ml.data_loaders.DataLoader.has_existed_path", return_value=True)

    infer_config = InferConfig(
        destination="path/to/destination.csv",
        metadata=metadata,
        metadata_path="path/to/metadata.yaml",
        size=100,
        table_name=table_name,
        run_parallel=False,
        batch_size=100,
        random_seed=None,
        reports=["accuracy"],
        both_keys=True,
        log_level="DEBUG",
        loader=None,
        type_of_process="train"
    )

    assert infer_config.reports == ["accuracy"]

    assert set(infer_config.__dict__.keys()) == {
        "destination", "metadata", "metadata_path", "size", "table_name",
        "run_parallel", "batch_size", "random_seed", "reports", "both_keys",
        "log_level", "loader", "type_of_process", "slugify_table_name", "paths"
    }

    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.fixture
def test_init_infer_config_with_existed_input_data_in_infer_process(mocker, rp_logger):
    rp_logger.info(
        "Test the process of initialization of the instance of the class InferConfig "
        "during the inference process in case the input data is present"
    )
    table_name = "test_table"

    metadata = {
        "test_table": {
            "train_settings": {
                "source": "path/to/source.csv",
            },
            "infer_settings": {
                "reports": ["accuracy"]
            }
        }
    }

    mocker.patch("syngen.ml.data_loaders.DataLoader.has_existed_path", return_value=True)

    infer_config = InferConfig(
            destination="path/to/destination.csv",
            metadata=metadata,
            metadata_path="path/to/metadata.yaml",
            size=100,
            table_name=table_name,
            run_parallel=False,
            batch_size=100,
            random_seed=None,
            reports=["accuracy"],
            both_keys=True,
            log_level="DEBUG",
            loader=None,
            type_of_process="infer"
        )

    assert infer_config.reports == ["accuracy"]

    assert set(infer_config.__dict__.keys()) == {
        "destination", "metadata", "metadata_path", "size", "table_name",
        "run_parallel", "batch_size", "random_seed", "reports", "both_keys",
        "log_level", "loader", "type_of_process", "slugify_table_name", "paths"
    }

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_get_state_of_train_config(rp_logger):
    rp_logger.info("Test the method '__getstate__' of the class TrainConfig")
    train_config = TrainConfig(
        source="data/source.csv",
        epochs=10,
        drop_null=True,
        row_limit=1000,
        table_name="test_table",
        metadata={
            "test_table": {
                "train_settings": {
                    "source": "path/to/data.csv"
                }
            }
        },
        metadata_path=None,
        reports=["accuracy", "sample"],
        batch_size=32,
        loader=lambda x: pd.DataFrame()
    )
    expected_attributes = {
        "source",
        "epochs",
        "drop_null",
        "row_limit",
        "table_name",
        "metadata",
        "metadata_path",
        "reports",
        "batch_size",
        "paths",
        "slugify_table_name",
    }
    state = train_config.__getstate__()
    assert set(state.keys()) == expected_attributes
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("drop_null, row_limit, expected_size", [
    (False, None, 1000),
    (True, None, 801),
    (True, 100, 100)
])
@patch.object(TrainConfig, "_save_original_schema")
@patch.object(TrainConfig, "_remove_empty_columns")
@patch.object(TrainConfig, "_mark_removed_columns")
@patch.object(TrainConfig, "_save_input_data")
def test_preprocess_data(
    mock_save_input_data,
    mock_mark_removed_columns,
    mock_remove_empty_columns,
    mock_save_original_schema,
    drop_null,
    row_limit,
    expected_size,
    rp_logger
):
    rp_logger.info("Test the method '_preprocess_data' of the class TrainConfig")
    train_config = TrainConfig(
        source=f"{DIR_NAME}/unit/config/fixtures/data_types_detection_set.csv",
        epochs=10,
        drop_null=drop_null,
        row_limit=row_limit,
        table_name="test_table",
        metadata={
            "test_table": {
                "train_settings": {
                    "source": "path/to/data.csv"
                }
            }
        },
        metadata_path=None,
        reports=["accuracy", "sample"],
        batch_size=32,
        loader=None
    )
    train_config.preprocess_data()
    mock_save_input_data.assert_called_once()
    mock_save_original_schema.assert_called_once()
    mock_mark_removed_columns.assert_called_once()
    mock_remove_empty_columns.assert_called_once()
    train_config.row_subset == expected_size
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("drop_null, row_limit, expected_size, expected_metadata", [
    (False, None, 1000, {
            "test_table": {
                "train_settings": {
                    "source": "path/to/data.csv",
                    "reports": ["accuracy"]
                }
            }
        }),
    (True, None, 801, {
            "test_table": {
                "train_settings": {
                    "source": "path/to/data.csv",
                    "reports": ["accuracy", "sample"]
                }
            }
        }),
    (True, 100, 100, {
            "test_table": {
                "train_settings": {
                    "source": "path/to/data.csv",
                    "reports": ["accuracy", "sample"]
                }
            }
        })
])
@patch.object(TrainConfig, "_save_original_schema")
@patch.object(TrainConfig, "_remove_empty_columns")
@patch.object(TrainConfig, "_mark_removed_columns")
@patch.object(TrainConfig, "_save_input_data")
def test_check_reports_in_train_config(
    mock_save_input_data,
    mock_mark_removed_columns,
    mock_remove_empty_columns,
    mock_save_original_schema,
    drop_null,
    row_limit,
    expected_size,
    expected_metadata,
    rp_logger
):
    rp_logger.info("Test the method '_check_reports' of the class TrainConfig")
    train_config = TrainConfig(
        source=f"{DIR_NAME}/unit/config/fixtures/data_types_detection_set.csv",
        epochs=10,
        drop_null=drop_null,
        row_limit=row_limit,
        table_name="test_table",
        metadata={
            "test_table": {
                "train_settings": {
                    "source": "path/to/data.csv",
                    "reports": ["accuracy", "sample"]
                }
            }
        },
        metadata_path=None,
        reports=["accuracy", "sample"],
        batch_size=32,
        loader=None
    )
    train_config.preprocess_data()
    mock_save_input_data.assert_called_once()
    mock_save_original_schema.assert_called_once()
    mock_mark_removed_columns.assert_called_once()
    mock_remove_empty_columns.assert_called_once()
    train_config.row_subset == expected_size
    train_config.metadata = expected_metadata
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_check_reports_in_infer_config_with_absent_input_data(rp_logger):
    rp_logger.info(
        "Test the method '_check_reports' of the class InferConfig "
        "in case the input data of the table is absent"
    )
    infer_config = InferConfig(
        destination=None,
        metadata={
            "test_table": {
                "train_settings": {
                    "source": "path/to/data.csv",
                    "reports": ["accuracy", "sample"]
                }
            }
        },
        metadata_path=None,
        size=100,
        table_name="test_table",
        run_parallel=False,
        batch_size=32,
        random_seed=None,
        reports=["accuracy"],
        both_keys=False,
        loader=None,
        log_level="DEBUG",
        type_of_process="infer"
    )
    infer_config._check_reports()
    assert infer_config.reports == []
    rp_logger.info(SUCCESSFUL_MESSAGE)
