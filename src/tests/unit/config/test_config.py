import pandas as pd
import pytest

from syngen.ml.config import TrainConfig, InferConfig
from syngen.ml.data_loaders import DataLoader
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


@pytest.mark.parametrize("input_batch_size, expected_batch_size", [
    (100, 100),
    (1000, 1000),
    (1100, 1000)
])
def test_init_train_config(input_batch_size, expected_batch_size, rp_logger):
    rp_logger.info(
        "Test the process of initialization of the instance of the class TrainConfig"
    )
    path_to_source = f"{DIR_NAME}/unit/config/fixtures/data_types_detection_set.csv"
    table_name = "test_table"
    metadata = {
        "test_table": {
            "train_settings": {
                "source": path_to_source,
            },
            "infer_settings": {},
            "encryption": {"fernet_key": None},
            "keys": {},
            "format": {}
        }
    }
    df, _ = DataLoader(path=path_to_source).load_data()
    train_config = TrainConfig(
        data=df,
        schema={"fields": {}, "format": "CSV"},
        source=path_to_source,
        epochs=10,
        drop_null=False,
        row_limit=None,
        table_name=table_name,
        metadata=metadata,
        metadata_path=None,
        reports=["accuracy", "sample"],
        batch_size=input_batch_size,
        loader=None
    )
    assert train_config.row_subset == 1000
    assert train_config.batch_size == expected_batch_size
    assert train_config.slugify_table_name == "test-table"
    assert set(train_config.__dict__.keys()) == {
        "source", "epochs", "data", "schema", "drop_null", "row_limit", "table_name", "metadata",
        "metadata_path", "reports", "batch_size", "loader", "slugify_table_name",
        "paths", "row_subset"
    }
    assert set(train_config.__getstate__().keys()) == {
        "source", "epochs", "drop_null", "row_limit", "table_name", "metadata",
        "metadata_path", "reports", "batch_size", "slugify_table_name",
        "paths", "schema", "row_subset"
    }

    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.fixture
def test_init_infer_config(mocker, rp_logger):
    rp_logger.info(
        "Test the process of initialization of the instance of the class InferConfig"
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


def test_get_state_of_train_config(test_df, rp_logger):
    rp_logger.info("Test the method '__getstate__' of the class TrainConfig")
    train_config = TrainConfig(
        source="data/source.csv",
        data=test_df,
        schema={"fields": {}, "format": "CSV"},
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
        "schema",
        "epochs",
        "drop_null",
        "row_limit",
        "row_subset",
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
