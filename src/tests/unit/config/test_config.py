import pandas as pd

from syngen.ml.config import TrainConfig
from tests.conftest import SUCCESSFUL_MESSAGE


def test_get_state_of_train_config(rp_logger):
    rp_logger.info("Test the method '__getstate__' of the class TrainConfig")
    train_config = TrainConfig(
        source="data/source.csv",
        epochs=10,
        drop_null=True,
        row_limit=1000,
        table_name="test_table",
        metadata_path="metadata/path.yaml",
        print_report=True,
        batch_size=32,
        loader=lambda x: pd.DataFrame()
    )
    expected_attributes = {
        "source",
        "epochs",
        "drop_null",
        "row_limit",
        "table_name",
        "metadata_path",
        "print_report",
        "batch_size"
    }
    state = train_config.__getstate__()
    for attr in expected_attributes:
        assert attr in state
    assert "loader" not in state
    rp_logger.info(SUCCESSFUL_MESSAGE)
