import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from loguru import logger

from syngen.ml.data_loaders import DataFrameFetcher
from tests.conftest import SUCCESSFUL_MESSAGE, get_dataframe

SAMPLE = pd.DataFrame([{"id": 1, "value": 30}, {"id": 2, "value": 40}])


def invalid_function(table_name):
    logger.info(f"Fetched the dataframe of the table - {table_name}")
    raise ValueError("Produced 'ValueError' for testing purposes")


def test_fetch_dataframe_with_valid_callback_function(rp_logger):
    """
    Test the class DataFrameFetcher with provided valid callback function
    """
    rp_logger.info(
       "Test the class DataFrameFetcher with provided valid callback function"
    )
    fetcher = DataFrameFetcher(table_name="test_table", loader=get_dataframe)
    fetched_df, default_schema = fetcher.load_data()
    assert (
        assert_frame_equal(fetched_df, SAMPLE)
        is None
    )

    assert default_schema == {"fields": {}, "format": "CSV"}

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_fetch_dataframe_with_invalid_callback_function(caplog, rp_logger):
    """
    Test the class DataFrameFetcher with provided invalid callback function
    """
    rp_logger.info(
       "Test the class DataFrameFetcher with provided valid callback function"
    )
    fetcher = DataFrameFetcher(table_name="test_table", loader=invalid_function)
    with pytest.raises(ValueError):
        with caplog.at_level("ERROR"):
            fetcher.load_data()
            assert ("Failed to fetch the dataframe for the training process "
                    "using the provided callback function") in caplog.text

    rp_logger.info(SUCCESSFUL_MESSAGE)
