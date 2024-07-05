import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from loguru import logger

from syngen.ml.data_loaders import DataFrameFetcher
from tests.conftest import SUCCESSFUL_MESSAGE

SAMPLE = {
    "Column1": list(range(0, 5)),
    "Column2": list(range(5, 10)),
    "Column3": ["A", "B", "C", "D", "E"],
    "Column4": list(np.arange(0, 1, 0.2)),
    "Column5": [True, False, True, True, False]
}


def get_dataframe(table_name):
    logger.info(f"Fetched the dataframe of the table - {table_name}")
    df = pd.DataFrame(SAMPLE)
    return df


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
    fetched_df, default_schema = fetcher.fetch_data()
    assert (
            assert_frame_equal(
                fetched_df,
                pd.DataFrame(SAMPLE),
            )
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
            fetcher.fetch_data()
            assert ("Failed to fetch the dataframe for the training process "
                    "using the provided callback function") in caplog.text

    rp_logger.info(SUCCESSFUL_MESSAGE)
