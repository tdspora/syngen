from dataclasses import dataclass

import pandas as pd
from loguru import logger
from typing import Callable, Tuple, Dict


@dataclass
class DataFrameFetcher:
    """
    Class which fetches the dataframe by using the provided callback function
    """
    loader: Callable[[str], pd.DataFrame]
    table_name: str

    def fetch_data(self) -> Tuple[pd.DataFrame, Dict]:
        try:
            df = self.loader(self.table_name)
            default_schema = {"fields": {}, "format": "CSV"}
            logger.info(f"Successfully fetched dataframe for table: {self.table_name}")
            return df, default_schema
        except Exception as error:
            message = (
                "Failed to fetch the dataframe for the training process "
                "using the provided callback function. "
                f"Table: {self.table_name}, Error: {error}"
            )
            logger.error(message)
            raise
