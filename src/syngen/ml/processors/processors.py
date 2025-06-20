import os
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional, Callable
import json
from json import JSONDecodeError
from slugify import slugify

from loguru import logger
import pandas as pd
import numpy as np
from flatten_json import flatten, unflatten_list

from syngen.ml.data_loaders import DataLoader, DataFrameFetcher
from syngen.ml.utils import fetch_unique_root
from syngen.ml.context import get_context


class Processor:
    """
    Base class for data preprocessing and postprocessing

    Attributes:
        metadata (Dict): Processing metadata
        metadata_path (Optional[str]): Path to metadata file
        table_name (Optional[str]): Name of the table being processed
        loader (Optional[Callable]): Custom data loader function
    """
    def __init__(
        self,
        metadata: Dict,
        metadata_path: Optional[str],
        table_name: Optional[str],
        loader: Optional[Callable[[str], pd.DataFrame]] = None
    ):
        self.metadata_path = metadata_path
        self.metadata = metadata
        self.loader = loader
        self.path_to_flatten_metadata = (
            "model_artifacts/system_store/flatten_configs/"
            f"flatten_metadata_{fetch_unique_root(table_name, self.metadata_path)}.json"
        )


class PreprocessHandler(Processor):
    """
    The class for the preprocessing of the data before the training process
    """

    @staticmethod
    def _run_script():
        """
        Run the script before the training process
        if it exists in the predefined path
        """
        path_to_script = f"{os.getcwd()}/model_artifacts/script.py"
        if os.path.exists(path_to_script):
            os.system(f"python3 {path_to_script}")

    @staticmethod
    def _get_json_columns(data: pd.DataFrame) -> List[str]:
        """
        Get the list of columns which contain JSON data
        Returns:
            List[str]: List of column names containing valid JSON data
        """

        def is_json_column(series: pd.Series) -> bool:
            if pd.isnull(series).all():
                return False
            try:
                series.dropna().apply(json.loads)
                return True
            except (TypeError, JSONDecodeError):
                return False

        return [col for col in data.columns if is_json_column(data[col])]

    @staticmethod
    def _get_artifacts(
        data: pd.DataFrame,
        json_columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
        """
        Flatten JSON columns in DataFrame and create mapping of original to flattened columns.

        Args:
            data (pd.DataFrame): Input DataFrame with JSON columns
            json_columns (List[str]): List of columns containing JSON data

        Returns:
            Tuple containing:
            - Flattened DataFrame
            - Mapping of original columns to flattened columns
            - List of duplicated column names
        """
        flattened_dfs = list()
        flattening_mapping = dict()
        for column in json_columns:
            flattened_data = pd.DataFrame(
                [
                    flatten(json.loads(i), ".")
                    for i in data[column]
                ]
            )
            flattening_mapping[column] = flattened_data.columns.to_list()
            flattened_dfs.append(flattened_data)
        flattened_data = pd.concat([data, *flattened_dfs], axis=1)
        duplicated_columns = [
            key
            for key, value in dict(Counter(flattened_data.columns.to_list())).items()
            if value > 1
        ]
        flattened_data.drop(columns=flattening_mapping.keys(), inplace=True)
        flattened_df = flattened_data.T.loc[~flattened_data.T.index.duplicated(), :].T
        flattened_df = flattened_df.applymap(lambda x: np.NaN if x in [list(), dict()] else x)
        return flattened_df, flattening_mapping, duplicated_columns

    def _save_flatten_metadata(self, metadata: Dict):
        """
        Save the metadata of the flattening process
        """
        with open(f"{self.path_to_flatten_metadata}", "w") as f:
            json.dump(metadata, f)

    def _load_source(self, path_to_source: str, table_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load the data from the predefined source
        """
        if self.loader is not None:
            dataframe_fetcher = DataFrameFetcher(
                loader=self.loader,
                table_name=table_name
            )
            return dataframe_fetcher.fetch_data()
        return DataLoader(path=path_to_source).load_data()

    def _save_input_data(self, flattened_data: pd.DataFrame, table_name: str):
        """
        Save the input data to the predefined path
        """
        fernet_key = self.metadata[table_name].get("encryption", {}).get("fernet_key", None)
        path_to_input_data = (
            f"model_artifacts/tmp_store/{slugify(table_name)}/"
            f"input_data_{slugify(table_name)}.{'dat' if fernet_key else 'pkl'}"
        )
        DataLoader(
            path=path_to_input_data,
            table_name=table_name,
            metadata=self.metadata,
            sensitive=True
        ).save_data(flattened_data)

    def _handle_json_columns(self):
        """
        Preprocess the data contained JSON columns before the training process
        """
        flatten_metadata = dict()
        for table, settings in self.metadata.items():
            path_to_source = settings.get("train_settings", {}).get("source", "")
            data, schema = self._load_source(path_to_source, table)
            order_of_columns = data.columns.to_list()
            if json_columns := self._get_json_columns(data):
                list_of_json_columns = [f"'{column}'" for column in json_columns]
                list_of_json_columns = ', '.join(list_of_json_columns)
                logger.info(
                    f"The table '{table}' contains JSON columns: {list_of_json_columns}"
                )
                logger.info(f"Flattening the JSON columns in the table - '{table}'")
                (flattened_data,
                 flattening_mapping,
                 duplicated_columns) = self._get_artifacts(data, json_columns)
                self._save_input_data(flattened_data, table)
                flatten_metadata[table] = {
                    "flattening_mapping": flattening_mapping,
                    "duplicated_columns": duplicated_columns,
                    "order_of_columns": order_of_columns
                }
                logger.info(f"The table '{table}' has been successfully flattened")
        if flatten_metadata:
            self._save_flatten_metadata(flatten_metadata)

    def run(self):
        """
        Launch the preprocessing process:
        """
        self._run_script()
        self._handle_json_columns()


class PostprocessHandler(Processor):
    """
    Class for the postprocessing of the generated data
    """

    def __init__(
        self,
        metadata: Dict,
        metadata_path: Optional[str],
        table_name: Optional[str],
    ):
        super().__init__(metadata, metadata_path, table_name)

    def _fetch_flatten_config(self, table_name: str) -> Dict:
        """
        Fetch the metadata of the flattening process
        """
        with open(self.path_to_flatten_metadata, "r") as f:
            flatten_metadata = json.load(f).get(table_name)
            return flatten_metadata

    @staticmethod
    def _check_none_values(x: Any) -> bool:
        """
        Check if the value should be treated as null/none.

        Args:
            x: Value to check

        Returns:
            bool: True if value should be treated as null/none

        Note:
            Handles None, np.NaN, and empty containers as null values
        """
        if x is None:
            return True
        if isinstance(x, float) and np.isnan(x):
            return True
        if isinstance(x, (list, dict)) and not x:
            return True
        return False

    def _remove_none_from_struct(self, input_data: Any) -> Any:
        """
        Remove np.NaN or None values from nested data structures.

        Args:
            input_data: Input data structure (list, dict, or scalar)

        Returns:
            Cleaned data structure with None values removed
        """
        if isinstance(input_data, list):
            # Process list elements and filter empty results
            return [
                self._remove_none_from_struct(v)
                for v in input_data
                if not self._check_none_values(v)
            ]

        if isinstance(input_data, dict):
            # Process dictionary items and filter empty results
            return {
                k: self._remove_none_from_struct(v)
                for k, v in input_data.items()
                if not self._check_none_values(v)
            }
        return input_data

    @staticmethod
    def _restore_empty_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Restore empty dictionary values in the dataframe for nested fields.

        Args:
            df: Input DataFrame with potentially empty nested fields

        Returns:
            pd.DataFrame: DataFrame with restored empty dictionary values
        """
        df = df.copy().astype(object)
        columns = df.columns.tolist()

        # Find potential parent columns
        parent_columns = set()
        for col in columns:
            for potential_parent in columns:
                if f"{potential_parent}." in col:
                    parent_columns.add(potential_parent)

        # Restore empty dictionary values for parent columns
        for parent_col in parent_columns:
            df[parent_col] = df[parent_col].apply(lambda val: {} if pd.isna(val) else val)

        return df

    @staticmethod
    def _load_generated_data(path_to_generated_data: str, *args) -> pd.DataFrame:
        """
        Load generated data from the predefined path
        """
        data, schema = DataLoader(path=path_to_generated_data).load_data()
        return data

    @staticmethod
    def _remove_empty_elements(d: dict) -> dict:
        """
        Recursively remove keys with empty dictionaries or lists from a nested dictionary

        Args:
            d (dict): Input dictionary to clean

        Returns:
            dict: Cleaned dictionary with empty structures removed

        Note:
            This method preserves non-empty values including False, 0, and empty strings
        """

        def clean(data):
            if isinstance(data, dict):
                cleaned = {
                    key: cleaned_value
                    for key, value in data.items()
                    if (cleaned_value := clean(value)) not in ({}, [])
                }
                return cleaned

            if isinstance(data, list):
                cleaned = [
                    cleaned_item
                    for item in data
                    if (cleaned_item := clean(item)) not in ({}, [])
                ]
                return cleaned

            return data

        return clean(d)

    def _postprocess_generated_data(
        self,
        data: pd.DataFrame,
        flattening_mapping: Dict,
        duplicated_columns: List
    ) -> pd.DataFrame:
        """
        Postprocess the generated data
        """
        for old_column, new_columns in flattening_mapping.items():
            data[new_columns] = self._restore_empty_values(data[new_columns])

            data[old_column] = data[new_columns].apply(
                lambda row: unflatten_list(row.to_dict(), "."), axis=1
            )
            data[old_column] = data[old_column]. \
                apply(lambda row: self._remove_none_from_struct(row))
            data[old_column] = data[old_column]. \
                apply(lambda row: self._remove_empty_elements(row))
            data[old_column] = data[old_column]. \
                apply(lambda row: json.dumps(row, ensure_ascii=False))
            dropped_columns = set(i for i in new_columns if i not in duplicated_columns)
            data.drop(dropped_columns, axis=1, inplace=True)
        return data

    def run(self):
        """
        Launch the postprocessing of generated data,
        and save the processed data to the predefined path
        """

        if os.path.exists(self.path_to_flatten_metadata):
            logger.info("Start postprocessing of the generated data")
            for table in self.metadata.keys():
                flatten_metadata = self._fetch_flatten_config(table)
                flattening_mapping = flatten_metadata.get("flattening_mapping")
                duplicated_columns = flatten_metadata.get("duplicated_columns")
                order_of_columns = flatten_metadata.get("order_of_columns")
                path_to_generated_data = (
                    f"model_artifacts/tmp_store/{slugify(table)}/"
                    f"merged_infer_{slugify(table)}.csv"
                )
                data = self._load_generated_data(path_to_generated_data, table)
                data = self._postprocess_generated_data(
                    data,
                    flattening_mapping,
                    duplicated_columns
                )
                destination = self.metadata[table].get("infer_settings", {}).get("destination", "")
                path_to_destination = destination if destination else path_to_generated_data
                self._save_generated_data(data, path_to_destination, order_of_columns, table)
                logger.info("Finish postprocessing of the generated data")

    @staticmethod
    def _save_generated_data(
        generated_data: pd.DataFrame,
        path_to_destination: str,
        order_of_columns: List[str],
        *args
    ):
        """
        Save generated data to the path
        """
        generated_data = generated_data[order_of_columns]
        DataLoader(path=path_to_destination).save_data(
            generated_data,
            format=get_context().get_config(),
        )
