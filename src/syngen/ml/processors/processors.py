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
        table_name: str,
        loader: Optional[Callable[[str], pd.DataFrame]] = None
    ):
        self.metadata_path = metadata_path
        self.metadata = metadata
        self.table_name = table_name
        self.loader = loader
        self.path_to_flatten_metadata = (
            "model_artifacts/system_store/flatten_configs/"
            f"flatten_metadata_{fetch_unique_root(self.table_name, self.metadata_path)}.json"
        )
        self.initial_data_shape: Tuple = ()
        self.row_subset: int = int()


class PreprocessHandler(Processor):
    """
    The class for the preprocessing of the data before the training process
    """
    def _save_original_schema(self, original_schema: Dict):
        """
        Save the schema of the original data
        """
        path = (
            f"model_artifacts/tmp_store/{slugify(self.table_name)}"
            f"/original_schema_{slugify(self.table_name)}.pkl"
        )
        DataLoader(path).save_data(data=original_schema)

    def _check_if_data_is_empty(self, data: pd.DataFrame):
        """
        Check if the provided data is empty or not
        """
        if data.shape[0] < 1:
            raise ValueError(
                f"The empty table was provided. Unable to train the table - '{self.table_name}'"
            )

    @staticmethod
    def _remove_empty_columns(data: pd.DataFrame, schema: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Remove completely empty columns from dataframe and mark them as 'removed' in the schema
        """
        initial_data_columns = set(data.columns)
        data = data.dropna(how="all", axis=1)

        dropped_columns = initial_data_columns - set(data.columns)
        list_of_dropped_columns = [f"'{column}'" for column in dropped_columns]
        if list_of_dropped_columns:
            logger.info(f"Empty columns - {', '.join(list_of_dropped_columns)} were removed")
            schema["fields"].update({column: "removed" for column in dropped_columns})
        return data, schema

    def prepare_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare the subset of the data for the training process,
        and get the preprocessed data and schema
        """
        data, schema, original_schema = self._load_source()
        self._check_if_data_is_empty(data)
        self._save_original_schema(original_schema)
        preprocessed_data, preprocessed_schema = self._preprocess_data(data, schema)
        return preprocessed_data, preprocessed_schema

    def _preprocess_data(self, data: pd.DataFrame, schema: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply the parameters 'drop_null' and 'row_limit' to the data,
        and get the subset for the training process, and modified schema
        """
        drop_null = self.metadata[self.table_name]["train_settings"]["drop_null"]
        row_limit = self.metadata[self.table_name]["train_settings"]["row_limit"]
        self.initial_data_shape = data.shape
        self.row_subset = len(data)
        data, schema = self._remove_empty_columns(data, schema)
        if self.loader:
            warning_message = (
                "parameter will be ignored because the retrieval of the data "
                "is handled by a callback function"
            )
            if drop_null:
                logger.warning(f"The 'drop_null' {warning_message}")
            if row_limit is not None:
                logger.warning(f"The 'row_limit' {warning_message}")
        else:
            if drop_null:
                if not data.dropna().empty:
                    data = data.dropna()
                    if count_of_dropped_rows := self.initial_data_shape[0] - data.shape[0]:
                        logger.info(
                            f"As the parameter 'drop_null' set to 'True', "
                            f"{count_of_dropped_rows} rows of the table - '{self.table_name}' "
                            f"that have empty values have been dropped. "
                            f"The count of remained rows is {data.shape[0]}."
                        )
                else:
                    logger.warning(
                        "The specified 'drop_null' argument results in the empty dataframe, "
                        "so it will be ignored"
                    )
                self.row_subset = len(data)

            if row_limit:
                self.row_subset = min(row_limit, len(data))
                data = data.sample(n=self.row_subset)

        if len(data) < 100:
            logger.warning(
                "The input table is too small to provide any meaningful results. "
                "Please consider: 1) disable drop_null argument, 2) provide bigger table"
            )
        elif len(data) < 500:
            logger.warning(
                f"The amount of data is {len(data)} rows. It seems that it isn't enough "
                f"to supply high-quality results. To improve the quality of generated data "
                f"please consider any of the steps: 1) provide a bigger table, "
                f"2) disable drop_null argument"
            )
        logger.info(f"The subset of rows was set to {self.row_subset}")
        return data, schema

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
                ],
                index=data.index
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
        if os.path.exists(self.path_to_flatten_metadata):
            with open(self.path_to_flatten_metadata, "r") as f:
                existing_metadata = json.load(f)
            metadata = {**existing_metadata, **metadata}
        with open(f"{self.path_to_flatten_metadata}", "w") as f:
            json.dump(metadata, f)

    def _load_source(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Load the data from the predefined source
        """
        if self.loader is not None:
            dataframe_fetcher = DataFrameFetcher(
                loader=self.loader,
                table_name=self.table_name
            )
            original_schema = dataframe_fetcher.original_schema
            data, schema = dataframe_fetcher.fetch_data()
            return data, schema, original_schema
        path_to_source = self.metadata[self.table_name]["train_settings"]["source"]
        data, schema = DataLoader(path=path_to_source).load_data()
        original_schema = DataLoader(path=path_to_source).original_schema
        return data, schema, original_schema

    def _handle_json_columns(self, data: pd.DataFrame):
        """
        Preprocess the data contained JSON columns before the training process
        """
        flatten_metadata = dict()
        order_of_columns = data.columns.to_list()
        if json_columns := self._get_json_columns(data):
            list_of_json_columns = [f"'{column}'" for column in json_columns]
            list_of_json_columns = ', '.join(list_of_json_columns)
            logger.info(
                f"The table '{self.table_name}' contains JSON columns: {list_of_json_columns}"
            )
            logger.info(f"Flattening the JSON columns in the table - '{self.table_name}'")
            (
                flattened_data,
                flattening_mapping,
                duplicated_columns
            ) = self._get_artifacts(data, json_columns)
            flatten_metadata[self.table_name] = {
                "flattening_mapping": flattening_mapping,
                "duplicated_columns": duplicated_columns,
                "order_of_columns": order_of_columns
            }
            logger.info(f"The table '{self.table_name}' has been successfully flattened")
            self._save_flatten_metadata(flatten_metadata)
            return flattened_data
        return data

    def run(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Launch the preprocessing process:
        """
        self._run_script()
        preprocessed_data, schema = self.prepare_data()
        data = self._handle_json_columns(preprocessed_data)
        return data, schema


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
            data.drop(list(dropped_columns), axis=1, inplace=True)
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
