from typing import Dict, Optional, List, Tuple, Set
from dataclasses import dataclass, field
import pickle
from uuid import UUID
from datetime import datetime
import base32_crockford

import numpy as np
import dill
import pandas as pd
from scipy.stats import gaussian_kde
import tqdm
from loguru import logger

from syngen.ml.vae.models.features import (
    CategoricalFeature,
    CharBasedTextFeature,
    ContinuousFeature,
    DateFeature,
    BinaryFeature,
)
from syngen.ml.utils import (
    get_nan_labels,
    nan_labels_to_float,
    get_tmp_df,
    get_date_columns,
)
from syngen.ml.data_loaders import DataLoader
from syngen.ml.utils import slugify_parameters
from syngen.ml.utils import fetch_training_config


@dataclass
class Dataset:
    df: pd.DataFrame
    schema: Optional[Dict]
    metadata: Dict
    table_name: str
    paths: Dict
    features: Dict = field(init=False)
    columns: Dict = field(init=False)
    is_fitted: bool = field(init=False)
    all_columns: List = field(init=False)
    null_num_column_names: List = field(init=False)
    zero_num_column_names: List = field(init=False)
    nan_labels_dict: Dict = field(init=False)
    uuid_columns: Set = field(init=False)
    uuid_columns_types: Dict = field(init=False)
    dropped_columns: Set = field(init=False)
    order_of_columns: List = field(init=False)
    non_existent_columns: Set = field(init=False)

    def __post_init__(self):
        self._predefine_fields()
        self._set_metadata()

    def __getstate__(self) -> Dict:
        """
        Return a dictionary of the dataset's state
        """
        dataset_instance = self.__dict__.copy()
        attribute_keys_to_remove = ["df", "metadata", "non_existent_columns"]

        for attr_key in attribute_keys_to_remove:
            if attr_key in dataset_instance:
                del dataset_instance[attr_key]

        return dataset_instance

    def _predefine_fields(self):
        self.features = dict()
        self.columns = dict()
        self.is_fitted = False
        self.all_columns = list()
        self.null_num_column_names = list()
        self.zero_num_column_names = list()
        self.nan_labels_dict = dict()
        self.uuid_columns = set()
        self.uuid_columns_types = {}
        self.dropped_columns = fetch_training_config(
            self.paths["train_config_pickle_path"]
        ).dropped_columns
        self.non_existent_columns = set()
        self.order_of_columns = fetch_training_config(
            self.paths["train_config_pickle_path"]
        ).columns

    def __set_pk_key(self, config_of_keys: Dict):
        """
        Set up primary key for the table
        """
        self.primary_keys_mapping = {
            key: value
            for (key, value) in config_of_keys.items()
            if config_of_keys.get(key).get("type") == "PK"
        }
        self.primary_keys_list = list(self.primary_keys_mapping.keys())
        self.primary_key_name = self.primary_keys_list[0] if self.primary_keys_list else None
        pk_columns_lists = [val["columns"] for val in self.primary_keys_mapping.values()]
        self.pk_columns = [col for uq_cols in pk_columns_lists for col in uq_cols]

        if self.primary_key_name:
            logger.info(f"The primary key name was set: {self.primary_key_name}")
        if self.primary_key_name is None:
            logger.info("No primary key was set.")

    def __set_uq_keys(self, config_of_keys: Dict):
        """
        Set up unique keys for the table
        """
        self.unique_keys_mapping = {
            key: value
            for (key, value) in config_of_keys.items()
            if config_of_keys.get(key).get("type") == "UQ"
        }
        self.unique_keys_mapping_list = list(self.unique_keys_mapping.keys())
        self.unique_keys_list = (
            self.unique_keys_mapping_list if self.unique_keys_mapping_list else []
        )
        uq_columns_lists = [val["columns"] for val in self.unique_keys_mapping.values()]
        self.uq_columns = [col for uq_cols in uq_columns_lists for col in uq_cols]

        if self.unique_keys_list:
            logger.info(f"The unique keys were set: {self.unique_keys_list}")
        if not self.unique_keys_list:
            logger.info("No unique keys were set.")

    def _filter_dropped_keys(self, config_of_keys: Dict, type_of_key: str) -> Tuple[Dict, Set]:
        """
        Filter out keys that contain empty columns
        """
        filtered_keys = {}
        dropped_keys = set()

        for key, value in config_of_keys.items():
            if value.get("type") == type_of_key:
                if any(
                    column for column in value.get("columns") if column in self.dropped_columns
                ):
                    dropped_keys.add(key)
                else:
                    filtered_keys[key] = value

        return filtered_keys, dropped_keys

    def __set_fk_keys(self, config_of_keys: Dict):
        """
        Set up foreign keys for the table
        """
        self.foreign_keys_mapping, dropped_fk_keys = self._filter_dropped_keys(
            config_of_keys, "FK"
        )
        self.foreign_keys_list = list(self.foreign_keys_mapping.keys())
        fk_columns_lists = [val["columns"] for val in self.foreign_keys_mapping.values()]
        self.fk_columns = [col for fk_cols in fk_columns_lists for col in fk_cols]

        if dropped_fk_keys:
            logger.info(
                f"The following foreign keys were dropped: {', '.join(dropped_fk_keys)} "
                f"as they contain empty columns: "
                f"{', '.join(self.dropped_columns.union(self.fk_columns))}"
            )

        if self.foreign_keys_list:
            logger.info(
                f"The following foreign keys were set: {', '.join(self.foreign_keys_list)}"
            )
        if not self.foreign_keys_list:
            logger.info("No foreign keys were set.")

    def __set_types(self, pk_uq_keys_mapping):
        """
        Set up list of data types of primary and unique keys
        """
        self.pk_uq_keys_types = {}
        for key_name, config in pk_uq_keys_mapping.items():
            key_columns = config.get("columns")
            for column in key_columns:
                column_type = (
                    str
                    if column
                    in (
                        self.str_columns
                        | self.categ_columns
                        | self.date_columns
                        | self.long_text_columns
                        | self.uuid_columns
                    )
                    else float
                )
                self.pk_uq_keys_types[column] = column_type

    def __map_text_pk(self):
        for pk, pk_type in self.pk_uq_keys_types.items():
            if pk_type is str:
                mapper = {k: n for n, k in enumerate(self.df[pk])}
                with open(f"{self.paths['fk_kde_path']}{pk}_mapper.pkl", "wb") as file:
                    pickle.dump(mapper, file)

    def __set_metadata(self, metadata: dict, table_name: str):
        config_of_keys = metadata.get(table_name, {}).get("keys")

        if config_of_keys is not None:
            self.__set_pk_key(config_of_keys)
            self.__set_uq_keys(config_of_keys)
            self.__set_fk_keys(config_of_keys)
        else:
            self.primary_keys_mapping = {}
            self.primary_keys_list = []
            self.primary_key_name = None
            self.pk_columns = []
            self.unique_keys_mapping = {}
            self.unique_keys_mapping_list = []
            self.unique_keys_list = []
            self.uq_columns = []
            self.foreign_keys_mapping = {}
            self.foreign_keys_list = []
            self.fk_columns = []

    def _set_metadata(self):
        table_config = self.metadata.get(self.table_name, {})
        self._set_non_existent_columns(table_config)
        self._update_table_config(table_config)
        self.__set_metadata(self.metadata, self.table_name)
        self.__data_pipeline(self.df, self.schema)

    @staticmethod
    def _update_schema(schema: Dict[str, Dict[str, str]], df: pd.DataFrame):
        """
        Synchronize the schema of the table with dataframe
        """
        schema["fields"] = {
            column: data_type
            for column, data_type in schema.get("fields").items()
            if column in df.columns
        }
        return schema

    def _check_if_column_in_removed(self, schema: Dict):
        """
        Exclude the column from the list of categorical columns
        if it was removed previously as empty column
        """
        removed = [
            col for col, data_type in schema.get("fields", {}).items() if data_type == "removed"
        ]
        for col in list(self.categ_columns):
            if col in removed:
                self.categ_columns.remove(col)
                logger.warning(
                    f"The column '{col}' was excluded from the list of categorical columns "
                    f"as far as this column is empty and was removed from the table - "
                    f"'{self.table_name}'"
                )
            continue

    def _check_if_column_existed(self):
        """
        Exclude the column from the list of categorical columns
        if it doesn't exist in the table
        """
        removed_columns = []
        for col in self.categ_columns:
            if col not in self.df.columns:
                removed_columns.append(col)
            continue

        self.categ_columns = set([i for i in self.categ_columns if i not in removed_columns])

        if removed_columns:
            logger.warning(
                f"The columns - {', '.join(removed_columns)} were mentioned as categorical "
                f"in the metadata of the table - '{self.table_name}'. "
                f"It seems that the columns are absent in the table - '{self.table_name}'. "
                f"Please, check the metadata file"
            )

    def _check_if_not_key(self, column: str, column_list: List, key_type: str):
        """
        Exclude the column from the list of categorical columns
        if it relates to certain type of key
        """
        if column in column_list:
            logger.warning(
                f"The column '{column}' was excluded from the list of categorical columns "
                f"as far as this column was set as the {key_type} of the table - "
                f"'{self.table_name}'"
            )
            self.categ_columns.discard(column)

    def _check_if_not_key_column(self):
        """
        Exclude the column from the list of categorical columns
        if it relates to primary key, unique key or foreign key
        """
        for col in list(self.categ_columns):
            self._check_if_not_key(column=col, column_list=self.pk_columns, key_type="primary key")
            self._check_if_not_key(column=col, column_list=self.uq_columns, key_type="unique key")
            self._check_if_not_key(column=col, column_list=self.fk_columns, key_type="foreign key")

    def _check_if_column_binary(self):
        """
        Remove the column from the list of binary columns
        """

        self.binary_columns = set(
            [col for col in self.binary_columns if col not in self.categ_columns]
        )

    def _fetch_categorical_columns(self):
        """
        Fetch the categorical columns from the metadata
        """
        metadata_of_table = self.metadata.get(self.table_name)

        self.categ_columns = set()

        if metadata_of_table is not None:
            self.categ_columns = set(
                metadata_of_table.get("train_settings", {})
                .get("column_types", {})
                .get("categorical", [])
            )
        if self.categ_columns:
            logger.info(
                f"The columns - {', '.join(self.categ_columns)} were defined as categorical "
                f"due to the information from the metadata of the table - '{self.table_name}'"
            )

    def _check_if_column_categorical(self, schema: Dict):
        if self.categ_columns:
            self._check_if_column_in_removed(schema=schema)
            self._check_if_column_existed()
            self._check_if_not_key_column()
            self._check_if_column_binary()

    def _set_binary_columns(self, df: pd.DataFrame):
        """
        Set up the list of binary columns based on the count of unique values in the column
        """
        self.binary_columns = set(
            [col for col in df.columns if df[col].fillna("?").nunique() == 2]
        )

    def _define_categorical_columns(self, df):
        """
        Define the list of categorical columns based on the count of unique values in the column
        """
        defined_columns = set(
            [
                col
                for col in df.columns
                if df[col].dropna().nunique() <= 50 and col not in self.binary_columns
            ]
        )
        self.categ_columns.update(defined_columns)

    def _select_str_columns(self, df):
        if self.schema.get("format", "") == "CSV":
            data_subset = df.select_dtypes(include=[pd.StringDtype(), "object"])
        else:
            text_columns = [
                col
                for col, data_type in self.schema.get("fields", {}).items()
                if data_type == "string"
            ]
            data_subset = df[text_columns]
        return data_subset

    def _set_categorical_columns(self, df: pd.DataFrame, schema: Dict):
        """
        Set up the list of categorical columns
        """
        self._fetch_categorical_columns()
        self._define_categorical_columns(df)
        self._check_if_column_categorical(schema=schema)

    def _set_long_text_columns(self, df: pd.DataFrame):
        """
        Set up the list of columns with long texts (> 200 symbols)
        """
        data_subset = self._select_str_columns(df)

        self.long_text_columns = set()
        if not data_subset.empty:
            data_subset = data_subset.loc[
                :, data_subset.apply(lambda x: (x.str.len() > 200).any())
            ]
            self.long_text_columns = set(data_subset.columns)
            self.long_text_columns -= self.categ_columns
            if self.long_text_columns:
                logger.info(
                    f"Please note that the columns - {self.long_text_columns} contain "
                    f"long texts (> 200 symbols). Such texts' handling consumes "
                    f"significant resources and results in poor quality content, "
                    f"therefore this column(-s) will be generated using "
                    f"a simplified statistical approach"
                )

    @staticmethod
    def _is_valid_ulid(uuid):
        """
        Check if uuid_to_test is a valid ULID (https://github.com/ulid/spec)
        """
        ulid_timestamp = uuid[:10]
        try:
            assert len(uuid) == 26
            ulid_timestamp_int = base32_crockford.decode(ulid_timestamp)
            datetime.fromtimestamp(ulid_timestamp_int / 1000.0)
            return "ulid"
        except Exception:
            return

    def _is_valid_uuid(self, x):
        """
        Check if uuid_to_test is a valid UUID
        """
        result = []
        for i in x.dropna():
            for v in [1, 2, 3, 4, 5]:
                try:
                    uuid_obj = UUID(i, version=v)
                    if str(uuid_obj) == i or str(uuid_obj).replace("-", "") == i:
                        result.append(v)
                except ValueError:
                    result.append(self._is_valid_ulid(i))
        if result:
            return max(set(result), key=result.count)
        else:
            return 0

    def _set_uuid_columns(self, df: pd.DataFrame):
        """
        Set up the list of columns with UUIDs
        """

        data_subset = self._select_str_columns(df)

        if not data_subset.empty:
            data_subset = data_subset.apply(self._is_valid_uuid)
            self.uuid_columns_types = dict(data_subset[data_subset.isin([1, 2, 3, 4, 5, "ulid"])])
            self.uuid_columns = set(self.uuid_columns_types.keys())

    def _set_date_columns(self, df: pd.DataFrame):
        """
        Set up the list of date columns
        """
        self.date_columns = (
            get_date_columns(df, list(self.str_columns))
            - self.categ_columns
            - self.binary_columns
            - self.long_text_columns
        )

    def _remove_non_existent_columns(self, columns: list, key: str, key_type: str) -> list:
        """
        Remove the columns from the table metadata which are absent in the table
        """
        updated_columns = []
        for column in columns:
            if column in self.non_existent_columns:
                logger.warning(
                    f"The column - '{column}' was excluded from the {key_type} - "
                    f"'{key}' as far as this column doesn't exist in the table - "
                    f"'{self.table_name}'"
                )
            else:
                updated_columns.append(column)
        return updated_columns

    def _update_table_config(self, table_config: Dict):
        """
        Update the table metadata by removing the columns which are absent in the table
        but mentioned in the metadata
        """
        table_metadata = table_config.get("keys", {})
        for key in list(table_metadata.keys()):
            key_type = table_metadata[key].get("type")
            updated_columns = self._remove_non_existent_columns(
                table_metadata[key].get("columns", []), key, key_type
            )
            table_metadata[key]["columns"] = updated_columns
            if not table_metadata[key]["columns"]:
                logger.warning(
                    f"All columns in the key {key} are empty, "
                    f"so it will be removed from the table's metadata"
                )
                table_metadata[key].pop(key, None)
                self.metadata[self.table_name]["keys"].pop(key, None)

    def _set_non_existent_columns(self, table_config: Dict):
        """
        Set up the list of columns which are absent in the table
        """
        non_existent_columns = {
            column
            for key_config in table_config.get("keys", {}).values()
            for column in key_config.get("columns", [])
            if column not in self.df.columns
        }

        self.non_existent_columns = non_existent_columns - self.dropped_columns

    def _general_data_pipeline(
        self, df: pd.DataFrame, schema: Dict, check_object_on_float: bool = True
    ):
        """
        Divide columns in dataframe into groups - binary, categorical, integer, float, string, date
        in case metadata of the table is absent

        :param df: dataframe
        :param schema: metadata of the table
        :param check_object_on_float: if True, check if object columns can be converted to float
        """
        if check_object_on_float:
            columns_nan_labels = get_nan_labels(df)
            df = nan_labels_to_float(df, columns_nan_labels)

        self._set_uuid_columns(df)
        self._set_binary_columns(df)
        self._set_categorical_columns(df, schema)
        self._set_long_text_columns(df)
        tmp_df = get_tmp_df(df)
        self.float_columns = set(tmp_df.select_dtypes(include=["float", "float64"]).columns)
        self.int_columns = set(tmp_df.select_dtypes(include=["int", "int64"]).columns)

        float_to_int_cols = set()
        for col in self.float_columns:
            if all(x.is_integer() for x in tmp_df[col]):
                float_to_int_cols.add(col)

        self.int_columns = (self.int_columns | float_to_int_cols) - (
            self.categ_columns | self.binary_columns
        )
        self.float_columns = (
            self.float_columns - self.categ_columns - self.int_columns - self.binary_columns
        )
        self.str_columns = (
            set(tmp_df.columns)
            - self.float_columns
            - self.categ_columns
            - self.int_columns
            - self.binary_columns
            - self.long_text_columns
            - self.uuid_columns
        )
        self.categ_columns -= self.long_text_columns
        self._set_date_columns(df)
        self.str_columns -= self.date_columns
        self.uuid_columns = self.uuid_columns - self.categ_columns - self.binary_columns
        self.uuid_columns_types = {
            k: v for k, v in self.uuid_columns_types.items() if k in self.uuid_columns
        }

    def _avro_data_pipeline(self, df, schema):
        """
        Divide columns in dataframe into groups - binary, categorical, integer, float, string, date
        in case metadata of the table in Avro format is present
        """
        logger.info(f"The schema of table - {self.table_name} was received")
        self._set_uuid_columns(df)
        self._set_binary_columns(df)
        self._set_categorical_columns(df, schema)
        self._set_long_text_columns(df)
        self.int_columns = set(
            column for column, data_type in schema.items() if data_type == "int"
        )
        self.int_columns = self.int_columns - self.categ_columns - self.binary_columns
        self.float_columns = set(
            column for column, data_type in schema.items() if data_type == "float"
        )
        self.float_columns = self.float_columns - self.categ_columns - self.binary_columns
        self.str_columns = set(
            column for column, data_type in schema.items() if data_type == "string"
        )
        self.categ_columns -= self.long_text_columns
        self.str_columns = (
            self.str_columns
            - self.categ_columns
            - self.binary_columns
            - self.long_text_columns
            - self.uuid_columns
        )
        self._set_date_columns(df)
        self.str_columns -= self.date_columns
        self.uuid_columns = self.uuid_columns - self.categ_columns - self.binary_columns
        self.uuid_columns_types = {
            k: v for k, v in self.uuid_columns_types.items() if k in self.uuid_columns
        }

    def __data_pipeline(self, df: pd.DataFrame, schema: Optional[Dict]):
        if schema.get("format") == "CSV":
            self._general_data_pipeline(df, schema)
        elif schema.get("format") == "Avro":
            schema = self._update_schema(schema, df)
            self._avro_data_pipeline(df, schema.get("fields"))

        assert len(self.str_columns) + len(self.float_columns) + len(self.int_columns) + len(
            self.date_columns
        ) + len(self.categ_columns) + len(self.binary_columns) + len(self.long_text_columns) + len(
            self.uuid_columns
        ) == len(
            df.columns
        ), (
            "According to number of columns with defined types, "
            "column types are not identified correctly"
        )

        logger.debug(
            f"Count of string columns: {len(self.str_columns)}; "
            + f"Count of float columns: {len(self.float_columns)}; "
            + f"Count of int columns: {len(self.int_columns)}; "
            + f"Count of categorical columns: {len(self.categ_columns)}; "
            + f"Count of date columns: {len(self.date_columns)}; "
            + f"Count of binary columns: {len(self.binary_columns)}; "
            + f"Count of long text columns: {len(self.long_text_columns)}; "
            + f"Count of uuid columns: {len(self.uuid_columns)}"
        )
        for column in self.uuid_columns:
            logger.info(f"Column '{column}' defined as UUID column")

    def assign_feature(self, feature, columns):
        name = feature.original_name

        if name in self.features:
            raise Exception(f"{name} is already contained in features")

        if not isinstance(columns, (list, tuple)):
            columns = [columns]

        self.features[name] = feature
        self.columns[name] = columns

    def set_nan_params(self, nan_labels: dict):
        """Save params that are used to keep and replicate nan and empty values

        Args:
            nan_labels (dict): dictionary that matches column name to the label of missing value
                               (e.g. {'Score': 'Not available'})
        """
        self.nan_labels_dict = nan_labels

    def fit(self, data):
        for name, feature in self.features.items():
            feature.fit(data[self.columns[name]])

        self.all_columns = [col for col in self.columns]
        self.is_fitted = True

    def transform(self, data, excluded_features=set()):
        transformed_features = list()
        selected_features = {
            name: feature
            for name, feature in self.features.items()
            if name not in excluded_features
        }
        for name, feature in selected_features.items():
            transformed_features.append(feature.transform(data[self.columns[name]]))
        return transformed_features

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def _check_count_features(self, data):
        return (len(data) == len(self.features)) or (
            len(data) + len(self.fk_columns) == len(self.features)
        )

    def inverse_transform(self, data, excluded_features=set()):
        inverse_transformed_data = list()
        column_names = list()
        if not isinstance(data, list):
            data = [data]
        assert self._check_count_features(data)

        for transformed_data, (name, feature) in tqdm.tqdm(
            iterable=zip(data, self.features.items()),
            desc="Generation of the data...",
            total=len(data),
        ):
            if name not in excluded_features and name not in self.fk_columns:
                column_names.extend(self.columns[name])
                inverse_transformed_data.append(feature.inverse_transform(transformed_data))

        stacked_data = np.column_stack(inverse_transformed_data)
        data = pd.DataFrame(stacked_data, columns=column_names)

        return data

    def _preprocess_str_params(self, feature: str) -> Tuple[int, int]:
        max_len = int(self.df[feature].apply(lambda line: len(line)).max())
        rnn_units = 16
        if 1 <= max_len < 7:
            rnn_units = 32

        if 6 < max_len < 13:
            rnn_units = 128

        if 12 < max_len < 17:
            rnn_units = 256

        if max_len > 16:
            rnn_units = 512
        return max_len, rnn_units

    def _preprocess_nan_cols(
        self, feature: str, fillna_strategy: str = None, zero_cutoff: float = 0.3
    ) -> tuple:
        """Fill NaN values in numeric column with some value according to strategy.
        Fill NaN values in string columns can only work in 'mode' strategy.
        If NaN values exist additional column is created and added to DataFrame.
        This column has value of 1 in case corresponding row contains NaN and 0 otherwise.
        New column name is built like 'column name'+'_null'.

        Args:
            feature (str): Feature name.
            fillna_strategy (str, optional): Can be 'mean', 'mode' or None.
                                             If None NaN values in column are replaced with 0.
                                             Defaults to None.
                                             Note: string columns only work with 'mode'.

        Returns:
            tuple: Tuple that consists of either feature name or both feature name
            and new null feature name.
        """
        isnull_feature = pd.isnull(self.df[feature])
        many_zeros_feature = (self.df[feature] == 0).sum() / (len(self.df[feature])) > zero_cutoff
        if many_zeros_feature:
            feature_zero = feature + "_zero"
            self.df[feature_zero] = self.df[feature].apply(lambda x: 0 if x == 0 else 1)
            if not isnull_feature.any():
                return (feature, feature_zero)
        if isnull_feature.any():
            nan_number = isnull_feature.sum()
            logger.info(
                f"Column '{feature}' contains {nan_number} "
                f"({round(nan_number * 100 / len(isnull_feature))}%) "
                f"empty values out of {len(isnull_feature)}. "
                f"Filling them with {fillna_strategy or 'zero'}."
            )
            if fillna_strategy == "mean":
                fillna_value = self.df[feature].mean()
            elif fillna_strategy == "mode":
                fillna_value = self.df[feature].dropna().mode().sample(1).values[0]
            elif fillna_strategy == "text":
                fillna_value = ""
            else:
                fillna_value = 0

            feature_null = feature + "_null"
            self.df[feature_null] = isnull_feature.astype(int)
            self.df[feature] = self.df[feature].fillna(fillna_value)
            if not many_zeros_feature:
                return (feature, feature_null)
            else:
                return (feature, feature_null, feature_zero)
        else:
            return (feature,)

    def _preprocess_categ_params(self, feature: str):
        self.df[feature] = self.df[feature].fillna("?").astype(str)
        return feature

    @staticmethod
    @slugify_parameters(exclude_params=("fk_kde_path", "fk_column"))
    def _fetch_mapper(fk_kde_path, table_name, pk_table, pk_column, fk_column):
        """
        Fetch the mapper for foreign key in data type - 'string'
        """
        try:
            with open(
                f"{fk_kde_path.replace(table_name, pk_table)}{pk_column}_mapper.pkl",
                "rb",
            ) as file:
                mapper = pickle.load(file)
            return mapper
        except FileNotFoundError:
            logger.warning(
                f"The mapper for the {fk_column} text key is not found. "
                f"Simple sampling will be used."
            )

    @staticmethod
    @slugify_parameters(exclude_params=("kde", "fk_kde_path"))
    def _save_kde_artifacts(kde, fk_kde_path, fk_column):
        """
        Save KDE artifacts
        """
        with open(f"{fk_kde_path}{fk_column}.pkl", "wb") as file:
            dill.dump(kde, file)

        logger.info(f"KDE artifacts saved to {fk_kde_path}{fk_column}.pkl")

    def _preprocess_fk_params(self):
        for fk in self.foreign_keys_list:
            fk_columns = self.foreign_keys_mapping.get(fk).get("columns")
            for fk_column in fk_columns:
                fk_column_values = self.df[fk_column]
                correspondent_pk_table = self.foreign_keys_mapping[fk]["references"]["table"]
                correspondent_pk_col = self.foreign_keys_mapping[fk]["references"]["columns"][0]
                if fk_column_values.dtype in (pd.StringDtype(), "object"):
                    mapper = self._fetch_mapper(
                        fk_kde_path=self.paths["fk_kde_path"],
                        table_name=self.table_name,
                        pk_table=correspondent_pk_table,
                        pk_column=correspondent_pk_col,
                        fk_column=fk_column,
                    )
                    if mapper is None:
                        continue
                    fk_column_values = fk_column_values.map(mapper)
                noise_to_prevent_singularity = np.random.normal(0, 0.0001, len(fk_column_values))
                kde = gaussian_kde(fk_column_values + noise_to_prevent_singularity)
                self._save_kde_artifacts(
                    kde=kde, fk_kde_path=self.paths["fk_kde_path"], fk_column=fk_column
                )

    def _drop_fk_columns(self):
        """
        Drop columns in dataframe which defined as foreign key
        """
        for fk_column in set(self.fk_columns):
            self.df = self.df.drop(fk_column, axis=1)
            logger.debug(
                f"The column - '{fk_column}' dropped from the training process "
                f"as it is defined as FK column and will be sampled from the PK table"
            )

    def __sample_only_joined_rows(self, fk):
        references = self.foreign_keys_mapping.get(fk).get("references")
        pk_table = references.get("table")
        pk_table_data, schema = DataLoader(
            f"model_artifacts/tmp_store/{pk_table}/input_data_{pk_table}.csv"
        ).load_data()
        pk_column_label = references.get("columns")[0]

        drop_index = self.df[~self.df[fk].isin(pk_table_data[pk_column_label].values)].index
        if len(drop_index) > 0:
            logger.info(
                f"{len(drop_index)} rows were deleted, as they did not have matching primary keys."
            )
            logger.info(f"{len(self.df) - len(drop_index)} rows are left in table as input.")
        self.df = self.df.drop(drop_index)

    def _assign_char_feature(self, feature):
        """
        Assign text based feature to text columns
        """
        features = self._preprocess_nan_cols(feature, fillna_strategy="text")
        max_len, rnn_units = self._preprocess_str_params(features[0])
        self.assign_feature(
            CharBasedTextFeature(features[0], text_max_len=max_len, rnn_units=rnn_units),
            features[0],
        )
        logger.info(f"Column '{features[0]}' assigned as text based feature")

        if len(features) > 1:
            for feature in features[1:]:
                if feature.endswith("_null"):
                    self.null_num_column_names.append(feature)
                if feature.endswith("_zero"):
                    self.zero_num_column_names.append(feature)
                self.assign_feature(ContinuousFeature(feature, column_type=float), feature)
                logger.info(f"Column '{feature}' assigned as float based feature")

    def _assign_float_feature(self, feature):
        """
        Assign float based feature to float columns
        """
        # num_bins = self.find_clusters(df, float_columns)
        features = self._preprocess_nan_cols(feature, fillna_strategy="mean")
        if len(features) == 2 and features[1].endswith("_null"):
            self.null_num_column_names.append(features[1])
        if len(features) == 2 and features[1].endswith("_zero"):
            self.zero_num_column_names.append(features[1])
        if len(features) == 3:
            self.null_num_column_names.append(features[1])
            self.zero_num_column_names.append(features[2])
        for feature in features:
            self.assign_feature(ContinuousFeature(feature, column_type=float), feature)
            logger.info(f"Column '{feature}' assigned as float based feature")

    def _assign_int_feature(self, feature):
        """
        Assign int based feature to int columns
        """
        features = self._preprocess_nan_cols(feature, fillna_strategy="mean")
        self.assign_feature(ContinuousFeature(features[0], column_type=int), features[0])
        logger.info(f"Column '{features[0]}' assigned as int based feature")
        if len(features) > 1:
            for feature in features[1:]:
                if feature.endswith("_null"):
                    self.null_num_column_names.append(feature)
                if feature.endswith("_zero"):
                    self.zero_num_column_names.append(feature)
                self.assign_feature(ContinuousFeature(feature, column_type=float), feature)
                logger.info(f"Column '{feature}' assigned as float based feature")

    def _assign_categ_feature(self, feature):
        """
        Assign categorical based feature to categorical columns
        """
        feature = self._preprocess_categ_params(feature)
        self.assign_feature(CategoricalFeature(feature), feature)
        logger.info(f"Column '{feature}' assigned as categorical based feature")

    def _assign_date_feature(self, feature):
        """
        Assign date feature to date columns
        """
        features = self._preprocess_nan_cols(feature, fillna_strategy="mode")
        self.assign_feature(DateFeature(features[0]), features[0])
        logger.info(f"Column '{features[0]}' assigned as date feature")
        if len(features) == 2:
            self.null_num_column_names.append(features[1])
            self.assign_feature(ContinuousFeature(features[1]), features[1])
            logger.info(f"Column '{features[1]}' assigned as float feature")

    def _assign_binary_feature(self, feature):
        """
        Assign binary feature to binary columns
        """
        feature = self._preprocess_categ_params(feature)
        self.assign_feature(BinaryFeature(feature), feature)
        logger.info(f"Column '{feature}' assigned as binary feature")

    def _assign_fk_feature(self):
        """
        Assign corresponding to FK null column and preprocess if required.
        """
        for fk_name, config in self.foreign_keys_mapping.items():
            if "joined_sample" in config and config["joined_sample"]:
                self.__sample_only_joined_rows(fk_name)
            else:
                for fk_column in self.fk_columns:
                    features = self._preprocess_nan_cols(fk_column, fillna_strategy="mode")
                    if len(features) > 1:
                        self.assign_feature(
                            ContinuousFeature(features[1], column_type=int), features[1]
                        )

    def pipeline(self) -> pd.DataFrame:
        columns_nan_labels = get_nan_labels(self.df)
        self.df = nan_labels_to_float(self.df, columns_nan_labels)

        if self.foreign_keys_list:
            self._assign_fk_feature()
            self._preprocess_fk_params()
            self._drop_fk_columns()

        self.primary_keys_mapping.update(self.unique_keys_mapping)
        pk_uq_keys_mapping = self.primary_keys_mapping
        if pk_uq_keys_mapping:
            self.__set_types(pk_uq_keys_mapping)
            self.__map_text_pk()

        for column in self.df.columns:
            if column in self.str_columns:
                self._assign_char_feature(column)
            elif column in self.float_columns:
                self._assign_float_feature(column)
            elif column in self.int_columns:
                self._assign_int_feature(column)
            elif column in self.categ_columns:
                self._assign_categ_feature(column)
            elif column in self.date_columns:
                self._assign_date_feature(column)
            elif column in self.binary_columns:
                self._assign_binary_feature(column)

        self.set_nan_params(columns_nan_labels)

        self.fit(self.df)

        return self.df
