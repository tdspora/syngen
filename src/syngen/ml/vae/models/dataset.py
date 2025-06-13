from typing import Dict, Optional, List, Tuple, Set, Literal
import pickle
from uuid import UUID
from datetime import datetime
import base32_crockford
from collections import Counter
import re

import numpy as np
import dill
import pandas as pd
from pandas.tseries.api import guess_datetime_format
from scipy.stats import gaussian_kde
import tqdm
from loguru import logger

from syngen.ml.vae.models.features import (
    CategoricalFeature,
    CharBasedTextFeature,
    EmailFeature,
    ContinuousFeature,
    DateFeature,
    BinaryFeature,
)
from syngen.ml.utils import (
    get_nan_labels,
    nan_labels_to_float,
    get_date_columns,
    fetch_timezone,
    TIMEZONE_REGEX
)
from syngen.ml.utils import slugify_parameters
from syngen.ml.utils import fetch_config, clean_up_metadata
from syngen.ml.mlflow_tracker import MlflowTracker


class BaseDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        schema: Optional[Dict],
        metadata: Dict,
        table_name: str,
        main_process: str,
        paths: Dict
    ):
        self.df = df
        self.fields = schema.get("fields", {})
        self.schema_format = schema.get("format")
        self.metadata = metadata
        self.table_name = table_name
        self.paths = paths
        self.main_process = main_process
        self.features: Dict = dict()
        self.columns: Dict = dict()
        self.is_fitted: bool = False
        self.all_columns: List = list()
        self.null_num_column_names: List = list()
        self.zero_num_column_names: List = list()
        self.nan_labels_dict: Dict = dict()
        self.uuid_columns: Set = set()
        self.uuid_columns_types: Dict = dict()
        self.dropped_columns: Set = set()
        self.tech_columns: Set = set()
        self.order_of_columns: List = list()
        self.custom_categorical_columns: Set = set()
        self.categorical_columns: Set = set()
        self.str_columns: Set = set()
        self.float_columns: Set = set()
        self.int_columns: Set = set()
        self.date_columns: Set = set()
        self.date_mapping: Dict = dict()
        self.binary_columns: Set = set()
        self.email_columns: Set = set()
        self.long_text_columns: Set = set()
        self.primary_keys_mapping: Dict = dict()
        self.primary_keys_list: List = list()
        self.primary_key_name: Optional[str] = None
        self.pk_columns: List = list()
        self.unique_keys_mapping: Dict = dict()
        self.unique_keys_mapping_list: List = list()
        self.unique_keys_list: List = list()
        self.uq_columns_lists: List = list()
        self.uq_columns: List = list()
        self.foreign_keys_mapping: Dict = dict()
        self.foreign_keys_list: List = list()
        self.fk_columns: List = list()
        self.keys_mapping: Dict = dict()
        train_config = fetch_config(
            self.paths["train_config_pickle_path"]
        )
        self.dropped_columns: Set = train_config.dropped_columns
        self.order_of_columns: List = train_config.columns
        self.format = self.metadata[self.table_name].get("format", {})
        self.nan_labels_dict = dict()
        self.nan_labels_in_uuid = dict()
        self.cast_to_integer = set()
        self.cast_to_float = set()


class Dataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        schema: Optional[Dict],
        metadata: Dict,
        table_name: str,
        main_process: str,
        paths: Dict
    ):
        super().__init__(
            df,
            schema,
            metadata,
            table_name,
            main_process,
            paths
        )

    def _detect_categorical_columns(self):
        """
        Define binary and categorical columns
        """
        self._set_binary_columns()
        self._set_categorical_columns()
        self.binary_columns -= self.categorical_columns

    def _preprocess_df(self, excluded_columns: Set[str]):
        """
        Preprocess the dataframe
        """
        self._cast_to_numeric(excluded_columns)
        self.nan_labels_dict = get_nan_labels(self.df, excluded_columns)
        self.df = nan_labels_to_float(self.df, self.nan_labels_dict)

    def _preparation_step(self):
        """
        Define binary and categorical columns,
        preprocess the dataframe before the detection of data types of columns
        """
        table_config = self.metadata.get(self.table_name, {})
        self._set_non_existent_columns(table_config)
        self._update_metadata(table_config)
        self._set_metadata()
        self._detect_categorical_columns()
        excluded_columns = set().union(
            self.categorical_columns,
            self.binary_columns
        )
        self._preprocess_df(excluded_columns)
        self._update_schema()

    def _cast_to_numeric(self, excluded_columns: Set[str]):
        """
        Cast the values in the column to 'integer' or 'float' data type
        in case all of them might be cast to this data type
        """
        text_columns = self._select_str_columns()
        list_of_columns = set(text_columns) - excluded_columns
        for column in list_of_columns:
            try:
                if self.df[column].dropna().apply(lambda x: float(x).is_integer()).all():
                    self.df[column] = pd.to_numeric(self.df[column], downcast="integer")
                    self.cast_to_integer.add(column)
                elif self.df[column].dropna().apply(lambda x: not float(x).is_integer()).any():
                    self.df[column] = pd.to_numeric(self.df[column], downcast="float")
                    self.cast_to_float.add(column)
            except ValueError:
                continue
        if self.cast_to_integer:
            columns = [f"'{item}'" for item in self.cast_to_integer]
            logger.info(
                f"The columns - {', '.join(columns)} "
                "have been cast to the 'integer' data type"
            )
        if self.cast_to_float:
            columns = [f"'{item}'" for item in self.cast_to_float]
            logger.info(
                f"The columns - {', '.join(columns)} "
                "have been cast to the 'float' data type"
            )

    def __getstate__(self) -> Dict:
        """
        Return a dictionary of the dataset's state
        """
        clean_up_metadata(self.metadata)
        dataset_instance = self.__dict__.copy()
        attribute_keys_to_remove = ["df", "non_existent_columns"]

        for attr_key in attribute_keys_to_remove:
            if attr_key in dataset_instance:
                del dataset_instance[attr_key]
        return dataset_instance

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

        if self.primary_key_name is None:
            logger.info("No primary key was set.")
        if self.primary_key_name:
            logger.info(f"The primary key name was set: {self.primary_key_name}")
            self._validate_pk_key()

    def _validate_pk_key(self):
        """
        Check null values and uniqueness in primary key
        """
        errors = []
        # Check NA values in primary key columns
        if self.df[self.pk_columns].isna().any(axis=None):
            pk_columns_with_na = [
                column for column in self.pk_columns if self.df[column].isna().any()
            ]
            error_msg = (
                f"The primary key '{self.primary_key_name}' "
                f"contains null values in columns: {pk_columns_with_na}. "
            )
            errors.append(error_msg)

        # Check uniqueness of primary key
        if self.df[self.pk_columns].duplicated().any():
            error_msg = (
                f"The primary key '{self.primary_key_name}' "
                f"contains duplicates. "
            )
            errors.append(error_msg)

        if errors:
            raise ValueError(
                " ".join(errors) + "Please check the original data."
            )
        else:
            logger.info("Values in primary key are unique.")

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
        self.uq_columns_lists = [
            val["columns"] for val in self.unique_keys_mapping.values()
        ]
        self.uq_columns = [
            col for uq_cols in self.uq_columns_lists for col in uq_cols
        ]

        if not self.unique_keys_list:
            logger.info("No unique keys were set.")
        if self.unique_keys_list:
            logger.info(f"The unique keys were set: {self.unique_keys_list}")
            self._validate_uq_keys()

    def _validate_uq_keys(self):
        """
        Check null values and uniqueness in unique keys
        """
        uq_keys_mapping = dict(
            zip(self.unique_keys_mapping_list, self.uq_columns_lists)
        )
        errors = []
        for key_name, key_columns in uq_keys_mapping.items():
            # explicitly check for > 1 null values in unique key columns
            all_na_mask = self.df[key_columns].isna().all(axis=1)
            if all_na_mask.sum() > 1:
                error_msg = (
                    f"The unique key '{key_name}' contains > 1 null values. "
                )
                errors.append(error_msg)

            not_all_na_mask = ~all_na_mask
            if self.df[not_all_na_mask][key_columns].duplicated().any():
                error_msg = (
                    f"Values in the unique key '{key_name}' are not unique. "
                )
                errors.append(error_msg)

        if errors:
            raise ValueError(
                " ".join(errors) + "Please check the original data."
            )
        else:
            logger.info("Values in unique keys are unique.")

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
                        | self.email_columns
                        | self.categorical_columns
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

    def _set_metadata(self):
        config_of_keys = self.metadata.get(self.table_name, {}).get("keys")

        if config_of_keys is not None:
            self.__set_pk_key(config_of_keys)
            self.__set_uq_keys(config_of_keys)
            self.__set_fk_keys(config_of_keys)

        self.keys_mapping = {
            "primary key": self.pk_columns,
            "unique key": self.uq_columns,
            "foreign key": self.fk_columns
        }

    def launch_detection(self):
        self._preparation_step()
        self._launch_detection()

    def _launch_detection(self):
        self._common_detection()
        self._detection_pipeline()

    def _common_detection(self):
        """
        Identify and classify data types within the dataset, including
        binary columns, categorical columns, UUID columns, long text columns,
        and email columns.

        This process is agnostic to the file format of the dataset.
        """
        self._set_uuid_columns()
        self._set_long_text_columns()
        self._set_email_columns()
        if self.nan_labels_dict and self.format.get("na_values", []):
            logger.info(
                f"Despite the fact that data loading utilized the 'format' section "
                f"for handling NA values, some values have been detected by the algorithm "
                f"as NA labels in the columns - {self.nan_labels_dict}"
            )

    def _update_schema(self):
        """
        Synchronize the schema of the table with dataframe
        """
        self.fields = {
            column: data_type
            for column, data_type in self.fields.items()
            if column in self.df.columns
        }
        int_fields = {
            column: "int"
            for column, dtype in self.fields.items()
            if column in self.cast_to_integer
        }
        self.fields.update(int_fields)
        float_fields = {
            column: "float"
            for column, dtype in self.fields.items()
            if column in self.cast_to_float
        }
        self.fields.update(float_fields)
        for column in self.nan_labels_dict.keys():
            self.fields[column] = (
                "int" if all(x.is_integer() for x in self.df[column].dropna()) else "float"
            )

    def _check_if_column_not_removed(self, column_type: str):
        """
        Exclude the column from the list of columns
        if it was removed previously as empty column
        """
        removed = [
            col
            for col, data_type in self.fields.items()
            if data_type == "removed"
        ]
        columns = getattr(self, f"{column_type}_columns")
        for col in list(columns):
            if col in removed:
                columns.remove(col)
                logger.warning(
                    f"The column '{col}' was excluded from the list of {column_type} columns "
                    f"as far as this column is empty and was removed from the table - "
                    f"'{self.table_name}'"
                )
            continue
        setattr(self, f"{column_type}_columns", columns)

    def _check_if_column_existed(self, column_type: str):
        """
        Exclude the column from the list of columns
        if it doesn't exist in the table
        """
        removed_columns = []
        columns = getattr(self, f"{column_type}_columns")
        for col in columns:
            if col not in self.df.columns:
                removed_columns.append(col)
            continue

        setattr(
            self,
            f"{column_type}_columns",
            set([i for i in columns if i not in removed_columns])
        )

        if removed_columns:
            logger.warning(
                f"The columns - {', '.join(removed_columns)} were mentioned as {column_type} "
                f"in the metadata of the table - '{self.table_name}'. "
                f"It seems that the columns are absent in the table - '{self.table_name}'. "
                f"Please, check the metadata file"
            )

    def _check_if_column_not_key(self, column_type: str, column_list: List, key_type: str):
        """
        Exclude the column from the list of columns
        if it relates to certain type of key
        """
        columns = getattr(self, f"{column_type}_columns")
        columns_to_remove = {column for column in columns if column in column_list}
        for column in columns_to_remove:
            logger.warning(
                f"The column '{column}' was excluded from the list of {column_type} columns "
                f"as far as this column was set as the {key_type} of the table - "
                f"'{self.table_name}'"
            )
            columns.discard(column)
        setattr(
            self,
            f"{column_type}_columns",
            columns
        )

    def _fetch_categorical_columns(self):
        """
        Fetch the categorical columns from the metadata
        """
        metadata_of_table = self.metadata.get(self.table_name)

        if metadata_of_table is not None:
            self.custom_categorical_columns = set(
                metadata_of_table.get("train_settings", {})
                .get("column_types", {})
                .get("categorical", [])
            )
        if self.custom_categorical_columns:
            logger.info(
                f"The columns - {', '.join(self.custom_categorical_columns)} were defined "
                f"as categorical due to the information from the metadata of the table - "
                f"'{self.table_name}'"
            )

    def _validate_columns(self, column_type: str):
        """
        Validate categorical columns
        """
        self._check_if_column_not_removed(column_type)
        self._check_if_column_existed(column_type)
        for key_type, key_list in self.keys_mapping.items():
            self._check_if_column_not_key(
                column_type=column_type,
                column_list=key_list,
                key_type=key_type
            )

    def _set_binary_columns(self):
        """
        Set up the list of binary columns based on the count of unique values in the column
        """
        self.binary_columns = set(
            [col for col in self.df.columns if self.df[col].fillna("?").nunique() == 2]
        )

    def _define_categorical_columns(self):
        """
        Define the list of categorical columns based on the count of unique values in the column
        """
        self.categorical_columns = set(
            [
                col
                for col in self.df.columns
                if self.df[col].fillna("?").nunique() <= 50
                and col not in self.binary_columns
            ]
        )
        self.categorical_columns.update(self.custom_categorical_columns)

    # TODO: cache this function calls (?)
    def _select_str_columns(self) -> List[str]:
        """
        Select the text columns
        """
        if self.schema_format == "CSV":
            text_columns = [
                col for col, dtype in dict(self.df.dtypes).items()
                if dtype in ["object", "string"]
            ]
        else:
            text_columns = [
                col
                for col, data_type in self.fields.items()
                if data_type == "string"
            ]
        return text_columns

    def _set_categorical_columns(self):
        """
        Set up the list of categorical columns
        """
        self._fetch_categorical_columns()
        self._define_categorical_columns()
        if self.categorical_columns:
            self._validate_columns(column_type="categorical")

    def _set_long_text_columns(self):
        """
        Set up the list of columns with long texts (> 200 symbols)
        """
        text_columns = self._select_str_columns()
        data_subset = self.df[text_columns]

        if not data_subset.empty:
            data_subset = data_subset.loc[
                :, data_subset.apply(lambda x: (x.astype(str).str.len() > 200).any())
            ]
            self.long_text_columns = set(data_subset.columns)
            self.long_text_columns = (self.long_text_columns - self.categorical_columns
                                      - self.binary_columns)
            if self.long_text_columns:
                list_of_long_text_columns = [f"'{column}'" for column in self.long_text_columns]
                logger.info(
                    f"Please note that the columns - {', '.join(list_of_long_text_columns)} "
                    f"contain long texts (> 200 symbols). Such texts' handling consumes "
                    f"significant resources and results in poor quality content, "
                    f"therefore this column(-s) will be generated using "
                    f"a simplified statistical approach"
                )

    def _set_email_columns(self):
        """
        Set up the list of columns with emails (defined by count of @ symbols)
        """
        text_columns = self._select_str_columns()
        data_subset = self.df[text_columns]

        if not data_subset.empty:
            # @ presents in more than 4/5 of not None values of every column
            # Email pattern
            email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'

            # Vectorized operation to check for email pattern
            email_mask = data_subset.apply(
                lambda col: col.apply(
                    lambda x: isinstance(x, str) and bool(re.search(email_pattern, x))
                )
            )

            # Count the number of emails in each column
            count_emails = email_mask.sum(axis=0)

            # Adjusted count to inverse 4/5 rule
            adjusted_count = count_emails * 1.25

            # Count non-NA values in each column
            non_na_values_count = data_subset.notna().sum(axis=0)

            # Filter columns where adjusted count is greater than non-NA count
            filter_mask = adjusted_count > non_na_values_count

            # Select columns that match the filter
            email_columns = data_subset.columns[filter_mask]

            # Update the email_columns attribute
            self.email_columns = (
                set(email_columns) - self.categorical_columns - self.binary_columns
            )

    @staticmethod
    def _is_valid_ulid(uuid):
        """
        Check if uuid_to_test is a valid ULID (https://github.com/ulid/spec)
        """
        # ULID pattern check using regex
        try:
            if not re.match(r'^[0123456789ABCDEFGHJKMNPQRSTVWXYZ]{26}$', uuid):
                return
            ulid_timestamp = uuid[:10]
            ulid_timestamp_int = base32_crockford.decode(ulid_timestamp)
            datetime.fromtimestamp(ulid_timestamp_int / 1000.0)
            return "ulid"
        except Exception:
            return

    def _get_uuid_version(self, value):
        """
        Get the version of UUID value if the value is a UUID
        """
        for v in range(1, 6):
            try:
                uuid_obj = UUID(value, version=v)
                if str(uuid_obj) == value or str(uuid_obj).replace("-", "") == value:
                    return v
            except (ValueError, AttributeError, TypeError):
                continue
        return None

    def _process_values(self, column: pd.DataFrame):
        """
        Process a text column and collect uuid types in the list - 'result',
        and non uuid values into the set - 'non_uuid_values'
        """
        result = []
        non_uuid_values = set()
        for i in column.dropna().unique():
            uuid_version = self._get_uuid_version(i)
            if uuid_version:
                result.append(uuid_version)
            else:
                if ulid := self._is_valid_ulid(i):
                    result.append(ulid)
                else:
                    non_uuid_values.add(i)
        return result, non_uuid_values

    def _is_valid_uuid(self, column: pd.DataFrame):
        """
        Check if a column is a valid UUID column.
        If there are no NaNs and a single non UUID/ULID value,
        it is treated as a nan label and set as NaN during a training process.
        """
        result, non_uuid_values = self._process_values(column)

        if result:
            return self._handle_non_uuid_values(
                column,
                result,
                non_uuid_values,
            )
        else:
            return

    def _log_warning(self, column, non_uuid_values, contain_nan):
        """
        Build the log warning message related to columns containing UUID/ULID values
        """
        warning_msg = f"Column '{column.name}' contains UUID/ULID values"
        if non_uuid_values:
            non_uuid_values_quoted = [f"'{value}'" for value in non_uuid_values]
            if len(non_uuid_values_quoted) > 10:
                warning_msg += (
                    f", and more than {len(non_uuid_values_quoted)} non-UUID/ULID values"
                )
            else:
                warning_msg += f", and non-UUID/ULID values - {', '.join(non_uuid_values_quoted)}"
        if contain_nan:
            warning_msg += ", and null value/s"
        warning_msg += ". The column will be treated as a text column."
        logger.warning(warning_msg)

    def _handle_non_uuid_values(self, column, result, non_uuid_values):
        """
        Handle a UUID column containing non UUID/ULID values
        """
        contain_nan = column.isnull().sum() > 0
        most_common_uuid_type = max(set(result), key=result.count)
        if not non_uuid_values:
            return most_common_uuid_type

        if not contain_nan and len(non_uuid_values) == 1:
            self.__handle_nan_label_in_uuid(column, non_uuid_values)
            return most_common_uuid_type

        if len(non_uuid_values) > 1 or (contain_nan and non_uuid_values):
            self._log_warning(column, non_uuid_values, contain_nan)
            return

    def __handle_nan_label_in_uuid(self, column, non_uuid_values):
        """
        Replaces the unique non-UUID/ULID value with NaNs
        Updates the nan_labels_in_uuid dictionary and
        adds it to 'nan_labels_dict' dict
        """
        unique_non_uuid = next(iter(non_uuid_values))

        logger.info(
            f"Column '{column.name}' contains a unique non-UUID/ULID "
            f"value '{unique_non_uuid}'. It will be treated "
            f"as a null label and replaced with nulls during the training process."
        )
        self.nan_labels_in_uuid[column.name] = unique_non_uuid
        self.df[column.name].replace(unique_non_uuid, np.nan, inplace=True)
        # update the nan_labels_dict with nan_labels_in_uuid
        self.nan_labels_dict.update(self.nan_labels_in_uuid)

    def _set_uuid_columns(self):
        """
        Set up the list of columns with UUIDs
        """

        text_columns = self._select_str_columns()
        data_subset = self.df[text_columns]

        if not data_subset.empty:
            data_subset = data_subset.apply(self._is_valid_uuid)

            self.uuid_columns_types = dict(data_subset[data_subset.isin([1, 2, 3, 4, 5, "ulid"])])
            self.uuid_columns = set(self.uuid_columns_types.keys())

    def _set_date_columns(self):
        """
        Set up the list of date columns
        """
        self.date_columns = (
            get_date_columns(self.df, list(self.str_columns))
            - self.email_columns
            - self.categorical_columns
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

    def _update_metadata(self, table_config: Dict):
        """
        Update the metadata of the table by removing the columns which are absent in the table
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

    def __define_date_format(self, column: str) -> str:
        """
        Define the most common date format in the column
        Supported formats:
        - %d-%m-%Y, %d/%m/%Y, %d.%m.%Y, %m-%d-%Y, %m/%d/%Y, %m.%d.%Y, %Y-%m-%d, %Y/%m/%d, %Y.%m.%d;
        - %d %B %Y, %d %b %Y, %Y %B %d, %Y %b %d; %B %d, %Y; %b %d, %Y;
        - %Y-%m-%dT%H:%M:%S, %Y-%m-%d %H:%M:%S, %Y-%m-%d-%H:%M:%S;
        - %Y/%m/%dT%H:%M:%S, %Y/%m/%d %H:%M:%S, %Y/%m/%d-%H:%M:%S;
        - %Y.%m.%dT%H:%M:%S, %Y.%m.%d %H:%M:%S, %Y.%m.%d-%H:%M:%S;
        - %Y%m%dT%H:%M:%S, %Y%m%d %H:%M:%S, %Y%m%d-%H:%M:%S;
        - %d-%m-%YT%H:%M:%S, %d-%m-%Y %H:%M:%S, %d-%m-%Y-%H:%M:%S;
        - %d/%m/%YT%H:%M:%S, %d/%m/%Y %H:%M:%S, %d/%m/%Y-%H:%M:%S;
        - %d.%m.%YT%H:%M:%S, %d.%m.%Y %H:%M:%S, %d.%m.%Y-%H:%M:%S;
        - %m-%d-%YT%H:%M:%S, %m-%d-%Y %H:%M:%S, %m-%d-%Y-%H:%M:%S;
        - %m/%d/%YT%H:%M:%S, %m/%d/%Y %H:%M:%S, %m/%d/%Y-%H:%M:%S;
        - %m.%d.%YT%H:%M:%S, %m.%d.%Y %H:%M:%S, %m.%d.%Y-%H:%M:%S;
        - %Y-%m-%dT%H:%M:%S.%f, %Y-%m-%d %H:%M:%S.%f, %Y-%m-%d-%H:%M:%S.%f;
        - %Y/%m/%dT%H:%M:%S.%f, %Y/%m/%d %H:%M:%S.%f, %Y/%m/%d-%H:%M:%S.%f;
        - %Y.%m.%dT%H:%M:%S.%f, %Y.%m.%d %H:%M:%S.%f, %Y.%m.%d-%H:%M:%S.%f;
        - %Y%m%dT%H:%M:%S.%f, %Y%m%d %H:%M:%S.%f, %Y%m%d-%H:%M:%S.%f;
        - %d-%m-%YT%H:%M:%S.%f, %d-%m-%Y %H:%M:%S.%f, %d-%m-%Y-%H:%M:%S.%f;
        - %d/%m/%YT%H:%M:%S.%f, %d/%m/%Y %H:%M:%S.%f, %d/%m/%Y-%H:%M:%S.%f;
        - %d.%m.%YT%H:%M:%S.%f, %d.%m.%Y %H:%M:%S.%f, %d.%m.%Y-%H:%M:%S.%f;
        - %m-%d-%YT%H:%M:%S.%f, %m-%d-%Y %H:%M:%S.%f, %m-%d-%Y-%H:%M:%S.%f;
        - %m/%d/%YT%H:%M:%S.%f, %m/%d/%Y %H:%M:%S.%f, %m/%d/%Y-%H:%M:%S.%f;
        - %m.%d.%YT%H:%M:%S.%f, %m.%d.%Y %H:%M:%S.%f, %m.%d.%Y-%H:%M:%S.%f;
        - %Y-%m-%dT%H:%M:%S%z, %Y-%m-%d %H:%M:%S%z, %Y-%m-%d-%H:%S%z;
        - %Y/%m/%dT%H:%M:%S%z, %Y/%m/%d %H:%M:%S%z, %Y/%m/%d-%H:%M:%S%z;
        - %Y.%m.%dT%H:%M:%S%z, %Y.%m.%d %H:%M:%S%z, %Y.%m.%d-%H:%M:%S%z;
        - %Y%m%dT%H:%M:%S%z, %Y%m%d %H:%M:%S%z, %Y%m%d-%H:%M:%S%z;
        - %d-%m-%YT%H:%M:%S%z, %d-%m-%Y %H:%M:%S%z, %d-%m-%Y-%H:%M:%S%z;
        - %d/%m/%YT%H:%M:%S%z, %d/%m/%Y %H:%M:%S%z, %d/%m/%Y-%H:%M:%S%z;
        - %d.%m.%YT%H:%M:%S%z, %d.%m.%Y %H:%M:%S%z, %d.%m.%Y-%H:%M:%S%z;
        - %m-%d-%YT%H:%M:%S%z, %m-%d-%Y %H:%M:%S%z, %m-%d-%Y-%H:%M:%S%z;
        - %m/%d/%YT%H:%M:%S%z, %m/%d/%Y %H:%M:%S%z, %m/%d/%Y-%H:%M:%S%z;
        - %m.%d.%YT%H:%M:%S%z, %m.%d.%Y %H:%M:%S%z, %m.%d.%Y-%H:%M:%S%z;
        - %Y-%m-%dT%H:%M:%S %z, %Y-%m-%d %H:%M:%S %z, %Y-%m-%d-%H:%M:%S %z;
        - %Y/%m/%dT%H:%M:%S %z, %Y/%m/%d %H:%M:%S %z, %Y/%m/%d-%H:%M:%S %z;
        - %Y.%m.%dT%H:%M:%S %z, %Y.%m.%d %H:%M:%S %z, %Y.%m.%d-%H:%M:%S %z;
        - %Y%m%dT%H:%M:%S %z, %Y%m%d %H:%M:%S %z, %Y%m%d-%H:%M:%S %z;
        - %d-%m-%YT%H:%M:%S %z, %d-%m-%Y %H:%M:%S %z, %d-%m-%Y-%H:%M:%S %z;
        - %d/%m/%YT%H:%M:%S %z, %d/%m/%Y %H:%M:%S %z, %d/%m/%Y-%H:%M:%S %z;
        - %d.%m.%YT%H:%M:%S %z, %d.%m.%Y %H:%M:%S %z, %d.%m.%Y-%H:%M:%S %z;
        - %m-%d-%YT%H:%M:%S %z, %m-%d-%Y %H:%M:%S %z, %m-%d-%Y-%H:%M:%S %z;
        - %m/%d/%YT%H:%M:%S %z, %m/%d/%Y %H:%M:%S %z, %m/%d/%Y-%H:%M:%S %z;
        - %m.%d.%YT%H:%M:%S %z, %m.%d.%Y %H:%M:%S %z, %m.%d.%Y-%H:%M:%S %z;
        - %Y-%m-%dT%H:%M:%S.%f, %Y-%m-%d %H:%M:%S.%f, %Y-%m-%d-%H:%M:%S.%f;
        - %Y/%m/%dT%H:%M:%S.%f, %Y/%m/%d %H:%M:%S.%f, %Y/%m/%d-%H:%M:%S.%f;
        - %Y.%m.%dT%H:%M:%S.%f, %Y.%m.%d %H:%M:%S.%f, %Y.%m.%d-%H:%M:%S.%f;
        - %Y%m%dT%H:%M:%S.%f, %Y%m%d %H:%M:%S.%f, %Y%m%d-%H:%M:%S.%f;
        - %d-%m-%YT%H:%M:%S.%f, %d-%m-%Y %H:%M:%S.%f, %d-%m-%Y-%H:%M:%S.%f;
        - %d/%m/%YT%H:%M:%S.%f, %d/%m/%Y %H:%M:%S.%f, %d/%m/%Y-%H:%M:%S.%f;
        - %d.%m.%YT%H:%M:%S.%f, %d.%m.%Y %H:%M:%S.%f, %d.%m.%Y-%H:%M:%S.%f;
        - %m-%d-%YT%H:%M:%S.%f, %m-%d-%Y %H:%M:%S.%f, %m-%d-%Y-%H:%M:%S.%f;
        - %m/%d/%YT%H:%M:%S.%f, %m/%d/%Y %H:%M:%S.%f, %m/%d/%Y-%H:%M:%S.%f;
        - %m.%d.%YT%H:%M:%S.%f, %m.%d.%Y %H:%M:%S.%f, %m.%d.%Y-%H:%M:%S.%f;
        - %Y-%m-%dT%H:%M:%S.%f %z, %Y-%m-%d %H:%M:%S.%f %z, %Y-%m-%d-%H:%M:%S.%f %z;
        - %Y/%m/%dT%H:%M:%S.%f %z, %Y/%m/%d %H:%M:%S.%f %z, %Y/%m/%d-%H:%M:%S.%f %z;
        - %Y.%m.%dT%H:%M:%S.%f %z, %Y.%m.%d %H:%M:%S.%f %z, %Y.%m.%d-%H:%M:%S.%f %z;
        - %Y%m%dT%H:%M:%S.%f %z, %Y%m%d %H:%M:%S.%f %z, %Y%m%d-%H:%M:%S.%f %z;
        - %d-%m-%YT%H:%M:%S.%f %z, %d-%m-%Y %H:%M:%S.%f %z, %d-%m-%Y-%H:%M:%S.%f %z;
        - %d/%m/%YT%H:%M:%S.%f %z, %d/%m/%Y %H:%M:%S.%f %z, %d/%m/%Y-%H:%M:%S.%f %z;
        - %d.%m.%YT%H:%M:%S.%f %z, %d.%m.%Y %H:%M:%S.%f %z, %d.%m.%Y-%H:%M:%S.%f %z;
        - %m-%d-%YT%H:%M:%S.%f %z, %m-%d-%Y %H:%M:%S.%f %z, %m-%d-%Y-%H:%M:%S.%f %z;
        - %m/%d/%YT%H:%M:%S.%f %z, %m/%d/%Y %H:%M:%S.%f %z, %m/%d/%Y-%H:%M:%S.%f %z;
        - %m.%d.%YT%H:%M:%S.%f %z, %m.%d.%Y %H:%M:%S.%f %z, %m.%d.%Y-%H:%M:%S.%f %z;
        - %Y-%m-%dT%H:%M:%S.%f%z, %Y-%m-%d %H:%M:%S%f%z, %Y-%m-%d-%H:%M:%S%f%z;
        - %Y/%m/%dT%H:%M:%S.%f%z, %Y/%m/%d %H:%M:%S%f%z, %Y/%m/%d-%H:%M:%S%f%z;
        - %Y.%m.%dT%H:%M:%S.%f%z, %Y.%m.%d %H:%M:%S%f%z, %Y.%m.%d-%H:%M:%S%f%z;
        - %Y%m%dT%H:%M:%S.%f%z, %Y%m%d %H:%M:%S%f%z, %Y%m%d-%H:%M:%S%f%z;
        - %d-%m-%YT%H:%M:%S.%f%z, %d-%m-%Y %H:%M:%S%f%z, %d-%m-%Y-%H:%M:%S%f%z;
        - %d/%m/%YT%H:%M:%S.%f%z, %d/%m/%Y %H:%M:%S%f%z, %d/%m/%Y-%H:%M:%S%f%z;
        - %d.%m.%YT%H:%M:%S.%f%z, %d.%m.%Y %H:%M:%S%f%z, %d.%m.%Y-%H:%M:%S%f%z;
        - %m-%d-%YT%H:%M:%S.%f%z, %m-%d-%Y %H:%M:%S%f%z, %m-%d-%Y-%H:%M:%S%f%z;
        - %m/%d/%YT%H:%M:%S.%f%z, %m/%d/%Y %H:%M:%S%f%z, %m/%d/%Y-%H:%M:%S%f%z;
        - %m.%d.%YT%H:%M:%S.%f%z, %m.%d.%Y %H:%M:%S%f%z, %m.%d.%Y-%H:%M:%S%f%z;
        - "%a, %d %b %Y %H:%M:%S %z", "%a, %d %B %Y %H:%M:%S %z";
        - "%A, %d %b %Y %H:%M:%S %z", "%A, %d %B %Y %H:%M:%S %z";
        - "%a, %d %b %Y %H:%M:%S %z", "%a, %d %B %Y %H:%M:%S.%f %z";
        - "%A, %d %b %Y %H:%M:%S %z", "%A, %d %B %Y %H:%M:%S.%f %z";
        - "%a, %d %b %Y %H:%M:%S %z", "%a, %d %B %Y %H:%M:%S.%f%z";
        - "%A, %d %b %Y %H:%M:%S %z", "%A, %d %B %Y %H:%M:%S.%f%z";
        - "%a, %d %b %Y %H:%M:%S %z", "%a, %d %B %Y %H:%M:%S%z";
        - "%A, %d %b %Y %H:%M:%S %z", "%A, %d %B %Y %H:%M:%S%z";

        Not supported formats (for them the default format will be used):
        - %m-%d-%y, %m/%d/%y, %m.%d.%y, %d-%m-%y, %d/%m/%y, %d.%m.%y, %y-%m-%d, %y/%m/%d, %y.%m.%d;
        - %m-%d-%yT%H:%M:%S, %m-%d-%y %H:%M:%S, %m-%d-%y-%H:%M:%S;
        - %d-%m-%yT%H:%M:%S, %d-%m-%y %H:%M:%S, %d-%m-%y-%H:%M:%S;
        - %y-%m-%dT%H:%M:%S, %y-%m-%d %H:%M:%S, %y-%m-%d-%H:%M:%S;
        - %m/%d/%yT%H:%M:%S, %m/%d/%y %H:%M:%S, %m/%d/%y-%H:%M:%S;
        - %d/%m/%yT%H:%M:%S, %d/%m/%y %H:%M:%S, %d/%m/%y-%H:%M:%S;
        - %y/%m/%dT%H:%M:%S, %y/%m/%d %H:%M:%S, %y/%m/%d-%H:%M:%S;
        - %m.%d.%yT%H:%M:%S, %m.%d.%y %H:%M:%S, %m.%d.%y-%H:%M:%S;
        - %d.%m.%yT%H:%M:%S, %d.%m.%y %H:%M:%S, %d.%m.%y-%H:%M:%S;
        - %y.%m.%dT%H:%M:%S, %y.%m.%d %H:%M:%S, %y.%m.%d-%H:%M:%S;
        - %m-%d-%yT%H:%M:%S.%f, %m-%d-%y %H:%M:%S.%f, %m-%d-%y-%H:%M:%S.%f;
        - %d-%m-%yT%H:%M:%S.%f, %d-%m-%y %H:%M:%S.%f, %d-%m-%y-%H:%M:%S.%f;
        - %y-%m-%dT%H:%M:%S.%f, %y-%m-%d %H:%M:%S.%f, %y-%m-%d-%H:%M:%S.%f;
        - %m/%d/%yT%H:%M:%S.%f, %m/%d/%y %H:%M:%S.%f, %m/%d/%y-%H:%M:%S.%f;
        - %d/%m/%yT%H:%M:%S.%f, %d/%m/%y %H:%M:%S.%f, %d/%m/%y-%H:%M:%S.%f;
        - %y/%m/%dT%H:%M:%S.%f, %y/%m/%d %H:%M:%S.%f, %y/%m/%d-%H:%M:%S.%f;
        - %m.%d.%yT%H:%M:%S.%f, %m.%d.%y %H:%M:%S.%f, %m.%d.%y-%H:%M:%S.%f;
        - %d.%m.%yT%H:%M:%S.%f, %d.%m.%y %H:%M:%S.%f, %d.%m.%y-%H:%M:%S.%f;
        - %y.%m.%dT%H:%M:%S.%f, %y.%m.%d %H:%M:%S.%f, %y.%m.%d-%H:%M:%S.%f;
        - %m-%d-%yT%H:%M:%S.%f %z, %m-%d-%y %H:%M:%S.%f %z, %m-%d-%y-%H:%M:%S.%f %z;
        - %d-%m-%yT%H:%M:%S.%f %z, %d-%m-%y %H:%M:%S.%f %z, %d-%m-%y-%H:%M:%S.%f %z;
        - %y-%m-%dT%H:%M:%S.%f %z, %y-%m-%d %H:%M:%S.%f %z, %y-%m-%d-%H:%M:%S.%f %z;
        - %m/%d/%yT%H:%M:%S.%f %z, %m/%d/%y %H:%M:%S.%f %z, %m/%d/%y-%H:%M:%S.%f %z;
        - %d/%m/%yT%H:%M:%S.%f %z, %d/%m/%y %H:%M:%S.%f %z, %d/%m/%y-%H:%M:%S.%f %z;
        - %y/%m/%dT%H:%M:%S.%f %z, %y/%m/%d %H:%M:%S.%f %z, %y/%m/%d-%H:%M:%S.%f %z;
        - %m.%d.%yT%H:%M:%S.%f %z, %m.%d.%y %H:%M:%S.%f %z, %m.%d.%y-%H:%M:%S.%f %z;
        - %d.%m.%yT%H:%M:%S.%f %z, %d.%m.%y %H:%M:%S.%f %z, %d.%m.%y-%H:%M:%S.%f %z;
        - %y.%m.%dT%H:%M:%S.%f %z, %y.%m.%d %H:%M:%S.%f %z, %y.%m.%d-%H:%M:%S.%f %z;
        - %m-%d-%yT%H:%M:%S.%f%z, %m-%d-%y %H:%M:%S.%f%z, %m-%d-%y-%H:%M:%S.%f%z;
        - %d-%m-%yT%H:%M:%S.%f%z, %d-%m-%y %H:%M:%S.%f%z, %d-%m-%y-%H:%M:%S.%f%z;
        - %y-%m-%dT%H:%M:%S.%f%z, %y-%m-%d %H:%M:%S.%f%z, %y-%m-%d-%H:%M:%S.%f%z;
        - %m/%d/%yT%H:%M:%S.%f%z, %m/%d/%y %H:%M:%S.%f%z, %m/%d/%y-%H:%M:%S.%f%z;
        - %d/%m/%yT%H:%M:%S.%f%z, %d/%m/%y %H:%M:%S.%f%z, %d/%m/%y-%H:%M:%S.%f%z;
        - %y/%m/%dT%H:%M:%S.%f%z, %y/%m/%d %H:%M:%S.%f%z, %y/%m/%d-%H:%M:%S.%f%z;
        - %m.%d.%yT%H:%M:%S.%f%z, %m.%d.%y %H:%M:%S.%f%z, %m.%d.%y-%H:%M:%S.%f%z;
        - %d.%m.%yT%H:%M:%S.%f%z, %d.%m.%y %H:%M:%S.%f%z, %d.%m.%y-%H:%M:%S.%f%z;
        - %y.%m.%dT%H:%M:%S.%f%z, %y.%m.%d %H:%M:%S.%f%z, %y.%m.%d-%H:%M:%S.%f%z;
        """

        date_text = self.df[column].dropna()

        n_samples = min(100, len(date_text))
        sample = date_text.sample(n_samples).values

        types = [self.__get_date_format(i) for i in sample]

        if not any(types):
            return "%d-%m-%Y"

        most_common = Counter(types).most_common()
        if most_common[0][0] is None:
            chosen_format = most_common[1][0] if len(most_common) > 1 else "%d-%m-%Y"
        else:
            chosen_format = most_common[0][0]

        return chosen_format

    @staticmethod
    def __get_date_format(date_string: str) -> str:
        """
        Get a date format and remove timezone abbreviation, if present
        """
        date_format = guess_datetime_format(date_string)
        if not date_format:
            return None

        match = TIMEZONE_REGEX.search(date_format)
        if match and (abbr := match.group("tz_abbr")):
            date_format = date_format.replace(abbr, "%Z")

        return date_format

    def _set_date_format(self):
        """
        Define the date format for each date column
        """
        self.date_mapping = {
            column: self.__define_date_format(column)
            for column in self.date_columns
        }

    def _csv_data_pipeline(self):
        """
        Divide columns in dataframe into groups -
        binary, categorical, integer, float, string, date
        in case metadata of the table is absent
        """
        for col in self.df.columns:
            col_no_na = self.df[col].dropna()

            if col_no_na.dtype in ["int", "int64"] or col in self.cast_to_integer:
                self.int_columns.add(col)
            elif col_no_na.dtype in ["float", "float64"] or col in self.cast_to_float:
                self.float_columns.add(col)

        float_to_int_cols = set()
        for col in self.float_columns:
            if all(x.is_integer() for x in self.df[col].dropna()):
                float_to_int_cols.add(col)

        self.int_columns = (self.int_columns | float_to_int_cols) - (
            self.categorical_columns | self.binary_columns
        )
        self.float_columns = (
            self.float_columns - self.categorical_columns - self.int_columns - self.binary_columns
        )
        self.str_columns = (
            set(self.df.columns)
            - self.float_columns
            - self.categorical_columns
            - self.int_columns
            - self.binary_columns
            - self.long_text_columns
            - self.email_columns
            - self.uuid_columns
        )
        self.categorical_columns -= self.long_text_columns
        self._set_date_columns()
        self.str_columns -= self.date_columns
        self.uuid_columns = self.uuid_columns - self.categorical_columns - self.binary_columns
        self.uuid_columns_types = {
            k: v for k, v in self.uuid_columns_types.items() if k in self.uuid_columns
        }
        self._set_date_format()

    def _avro_data_pipeline(self):
        """
        Divide columns in dataframe into groups - binary, categorical, integer, float, string, date
        in case metadata of the table in Avro format is present
        """
        self.int_columns = set(
            column for column, data_type in self.fields.items() if data_type == "int"
        )
        self.int_columns = (self.int_columns - self.categorical_columns - self.binary_columns)
        self.float_columns = set(
            column for column, data_type in self.fields.items() if data_type == "float"
        )
        self.float_columns = self.float_columns - self.categorical_columns - self.binary_columns
        self.str_columns = set(
            column for column, data_type in self.fields.items() if data_type == "string"
        )
        self.categorical_columns -= self.long_text_columns
        self.str_columns = (
            self.str_columns
            - self.categorical_columns
            - self.binary_columns
            - self.long_text_columns
            - self.email_columns
            - self.uuid_columns
        )
        self._set_date_columns()
        self.str_columns -= self.date_columns
        self.uuid_columns = self.uuid_columns - self.categorical_columns - self.binary_columns
        self.uuid_columns_types = {
            k: v for k, v in self.uuid_columns_types.items() if k in self.uuid_columns
        }
        self._set_date_format()

    def _detection_pipeline(self):
        if self.schema_format == "CSV":
            self._csv_data_pipeline()
        elif self.schema_format == "Avro":
            self._avro_data_pipeline()

        assert (
            len(self.str_columns) + len(self.float_columns) + len(self.int_columns)
            + len(self.date_columns) + len(self.categorical_columns) + len(self.binary_columns)
            + len(self.long_text_columns) + len(self.uuid_columns) + len(self.email_columns)
        ) == len(self.df.columns), (
            "According to number of columns with defined types, "
            "column types are not identified correctly"
        )

        logger.debug(
            f"Count of string columns: {len(self.str_columns)}; "
            + f"Count of email columns: {len(self.email_columns)}; "
            + f"Count of float columns: {len(self.float_columns)}; "
            + f"Count of int columns: {len(self.int_columns)}; "
            + f"Count of categorical columns: {len(self.categorical_columns)}; "
            + f"Count of date columns: {len(self.date_columns)}; "
            + f"Count of binary columns: {len(self.binary_columns)}; "
            + f"Count of long text columns: {len(self.long_text_columns)}; "
            + f"Count of uuid columns: {len(self.uuid_columns)}"
        )

    def assign_feature(self, feature, columns):
        name = feature.original_name

        if name in self.features:
            raise Exception(f"{name} is already contained in features")

        if not isinstance(columns, (list, tuple)):
            columns = [columns]

        self.features[name] = feature
        self.columns[name] = columns

    def fit(self):
        for name, feature in self.features.items():
            feature.fit(self.df[self.columns[name]], date_mapping=self.date_mapping)

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
        self.fit()
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
                logger.trace(f'Column {name} is being inverse transformed.')
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
        num_zeros = (self.df[feature] == 0).sum()
        many_zeros_feature = (num_zeros / (len(self.df[feature]))) > zero_cutoff
        if many_zeros_feature:
            feature_zero = feature + "_zero"
            self.df[feature_zero] = self.df[feature].apply(lambda x: 0 if x == 0 else 1)
            if not isnull_feature.any():
                logger.info(
                    f"Column '{feature}' contains {num_zeros} "
                    f"({round(num_zeros * 100 / (len(self.df[feature])))}%) "
                    f"zero values out of {(len(self.df[feature]))}. "
                    f"Ratio of zeros will be preserved in synthetic data."
                )
                return (feature, feature_zero)
        if isnull_feature.any():
            nan_number = isnull_feature.sum()
            logger.info(
                f"Column '{feature}' contains {nan_number} "
                f"({round(nan_number * 100 / len(isnull_feature), 2)}%) "
                f"empty values out of {len(isnull_feature)}. "
                f"Filling them with {fillna_strategy or 'zero'}."
            )
            if fillna_strategy == "mean":
                fillna_value = self.df[feature].mean()
            elif fillna_strategy == "mode":
                fillna_value = self.df[feature].dropna().mode().sample(1).values[0]
            elif (fillna_strategy == "text") or (fillna_strategy == "email"):
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

    def _preprocess_categ_params(self, feature: str, strategy: Literal["?", "fill"] = "?"):
        """
        Preprocess categorical columns by filling NaN values with a strategy
        """
        if self.df[feature].isnull().any():
            if strategy == "?":
                self.df[feature] = self.df[feature].fillna("?").astype(str)
            if strategy == "fill":
                self.df[feature] = self.df[feature].fillna(method="bfill").fillna(method="ffill")
                logger.info(
                    f"Filling NaN values in column - '{feature}' with 'bfill' and 'ffill' methods."
                )
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

    def _assign_email_feature(self, feature):
        """
        Assign email feature to text columns
        """
        features = self._preprocess_nan_cols(feature, fillna_strategy="email")
        max_len, rnn_units = 15, 32
        self.assign_feature(
            EmailFeature(features[0], text_max_len=max_len, rnn_units=rnn_units),
            features[0],
        )
        logger.info(f"Column '{features[0]}' assigned as email feature")

        # TODO: encapsulate this logic in a separate function
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

    def _assign_categ_feature(self, feature, strategy: Literal["?", "fill"] = "?"):
        """
        Assign categorical based feature to categorical columns
        """
        feature = self._preprocess_categ_params(feature, strategy)
        self.assign_feature(CategoricalFeature(feature), feature)
        logger.info(f"Column '{feature}' assigned as categorical based feature.")

    def _preprocess_dates_with_timezone(self, feature):
        """
        Preprocess date columns with timezone information,
        adding a new column with timezone information if applicable.
        """
        timezone_data = self.df[feature].map(lambda x: fetch_timezone(x))
        if timezone_data.isnull().all():
            return

        self.df[f"{feature}_tz"] = timezone_data
        percent_with_tz = round(timezone_data.notnull().mean() * 100, 2)
        unique_tz = timezone_data.dropna().unique()
        unique_tz = ', '.join(
            unique_tz[:5].tolist() + ["etc"] if len(unique_tz) > 5 else unique_tz
        )

        logger.info(
            f"Column '{feature}' contains {percent_with_tz}% dates with time zone. "
            f"Unique time zones: {unique_tz}."
        )
        self._assign_categ_feature(f"{feature}_tz", strategy="fill")

    def _assign_date_feature(self, feature):
        """
        Assign date feature to date columns
        """
        date_format = self.date_mapping.get(feature)
        if "%z" in date_format.lower():
            self._preprocess_dates_with_timezone(feature)
        features = self._preprocess_nan_cols(feature, fillna_strategy="mode")
        self.assign_feature(DateFeature(features[0]), features[0])
        logger.info(f"Column '{features[0]}' assigned as date feature.")
        if len(features) == 2:
            self.null_num_column_names.append(features[1])
            self.assign_feature(ContinuousFeature(features[1]), features[1])
            logger.info(f"Column '{features[1]}' assigned as float feature.")

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
        for fk_column in self.fk_columns:
            features = self._preprocess_nan_cols(fk_column, fillna_strategy="mode")
            if len(features) > 1:
                self.assign_feature(
                    ContinuousFeature(features[1], column_type=int), features[1]
                )

    def _assign_uuid_null_feature(self, feature):
        """
        Assign corresponding to uuid column null column and preprocess if required.
        """
        logger.info(f"Column '{feature}' defined as UUID column")
        features = self._preprocess_nan_cols(feature, fillna_strategy="text")
        if len(features) == 2:
            self.null_num_column_names.append(features[1])
            self.assign_feature(ContinuousFeature(features[1]), features[1])
            logger.info(f"Column '{features[1]}' assigned as float feature")

    def __prepare_primary_key_mapping(self):
        self.primary_keys_mapping.update(self.unique_keys_mapping)
        pk_uq_keys_mapping = self.primary_keys_mapping
        if pk_uq_keys_mapping:
            self.__set_types(pk_uq_keys_mapping)
            self.__map_text_pk()

    def _assign_feature(self, column: str):
        """
        Assign feature to the column based on its type
        """
        for column_type, handler in self.dispatcher.items():
            if column in getattr(self, column_type, []):
                handler(column)
                return

    @property
    def dispatcher(self):
        """
        Dispatcher for assigning features to columns
        """
        return {
            "str_columns": self._assign_char_feature,
            "email_columns": self._assign_email_feature,
            "float_columns": self._assign_float_feature,
            "int_columns": self._assign_int_feature,
            "categorical_columns": self._assign_categ_feature,
            "date_columns": self._assign_date_feature,
            "binary_columns": self._assign_binary_feature,
            "uuid_columns": self._assign_uuid_null_feature,
        }

    def __assign_features(self):
        """
        Assign features to the columns based on their types
        """
        for column in [col for col in self.df.columns]:
            self._assign_feature(column)

    def _ensure_technical_column_if_no_features(self, additional_message: str = ""):
        """
        Workaround for the case when all columns are dropped.
        Add a technical column to proceed with the training process.
        """
        if not self.features:
            tech_column = "syngen_tech_column"
            logger.info(
                f"Since all columns in the table '{self.table_name}' "
                f"are uuid/key/long text{additional_message} columns, "
                "there are no suitable columns to train on. "
                f"A technical column '{tech_column}' will be added "
                "to proceed with the training process "
                "and will be removed afterwards."
            )
            self.df[tech_column] = 1
            self._assign_float_feature(tech_column)
            self.tech_columns.add(tech_column)

    def pipeline(self) -> pd.DataFrame:
        if self.foreign_keys_list:
            self._assign_fk_feature()
            self._preprocess_fk_params()
            self._drop_fk_columns()

        self.__prepare_primary_key_mapping()

        self.__assign_features()
        self._ensure_technical_column_if_no_features()

        self.fit()

        # The end of the run related to the preprocessing stage
        MlflowTracker().end_run()

        return self.df
