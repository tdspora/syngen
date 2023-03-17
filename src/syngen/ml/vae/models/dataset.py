from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import pickle

from loguru import logger
import numpy as np
import dill
import pandas as pd
from scipy.stats import gaussian_kde

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
    get_date_columns
)
from syngen.ml.data_loaders import DataLoader
from syngen.ml.utils import slugify_parameters


@dataclass
class Dataset:
    df: pd.DataFrame
    schema: Optional[Dict]
    metadata: Optional[Dict]
    table_name: str
    fk_kde_path: str
    features: Dict = field(init=False)
    columns: Dict = field(init=False)
    is_fitted: bool = field(init=False)
    all_columns: List = field(init=False)
    null_num_column_names: List = field(init=False)
    zero_num_column_names: List = field(init=False)
    nan_labels_dict: Dict = field(init=False)

    def __post_init__(self):
        self.features = dict()
        self.columns = dict()
        self.is_fitted = False
        self.all_columns = list()
        self.null_num_column_names = list()
        self.zero_num_column_names = list()
        self.nan_labels_dict = dict()

    def __set_pk_key(self, config_of_keys: Dict):
        """
        Set up primary key for the table
        """
        self.primary_keys_mapping = {
            key: value for (key, value) in config_of_keys.items()
            if config_of_keys.get(key).get("type") == "PK"
        }
        self.primary_keys_list = list(self.primary_keys_mapping.keys())
        self.primary_key_name = self.primary_keys_list[0] if self.primary_keys_list else None
        pk_columns_lists = [val['columns'] for val in self.primary_keys_mapping.values()]
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
            key: value for (key, value) in config_of_keys.items()
            if config_of_keys.get(key).get("type") == "UQ"
        }
        self.unique_keys_mapping_list = list(self.unique_keys_mapping.keys())
        self.unique_keys_list = self.unique_keys_mapping_list if self.unique_keys_mapping_list else []
        uq_columns_lists = [val['columns'] for val in self.unique_keys_mapping.values()]
        self.uq_columns = [col for uq_cols in uq_columns_lists for col in uq_cols]

        if self.unique_keys_list:
            logger.info(f"The unique keys were set: {self.unique_keys_list}")
        if not self.unique_keys_list:
            logger.info("No unique keys were set.")

    def __set_fk_keys(self, config_of_keys: Dict):
        """
        Set up foreign keys for the table
        """
        self.foreign_keys_mapping = {
            key: value for (key, value) in config_of_keys.items()
            if config_of_keys.get(key).get("type") == "FK"
        }
        self.foreign_keys_list = list(self.foreign_keys_mapping.keys())
        fk_columns_lists = [val['columns'] for val in self.foreign_keys_mapping.values()]
        self.fk_columns = [col for fk_cols in fk_columns_lists for col in fk_cols]

        if self.foreign_keys_list:
            logger.info(f"The following foreign keys were set: {self.foreign_keys_list}")
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
                column_type = str if column in (self.str_columns | self.categ_columns | self.date_columns) else float
                self.pk_uq_keys_types[column] = column_type

    def __map_text_pk(self):
        for pk, pk_type in self.pk_uq_keys_types.items():
            if pk_type is str:
                mapper = {k: n for n, k in enumerate(self.df[pk])}
                with open(f"{self.fk_kde_path}{pk}_mapper.pkl", "wb") as file:
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

    def set_metadata(self):
        self.__set_metadata(self.metadata, self.table_name)
        self.__data_pipeline(self.df, self.schema)

    @staticmethod
    def _update_schema(schema: Dict[str, Dict[str, str]], df: pd.DataFrame):
        """
        Synchronize the schema of the table with dataframe
        """
        schema["fields"] = {
            column: data_type for column, data_type in schema.get("fields").items()
            if column in df.columns
        }
        return schema

    def _check_if_column_in_removed(self, schema: Dict):
        """
        Exclude the column from the list of categorical columns
        if it was removed previously as empty column
        """
        removed = [col for col, data_type in schema.get("fields", {}).items() if data_type == "removed"]
        for col in list(self.categ_columns):
            if col in removed:
                self.categ_columns.remove(col)
                logger.warning(
                    f"The column '{col}' was excluded from the list of categorical columns "
                    f"as far as this column is empty and was removed from the table - '{self.table_name}'"
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
                f"as far as this column was set as the {key_type} of the table - '{self.table_name}'")
            self.categ_columns.remove(column)

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
            [
                col for col in self.binary_columns
                if col not in self.categ_columns
            ]
        )

    def _fetch_categorical_columns(self, schema: Dict):
        """
        Fetch the categorical columns from the metadata
        """
        metadata_of_table = self.metadata.get(self.table_name)

        self.categ_columns = set()

        if metadata_of_table is not None:
            self.categ_columns = \
                set(metadata_of_table.get("train_settings", {}).get("column_types", {}).get("categorical", []))

            if self.categ_columns:
                self._check_if_column_in_removed(schema)
                self._check_if_column_existed()
                self._check_if_not_key_column()
                self._check_if_column_binary()

            if self.categ_columns:
                logger.info(
                    f"The columns - {', '.join(self.categ_columns)} were set as categorical "
                    f"due to the information from the metadata of the table - '{self.table_name}'")

    def _set_binary_columns(self, df: pd.DataFrame):
        """
        Set up the list of binary columns based on the count of unique values in the column
        """
        self.binary_columns = set([col for col in df.columns if df[col].fillna("?").nunique() == 2])

    def _define_categorical_columns(self, df):
        """
        Define the list of categorical columns based on the count of unique values in the column
        """
        defined_columns = set(
            [
                col for col in df.columns
                if df[col].dropna().nunique() <= 50 and col not in self.binary_columns
            ]
        )
        self.categ_columns.update(defined_columns)

    def _set_categorical_columns(self, df: pd.DataFrame, schema: Dict):
        """
        Set up the list of categorical columns
        """
        self._fetch_categorical_columns(schema)
        self._define_categorical_columns(df)

    def _general_data_pipeline(self, df: pd.DataFrame, schema: Dict, check_object_on_float: bool = False):
        """
        Divide columns in dataframe into groups - binary, categorical, integer, float, string, date
        in case metadata of the table is absent
        """
        if check_object_on_float:
            columns_nan_labels = get_nan_labels(df)
            df = nan_labels_to_float(df, columns_nan_labels)

        self._set_binary_columns(df)
        self._set_categorical_columns(df, schema)
        tmp_df = get_tmp_df(df)
        self.float_columns = set(tmp_df.select_dtypes(include=["float", "float64"]).columns)
        self.int_columns = set(tmp_df.select_dtypes(include=["int", "int64"]).columns)

        float_to_int_cols = set()
        for col in self.float_columns:
            if all(x.is_integer() for x in tmp_df[col]):
                float_to_int_cols.add(col)

        self.int_columns = (self.int_columns | float_to_int_cols) - (self.categ_columns | self.binary_columns)
        self.float_columns = self.float_columns - self.categ_columns - self.int_columns - self.binary_columns
        self.str_columns = \
            set(tmp_df.columns) - self.float_columns - self.categ_columns - self.int_columns - self.binary_columns
        self.date_columns = get_date_columns(tmp_df, list(self.str_columns))
        self.str_columns -= self.date_columns

    def _avro_data_pipeline(self, df, schema):
        """
        Divide columns in dataframe into groups - binary, categorical, integer, float, string, date
        in case metadata of the table in Avro format is present
        """
        logger.info(f"The schema of table - {self.table_name} was received")
        self._set_binary_columns(df)
        self._set_categorical_columns(df, schema)
        self.int_columns = set(column for column, data_type in schema.items() if data_type == 'int')
        self.int_columns = self.int_columns - self.categ_columns - self.binary_columns
        self.float_columns = set(column for column, data_type in schema.items() if data_type == 'float')
        self.float_columns = self.float_columns - self.categ_columns - self.binary_columns
        self.str_columns = set(column for column, data_type in schema.items() if data_type == 'string')
        self.str_columns = self.str_columns - self.categ_columns - self.int_columns - self.binary_columns
        self.date_columns = get_date_columns(df, list(self.str_columns)) - self.categ_columns - self.binary_columns
        self.str_columns -= self.date_columns

    def __data_pipeline(self, df: pd.DataFrame, schema: Optional[Dict]):
        if schema.get("format") == "CSV":
            self._general_data_pipeline(df, schema)
        elif schema.get("format") == 'Avro':
            schema = self._update_schema(schema, df)
            self._avro_data_pipeline(df, schema.get("fields"))

        assert len(self.str_columns) + \
               len(self.float_columns) + \
               len(self.int_columns) + \
               len(self.date_columns) + \
               len(self.categ_columns) + \
               len(self.binary_columns) == len(df.columns), "According to number of columns with defined types, " \
                                                            "column types are not identified correctly."

        logger.debug(
            f"Count of string columns: {len(self.str_columns)}; "
            + f"Count of float columns: {len(self.float_columns)}; "
            + f"Count of int columns: {len(self.int_columns)}; "
            + f"Count of categorical columns: {len(self.categ_columns)}; "
            + f"Count of date columns: {len(self.date_columns)}; "
            + f"Count of binary columns: {len(self.binary_columns)}"
        )

    def assign_feature(self, feature, columns):
        name = feature.name

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
        return (len(data) == len(self.features)) or (len(data) + len(self.fk_columns) == len(self.features))

    def inverse_transform(self, data, excluded_features=set()):
        inverse_transformed_data = list()
        column_names = list()
        if not isinstance(data, list):
            data = [data]
        assert self._check_count_features(data)

        for transformed_data, (name, feature) in zip(data, self.features.items()):
            if name not in excluded_features and name not in self.fk_columns:
                column_names.extend(self.columns[name])
                inverse_transformed_data.append(
                    feature.inverse_transform(transformed_data)
                )

        stacked_data = np.column_stack(inverse_transformed_data)
        data = pd.DataFrame(stacked_data, columns=column_names)

        return data

    def _preprocess_str_params(self, feature: str) -> Tuple[int, int]:
        self.df[feature] = self.df[feature].fillna("")
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
            tuple: Tuple that consists of either feature name or both feature name and new null feature name.
        """
        isnull_feature = pd.isnull(self.df[feature])
        many_zeros_feature = ((self.df[feature] == 0).sum() / (len(self.df[feature])) >
                              zero_cutoff)
        if many_zeros_feature:
            feature_zero = feature + '_zero'
            self.df[feature_zero] = self.df[feature].apply(lambda x: 0 if x == 0 else 1)
            if not isnull_feature.any():
                return (feature, feature_zero)
        if isnull_feature.any():
            nan_number = isnull_feature.sum()
            logger.info(f"Column '{feature}' contains {nan_number} ({round(nan_number * 100 / len(isnull_feature))}%) "
                        f"empty values out of {len(isnull_feature)}. Filling them with {fillna_strategy or 'zero'}.")
            if fillna_strategy == "mean":
                fillna_value = self.df[feature].mean()
            elif fillna_strategy == "mode":
                fillna_value = self.df[feature].dropna().mode().sample(1).values[0]
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
            with open(f"{fk_kde_path.replace(table_name, pk_table)}{pk_column}_mapper.pkl", "rb") as file:
                mapper = pickle.load(file)
            return mapper
        except FileNotFoundError:
            logger.warning(f"The mapper for the {fk_column} text key is not found. "
                           f"Simple sampling will be used.")

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
                        fk_kde_path=self.fk_kde_path,
                        table_name=self.table_name,
                        pk_table=correspondent_pk_table,
                        pk_column=correspondent_pk_col,
                        fk_column=fk_column
                    )
                    fk_column_values = fk_column_values.map(mapper)
                kde = gaussian_kde(fk_column_values)
                self._save_kde_artifacts(kde=kde, fk_kde_path=self.fk_kde_path, fk_column=fk_column)

    def __drop_fk_columns(self):
        """
        Drop columns in dataframe which defined as foreign key
        """
        for fk in self.foreign_keys_list:
            fk_columns = self.foreign_keys_mapping.get(fk).get("columns")
            for fk_column in fk_columns:
                self.df = self.df.drop(fk_column, axis=1)
                logger.debug(f"The column - '{fk_column}' of foreign key '{fk}' dropped from training "
                             f"and will be sampled from the PK table")

    def __sample_only_joined_rows(self, fk):
        references = self.foreign_keys_mapping.get(fk).get("references")
        pk_table = references.get("table")
        pk_table_data, schema = \
            DataLoader(f"model_artifacts/tmp_store/{pk_table}/input_data_{pk_table}.csv").load_data()
        pk_column_label = references.get("columns")[0]

        drop_index = self.df[~self.df[fk].isin(pk_table_data[pk_column_label].values)].index
        if len(drop_index) > 0:
            logger.info(f"{len(drop_index)} rows were deleted, as they did not have matching primary keys.")
            logger.info(f"{len(self.df) - len(drop_index)} rows are left in table as input.")
        self.df = self.df.drop(drop_index)

    def _assign_char_feature(self, feature):
        """
        Assign text based feature to text columns
        """
        max_len, rnn_units = self._preprocess_str_params(feature)
        self.assign_feature(
            CharBasedTextFeature(
                feature, text_max_len=max_len, rnn_units=rnn_units
            ),
            feature,
        )
        logger.debug(f"Column '{feature}' assigned as text based feature")

    def _assign_float_feature(self, feature):
        """
        Assign float based feature to float columns
        """
        # num_bins = self.find_clusters(df, float_columns)
        features = self._preprocess_nan_cols(feature, fillna_strategy="mean")
        if len(features) == 2 and features[1].endswith("_null"):
            self.null_num_column_names.append(features[1])
        if len(features) == 2 and features[1].endswith('_zero'):
            self.zero_num_column_names.append(features[1])
        if len(features) == 3:
            self.zero_num_column_names.append(features[2])
        for feature in features:
            self.assign_feature(
                ContinuousFeature(feature, column_type=float), feature
            )
            logger.debug(f"Column '{feature}' assigned as float based feature")

    def _assign_int_feature(self, feature):
        """
        Assign int based feature to int columns
        """
        # num_bins = self.find_clusters(df, int_columns)
        features = self._preprocess_nan_cols(feature, fillna_strategy="mean")
        if len(features) == 2 and features[1].endswith("_null"):
            self.null_num_column_names.append(features[1])
        if len(features) == 2 and features[1].endswith('_zero'):
            self.zero_num_column_names.append(features[1])
        if len(features) == 3:
            self.zero_num_column_names.append(features[2])
        for feature in features:
            self.assign_feature(
                ContinuousFeature(feature, column_type=int), feature
            )
            logger.debug(f"Column '{feature}' assigned as int based feature")

    def _assign_categ_feature(self, feature):
        """
        Assign categorical based feature to categorical columns
        """
        feature = self._preprocess_categ_params(feature)
        self.assign_feature(CategoricalFeature(feature), feature)
        logger.debug(f"Column '{feature}' assigned as categorical based feature")

    def _assign_date_feature(self, feature):
        """
        Assign date feature to date columns
        """
        features = self._preprocess_nan_cols(feature, fillna_strategy="mode")
        self.assign_feature(DateFeature(features[0]), features[0])
        logger.debug(f"Column '{features[0]}' assigned as date feature")
        if len(features) == 2:
            self.null_num_column_names.append(features[1])
            self.assign_feature(ContinuousFeature(features[1], column_type=int), features[1])
            logger.debug(f"Column '{features[1]}' assigned as int feature")

    def _assign_binary_feature(self, feature):
        """
        Assign binary feature to binary columns
        """
        feature = self._preprocess_categ_params(feature)
        self.assign_feature(BinaryFeature(feature), feature)
        logger.debug(f"Column '{feature}' assigned as binary feature")

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
            self.__drop_fk_columns()

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
