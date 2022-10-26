from typing import Set

from loguru import logger
import numpy as np
import dill
import pandas as pd
from scipy.stats import gaussian_kde

from syngen.ml.vae.models.features import InverseTransformer
from syngen.ml.vae.models.features import (
    CategoricalFeature,
    CharBasedTextFeature,
    ContinuousFeature,
    DateFeature,
    BinaryFeature,
)
from syngen.ml.pipeline.pipeline import (
    data_pipeline,
    get_nan_labels,
    nan_labels_to_float
)


class Dataset:
    def __init__(self, df: pd.DataFrame, metadata: dict, table_name: str, kde_path: str):
        self.df = df
        self.__set_metadata(metadata, table_name)
        self.features = dict()
        self.columns = dict()
        self.is_fitted = False
        self.all_columns = []
        self.null_num_column_names = []
        self.nan_labels_dict = {}
        self.fk_kde_path = kde_path

    def __set_metadata(self, metadata: dict, table_name: str):
        self.foreign_keys_list = []  # For compatibility with the Enterprise version
        self.token_keys_list = []  # For compatibility with the Enterprise version
        self.table_name = table_name
        config_of_keys = metadata.get(table_name, {}).get("keys")
        if config_of_keys is not None:
            self.foreign_keys_mapping = {
                key: value for (key, value) in config_of_keys.items()
                if config_of_keys.get(key).get("type") == "FK"
            }
            self.foreign_keys_list = list(self.foreign_keys_mapping.keys())
            self.foreign_key_names = self.foreign_keys_list if self.foreign_keys_list else None
        else:
            self.foreign_key_names = None

    def assign_feature(self, feature, columns):
        name = feature.name

        if name in self.features:
            raise Exception("%s is already contained in features" % name)

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
        for name, feature in self.features.items():
            if name not in (excluded_features and self.foreign_keys_list and self.token_keys_list):
                transformed_features.append(feature.transform(data[self.columns[name]]))
        return transformed_features

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def _check_count_features(self, data):
        return (len(data) == len(self.features)) or (len(data) + len(self.foreign_keys_list) == len(self.features))

    def inverse_transform(self, data, excluded_features=set()):
        inverse_transformed_data = list()
        column_names = list()
        if not isinstance(data, list):
            data = [data]
        assert self._check_count_features(data)

        self.inverse_transformers = {}

        for transformed_data, (name, feature) in zip(data, self.features.items()):
            if name not in excluded_features and name not in self.foreign_keys_list and name not in self.token_keys_list:
                column_names.extend(self.columns[name])
                inverse_transformed_data.append(
                    feature.inverse_transform(transformed_data)
                )
                self.inverse_transformers[name] = InverseTransformer(
                    name, feature.inverse_transform
                )

        stacked_data = np.column_stack(inverse_transformed_data)
        data = pd.DataFrame(stacked_data, columns=column_names)

        return data

    def _preprocess_str_params(self, feature: str):
        self.df[feature] = self.df[feature].fillna("")
        max_len = int(self.df[feature].apply(lambda line: len(line)).max())
        if 1 < max_len < 7:
            rnn_units = 32

        if 6 < max_len < 13:
            rnn_units = 128

        if 12 < max_len < 17:
            rnn_units = 256

        if max_len > 16:
            rnn_units = 512

        return max_len, rnn_units

    def _preprocess_float_params(
        self, feature: str, fillna_strategy: str = None
    ) -> tuple:
        """Fill NaN values in numeric column with some value according to strategy.
        If NaN values exist additional column is created and added to DataFrame.
        This column has value of 1 in case corresponding row contains NaN and 0 otherwise.
        New column name is built like 'column name'+'_null'.

        Args:
            feature (str): Feature name.
            fillna_strategy (str, optional): Can be 'mean', 'mode' or None.
                                             If None NaN values in column are replaced with 0.
                                             Defaults to None.

        Returns:
            tuple: Tuple that consists of either feature name or both feature name and new null feature name.
        """
        if fillna_strategy == "mean":
            fillna_value = self.df[feature].mean()
        elif fillna_strategy == "mode":
            fillna_value = self.df[feature].mode()
        else:
            fillna_value = 0

        isnull_feature = pd.isnull(self.df[feature])

        if isnull_feature.any():
            feature_null = feature + "_null"
            self.df[feature_null] = isnull_feature.astype(int)
            self.df[feature] = self.df[feature].fillna(fillna_value)
            return (feature, feature_null)
        else:
            return (feature,)

    def _preprocess_categ_params(self, feature: str):
        self.df[feature] = self.df[feature].fillna("?").astype(str)
        return feature

    def _preprocess_fk_params(self):
        for fk in self.foreign_key_names:
            fk_columns = self.foreign_keys_mapping.get(fk).get("columns")
            for fk_column in fk_columns:
                fk_column = self.df[fk_column]
                if fk_column.dtype != "object":
                    kde = gaussian_kde(fk_column)
                    with open(self.fk_kde_path, "wb") as file:
                        dill.dump(kde, file)
                    logger.info(f"KDE artifacts saved to {self.fk_kde_path}")
            yield fk, fk_columns

    def __drop_fk_columns(self, *args: Set[str]) -> object:
        """
        Drop columns which defined as foreign key
        """
        float_columns, int_columns, str_columns, categ_columns = args
        for fk, fk_columns in self._preprocess_fk_params():
            for fk_column in fk_columns:
                self.df = self.df.drop(fk_column, axis=1)
                float_columns.discard(fk_column)
                int_columns.discard(fk_column)
                str_columns.discard(fk_column)
                categ_columns.discard(fk_column)
                logger.debug(f"The column - {fk_column} of foreign key {fk} dropped from training "
                             f"and will be sampled from the PK table")
        return float_columns, int_columns, str_columns, categ_columns

    def _assign_char_feature(self, str_columns):
        """
        Assign text based feature to text columns
        """
        for feature in str_columns:
            max_len, rnn_units = self._preprocess_str_params(feature)
            self.assign_feature(
                CharBasedTextFeature(
                    feature, text_max_len=max_len, rnn_units=rnn_units
                ),
                feature,
            )
            logger.debug(f"Feature {feature} assigned as text based feature")

    def _assign_float_feature(self, float_columns):
        """
        Assign float based feature to float columns
        """
        # num_bins = self.find_clusters(df, float_columns)
        for feature in float_columns:
            features = self._preprocess_float_params(
                feature, fillna_strategy="mean"
            )
            if len(features) == 2:
                self.null_num_column_names.append(features[1])
            for feature in features:
                self.assign_feature(
                    ContinuousFeature(feature, column_type=float), feature
                )
                logger.debug(f"Feature {feature} assigned as float based feature")

    def _assign_int_feature(self, int_columns):
        """
        Assign int based feature to int columns
        """
        # num_bins = self.find_clusters(df, int_columns)
        for feature in int_columns:
            features = self._preprocess_float_params(feature, fillna_strategy="mean")
            if len(features) == 2:
                self.null_num_column_names.append(features[1])
            for feature in features:
                self.assign_feature(
                    ContinuousFeature(feature, column_type=int), feature
                )
                logger.debug(f"Feature {feature} assigned as int based feature")

    def _assign_categ_feature(self, categ_columns):
        """
        Assign categorical based feature to categorical columns
        """
        for feature in categ_columns:
            feature = self._preprocess_categ_params(feature)
            self.assign_feature(CategoricalFeature(feature), feature)
            logger.debug(f"Feature {feature} assigned as categorical based feature")

    def _assign_date_feature(self, date_columns):
        """
        Assign date feature to date columns
        """
        for feature in date_columns:
            self.assign_feature(DateFeature(feature), feature)
            logger.debug(f"Feature {feature} assigned as date feature")

    def _assign_binary_feature(self, binary_columns):
        """
        Assign binary feature to binary columns
        """
        for feature in binary_columns:
            feature = self._preprocess_categ_params(feature)
            self.assign_feature(BinaryFeature(feature), feature)
            logger.debug(f"Feature {feature} assigned as binary feature")

    def pipeline(self) -> pd.DataFrame:
        columns_nan_labels = get_nan_labels(self.df)
        self.df = nan_labels_to_float(self.df, columns_nan_labels)
        (
            str_columns,
            float_columns,
            categ_columns,
            date_columns,
            int_columns,
            binary_columns,
        ) = data_pipeline(self.df)

        if self.foreign_key_names:
            float_columns, int_columns, str_columns, categ_columns = self.__drop_fk_columns(float_columns,
                                                                                            int_columns,
                                                                                            str_columns,
                                                                                            categ_columns)
        if len(str_columns) > 0:
            self._assign_char_feature(str_columns)

        if len(float_columns) > 0:
            self._assign_float_feature(float_columns)

        if len(int_columns) > 0:
            self._assign_int_feature(int_columns)

        if len(categ_columns) > 0:
            self._assign_categ_feature(categ_columns)

        if len(date_columns) > 0:
            self._assign_date_feature(date_columns)

        if len(binary_columns) > 0:
            self._assign_binary_feature(binary_columns)

        self.set_nan_params(columns_nan_labels)

        self.fit(self.df)

        return self.df
