import os
import sys
from typing import List, Dict
from dateutil.parser import parse
import pickle
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from slugify import slugify
import pickle as pkl
import uuid
from ulid import ULID
import random
from loguru import logger


def generate_uuids(version: int, size: int):
    ulid = ULID()
    generated_uuid_column = []
    for i in range(size):
        if version != "ulid":
            generated_uuid_column.append(uuid.UUID(int=random.getrandbits(128), version=int(version)))
        else:
            generated_uuid_column.append(ulid.generate())
    return generated_uuid_column


def generate_uuid(size, dataset, uuid_columns, synthetic_infer):
    uuid_columns_types = dataset.uuid_columns_types
    for col in uuid_columns:
        synthetic_infer[col] = generate_uuids(uuid_columns_types[col], size)
    return synthetic_infer


def get_date_columns(df: pd.DataFrame, str_columns: List[str]):
    def date_finder(x, fuzzy=False):
        x_wo_na = x.dropna()
        count = 0
        for x in x_wo_na.values:
            try:
                date_for_check = datetime(8557, 7, 20)
                datetime_object = parse(x, default=date_for_check)
                # Check if the parsed date contains only the time component. If it does, then skip it.
                count += 1 if datetime_object.date() != date_for_check.date() else 0
            except (ValueError, OverflowError):
                continue
        if count > len(x_wo_na) * 0.8:
            return 1
        else:
            return np.nan

    date_columns = df[str_columns].apply(date_finder).dropna()

    if isinstance(date_columns, pd.DataFrame):
        names = date_columns.columns
    elif isinstance(date_columns, pd.Series):
        names = date_columns.index
    else:
        names = []
    return set(names)


def get_nan_labels(df: pd.DataFrame) -> dict:
    """Get labels that represent nan values in float/int columns

    Args:
        df (pd.DataFrame): table data

    Returns:
        dict: dict that maps nan str label to column name
    """
    columns_nan_labels = {}
    object_columns = df.select_dtypes(include=[pd.StringDtype(), "object"]).columns
    for column in object_columns:
        str_values = []
        float_val = None
        for val in df[column].unique():
            try:
                float_val = float(val)
            except (TypeError, ValueError):
                str_values.append(val)
        if (
                (float_val is not None)
                and (not np.isnan(float_val))
                and len(str_values) == 1
        ):
            nan_label = str_values[0]
            columns_nan_labels[column] = nan_label

    return columns_nan_labels


def nan_labels_to_float(df: pd.DataFrame, columns_nan_labels: dict) -> pd.DataFrame:
    """Replace str nan labels in float/int columns with actual np.nan and casting the column to float type.

    Args:
        df (pd.DataFrame): table data

    Returns:
        pd.DataFrame: DataFrame with str NaN labels in float/int columns replaced with np.nan
    """
    df_with_nan = df.copy()
    for column, label in columns_nan_labels.items():
        df_with_nan[column] = pd.to_numeric(
            df_with_nan[column].where(df_with_nan[column] != label, np.nan)
        )  # casting from object to int/float
    return df_with_nan


def get_tmp_df(df):
    tmp_col_len_min = float("inf")
    tmp_cols = {}
    for col in df.columns:
        tmp_cols[col] = pd.Series(df[col].dropna().values)
        tmp_col_len = len(tmp_cols[col])
        if tmp_col_len < tmp_col_len_min:
            tmp_col_len_min = tmp_col_len
    return pd.DataFrame(tmp_cols).iloc[:tmp_col_len_min, :]


def fillnan(df, str_columns, float_columns, categ_columns):
    for c in str_columns | categ_columns:
        df[c] = df[c].fillna("NaN")

    return df


def fetch_dataset(dataset_pickle_path: str):
    """
    Deserialize and return the object of class Dataset
    """
    with open(dataset_pickle_path, "rb") as f:
        dataset = pickle.loads(f.read())
    # Check if the serialized class has associated dataframe and
    # drop it as it might contain sensitive data. Save columns from the dataframe for later use.
    if hasattr(dataset, "df"):
        dataset = update_dataset(dataset)
        dataset.order_of_columns = dataset.df.columns.tolist()
        del dataset.df
        with open(dataset_pickle_path, "wb") as f:
            f.write(pickle.dumps(dataset))
    return dataset


def define_existent_columns(columns, original_columns):
    existent_columns = []
    for column in columns:
        if column in original_columns:
            existent_columns.append(column)
        continue
    return existent_columns


def update_dataset(dataset):
    for attr in vars(dataset):
        if attr in ["primary_keys_mapping", "unique_keys_mapping", "foreign_keys_mapping"]:
            attr_value = getattr(dataset, attr)
            updated_attr_value = attr_value.copy()
            for key, config in attr_value.items():
                updated_columns = define_existent_columns(config.get("columns", []), dataset.df.columns)
                config["columns"] = updated_columns
                updated_attr_value[key] = config

            setattr(dataset, attr, updated_attr_value)
    return dataset


def slugify_attribute(**kwargs):
    """
    Slugify the value of the attribute of the instance
    and set it to the new attribute
    """
    def wrapper(function):
        def inner_wrapper(*args):
            object_, *other = args
            for attribute, new_attribute in kwargs.items():
                fetched_attribute = object_.__getattribute__(attribute)
                value_of_new_attribute = slugify(fetched_attribute)
                object_.__setattr__(new_attribute, value_of_new_attribute)
            return function(*args)
        return inner_wrapper
    return wrapper


def slugify_parameters(exclude_params=()):
    """
    Slugify the values of parameters, excluding specified parameters
    """
    def wrapper(function):
        def inner_wrapper(**kwargs):
            updated_kwargs = {}
            for key, value in kwargs.items():
                if key in exclude_params:
                    updated_kwargs[key] = value
                else:
                    updated_kwargs[key] = slugify(value)
            return function(**updated_kwargs)
        return inner_wrapper

    return wrapper


def inverse_dict(dictionary: Dict) -> Dict:
    """
    Swap keys and values in the dictionary
    """
    return dict(zip(dictionary.values(), dictionary.keys()))


def trim_string(col):
    if isinstance(col.dtype, str):
        return col.str.slice(stop=10 * 1024)
    else:
        return col


def convert_to_time(timestamp):
    """
    Convert timestamp to datetime
    """
    timestamp = int(timestamp * 1e-9)
    if timestamp < 0:
        return datetime(1970, 1, 1) + timedelta(seconds=timestamp)
    else:
        return datetime.utcfromtimestamp(timestamp)


def check_if_features_assigned(dataset_pickle_path: str):
    """
    Check if features are assigned in the dataset
    """
    features = fetch_dataset(dataset_pickle_path).features
    if len(features) == 0:
        logger.info("No features to train VAE on")
        return False
    return True


def fetch_training_config(train_config_pickle_path):
    """
    Fetch the parameters of the training configuration
    """
    with open(train_config_pickle_path, "rb") as f:
        return pkl.load(f)


def setup_logger():
    """
    Setup logger with the specified level
    """
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOGURU_LEVEL"))
