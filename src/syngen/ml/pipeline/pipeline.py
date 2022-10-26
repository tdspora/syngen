from typing import List, Tuple
from dateutil.parser import parse

import pandas as pd
import numpy as np
from loguru import logger


def get_date_columns(df: pd.DataFrame, str_columns: List[str]):
    # TODO: extend pattern to more formats
    # pattern = r'\d{2}(\.|/|\-)\d{2}(\.|/|\-)(\d{2}|\d{4})'
    # pattern = r"\s{0,1}\d+[-/\\:]\s{0,1}\d+[-/\\:]\s{0,1}\d+"

    def date_finder(x, fuzzy=False):
        x_wo_na = x.dropna()
        count = 0
        for x in x_wo_na.values:
            try:
                parse(x, fuzzy=fuzzy)
                count += 1
            except ValueError:
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
    object_columns = df.select_dtypes(include=["object"]).columns
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


def data_pipeline(df: pd.DataFrame, check_object_on_float: bool = False) -> Tuple:
    if check_object_on_float:
        columns_nan_labels = get_nan_labels(df)
        df = nan_labels_to_float(df, columns_nan_labels)

    binary_columns = set([col for col in df.columns if df[col].fillna("?").nunique() == 2])
    categ_columns = set([col for col in df.columns if df[col].dropna().nunique() <= 50 and col not in binary_columns])
    tmp_df = get_tmp_df(df)
    float_columns = set(tmp_df.select_dtypes(include=["float", "float64"]).columns)
    int_columns = set(tmp_df.select_dtypes(include=["int", "int64"]).columns)

    float_to_int_cols = set()
    for col in float_columns:
        if all(x.is_integer() for x in tmp_df[col]):
            float_to_int_cols.add(col)

    int_columns = (int_columns | float_to_int_cols) - (categ_columns | binary_columns)
    float_columns = float_columns - categ_columns - int_columns - binary_columns
    str_columns = set(tmp_df.columns) - float_columns - categ_columns - int_columns - binary_columns
    date_columns = get_date_columns(tmp_df, list(str_columns))
    str_columns -= date_columns

    assert len(str_columns) + \
           len(float_columns) + \
           len(int_columns) + \
           len(date_columns) + \
           len(categ_columns) + \
           len(binary_columns) == len(df.columns), "According to number of columns with defined types, " \
                                                   "column types are not identified correctly."

    logger.debug(
        f"Count of string columns: {len(str_columns)}; "
        + f"Count of float columns: {len(float_columns)}; "
        + f"Count of int columns: {len(int_columns)}; "
        + f"Count of categorical columns: {len(categ_columns)}; "
        + f"Count of date columns: {len(date_columns)}; "
        + f"Count of binary columns: {len(binary_columns)}"
    )
    return str_columns, float_columns, categ_columns, date_columns, int_columns, binary_columns


def fillnan(df, str_columns, float_columns, categ_columns):
    for c in str_columns | categ_columns:
        df[c] = df[c].fillna("NaN")

    return df
