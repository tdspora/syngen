from typing import List
import base64

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from loguru import logger


def encode_categorical_features(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    encoder = OrdinalEncoder()
    dfs = [df.fillna("?") for df in dfs]

    encoder.fit(pd.concat(dfs, axis=0))
    res_dfs = []
    for df in dfs:
        encoded_df = encoder.transform(df)
        res_dfs.append(encoded_df)
    return res_dfs


def series_count_words(x):
    return len(str(x).split())


def set_empty_string(value):
    if value is None:
        return ""
    return str(value)


def text_to_continuous(df, text_columns: List[str]):
    for col in text_columns:
        df[col + "_char_len"] = df[col].apply(set_empty_string, convert_dtype=False).apply(len)
        df[col + "_word_count"] = (
            df[col].apply(set_empty_string, convert_dtype=False).apply(series_count_words)
        )
    return df


def transform_to_base64(path):
    try:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return "data:image/svg+xml;base64," + encoded_string.decode("utf-8")
    except FileNotFoundError:
        logger.warning(f"No file found at {path}")
        return ""
