from typing import List, Dict
import base64
from collections import Counter
import re
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from loguru import logger

# Seed for the accuracy report's own "pick a subset to display" draws (which column
# pairs get a bivariate plot, which ticks are labelled, which categorical bars are
# shown). Drawing from the global RNG made the same data produce a different report on
# every run. Matches the random_state=10 already used by the estimators in
# metrics_classes.metrics, so all report randomness shares one seed (EPMCTDM-7630).
# The aim is run-to-run stability, not user control.
METRIC_SAMPLE_SEED = 10


def encode_categorical_features(dfs: List[pd.DataFrame]) -> List[np.ndarray]:
    """
    Encode categorical features in a list of DataFrames using Ordinal Encoder.

    Returns bare float arrays, not DataFrames - the column names are lost, so a
    caller that needs them should use 'encode_categories' instead. Codes start at 0
    and the encoder is fitted on the concatenation, so they are shared across the
    input frames.
    """
    encoder = OrdinalEncoder()
    dfs = [df.fillna("?") for df in dfs]

    encoder.fit(pd.concat(dfs, axis=0))
    encoded = []
    for df in dfs:
        encoded.append(encoder.transform(df))
    return encoded


def encode_categories(original: pd.Series, synthetic: pd.Series) -> Dict:
    """
    Build an integer code for every category present in either dataset.

    Sorted because set iteration order is randomized per process, which changed the
    reported numbers on every run; by 'str' because the union can mix types.

    Every metric encoding these columns must use this helper - the first one to run
    mutates the shared frames, so one unfixed site re-breaks all the metrics after it.
    """
    categories = sorted(set(original) | set(synthetic), key=str)
    return {category: i + 1 for i, category in enumerate(categories)}


def series_count_words(x):
    return len(str(x).split())


def set_empty_string(value):
    if value is None:
        return ""
    return str(value)


def text_to_continuous(df, text_columns: List[str]):
    """
    Add character length and word count columns to the DataFrame
    """
    for col in text_columns:
        df[col + "_char_len"] = df[col].apply(set_empty_string, convert_dtype=False).apply(len)
        df[col + "_word_count"] = (
            df[col].apply(set_empty_string, convert_dtype=False).apply(series_count_words)
        )
    return df


def transform_to_base64(path):
    """
    Encode the image to base64 format string
    """
    try:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return "data:image/svg+xml;base64," + encoded_string.decode("utf-8")
    except FileNotFoundError:
        logger.warning(f"No file found at {path}")
        return ""


def get_outlier_ratio_iqr(column: pd.Series, factor=1.5):
    """
    Get the ratio of outliers in a column using the IQR method
    """
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (column < (q1 - factor * iqr)) | (column > (q3 + factor * iqr))
    return outlier_mask.mean()


def get_ratio_counts(ratio_counts, count: int = 30) -> Dict:
    """
    Select the most common, least common and some random items between them
    from the ratio_counts
    """
    ratio_counts = Counter(ratio_counts)

    most_least_count = int(count * 0.2)
    other_items = count - 2 * most_least_count

    most_common_items = ratio_counts.most_common(most_least_count)
    least_common_items = ratio_counts.most_common()[: -most_least_count - 1: -1]
    between_items = ratio_counts.most_common()[most_least_count:-most_least_count]
    # Seeded so the same data always yields the same displayed subset: drawing from
    # the global RNG made the univariate plot differ between runs (EPMCTDM-7630).
    selected_between_items = random.Random(METRIC_SAMPLE_SEED).sample(
        between_items, min(other_items, len(between_items))
    )

    updated_ratio_counts = dict(
        most_common_items + selected_between_items + least_common_items
    )

    return updated_ratio_counts


def plot_dist(column_data, sort=True, full_set=None):
    """
    Plot the distribution of the column data
    """
    counts = Counter(column_data)
    if full_set is not None:
        absent_keys = full_set - set(counts.keys())
        if len(absent_keys) > 0:
            for k in absent_keys:
                counts[k] = 0

        if sort:
            sorted_keys = sorted(counts, key=lambda x: counts[x])
            counts = {i: counts[i] for i in sorted_keys}

        size = len(column_data)
        counts = {key: (x / size * 100) for key, x in counts.items()}
        return counts


def sanitize_labels(label):
    """
    Sanitize labels by removing dollar signs and carets
    """
    return re.sub(r"\$|\^", "", label)
