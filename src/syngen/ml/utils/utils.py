import os
import sys
import re
from typing import List, Dict, Optional, Union
from dateutil import parser
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from slugify import slugify
import pickle as pkl
import uuid
from ulid import ULID
import random
from loguru import logger

MAX_ALLOWED_TIME_MS = 253402214400
MIN_ALLOWED_TIME_MS = -62135596800


class ProgressBarHandler:
    """
    Singleton class for handling the progress bar
    """
    _delta = None
    _progress = 0
    _message = None

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(ProgressBarHandler, cls).__new__(cls)
            cls.instance._progress = 0
            cls.instance._delta = None
            cls.instance._message = None
        return cls.instance

    @property
    def progress(self):
        """
        Get the current progress of the process
        """
        return self._progress

    @property
    def delta(self):
        """
        Get the delta of the progress
        inside which the progress should be changed
        """
        return self._delta

    @property
    def info(self):
        """
        Get the information about the current progress
        and the log message described the status of the process
        """
        return self._progress, self._message

    def set_progress(self, progress=None, delta=None, message=None):
        self._delta = delta if delta is not None else self._delta
        self._progress = self._progress if progress is None else progress
        self._progress = self._progress if self._progress <= 1.0 else 1.0
        self._message = message

    @classmethod
    def reset_instance(cls):
        cls.instance._progress = 0
        cls.instance._delta = None
        cls.instance._message = None


def is_format_first(date_format: str, format_type: str) -> bool:
    """
    Check if the date format starts with the specified string
    """
    return date_format.lower().startswith(f"%{format_type}")


def datetime_to_timestamp(dt, date_format):
    """
    Convert datetime to timestamp
    """
    if pd.isnull(dt):
        return np.nan
    try:
        dt = parser.parse(dt,
                          dayfirst=is_format_first(date_format, "d"),
                          yearfirst=is_format_first(date_format, "y")).replace(tzinfo=None)
        delta = dt - datetime(1970, 1, 1)
        return delta.total_seconds()
    except parser._parser.ParserError as e:
        year = re.match(r"\d+", e.args[0][5:]).group(0)
        if int(year) > 9999:
            return MAX_ALLOWED_TIME_MS
        elif int(year) < 1:
            return MIN_ALLOWED_TIME_MS


def timestamp_to_datetime(timestamp, delta=False):
    """
    Convert the timestamp to the datetime object or timedelta object
    """
    # Calculate the number of seconds in the UNIX epoch and the number of seconds left
    if pd.isnull(timestamp):
        return np.nan

    if timestamp >= MAX_ALLOWED_TIME_MS:
        return datetime(9999, 12, 31, 23, 59, 59, 999999)
    elif timestamp <= MIN_ALLOWED_TIME_MS:
        return datetime(1, 1, 1, 0, 0, 0, 0)

    seconds_since_epoch = int(timestamp)
    remaining_seconds = timestamp - seconds_since_epoch

    # Calculate the datetime for the UNIX epoch (January 1, 1970)
    epoch_datetime = datetime(1970, 1, 1)

    # Calculate the timedelta for the number of seconds in the UNIX epoch
    delta_of_time = timedelta(seconds=seconds_since_epoch) + timedelta(seconds=remaining_seconds)

    if delta:
        return delta_of_time

    else:
        return epoch_datetime + delta_of_time


def generate_uuids(version: Union[int, str], size: int):
    ulid = ULID()
    generated_uuid_column = []
    for i in range(size):
        if version != "ulid":
            generated_uuid_column.append(
                str(uuid.UUID(int=random.getrandbits(128), version=int(version)))
            )
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
                datetime_object = parser.parse(x, default=date_for_check)

                # Check if the parsed date contains only the time component.
                # If it does, then skip it.
                count += 1 if datetime_object.date() != date_for_check.date() else 0
            except (ValueError, OverflowError):
                continue

        if count == len(x_wo_na):
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
    object_columns = df.select_dtypes(
        include=[pd.StringDtype(), "object"]).columns
    for column in object_columns:
        if df[column].isna().sum() > 0:
            continue
        str_values = []
        float_val = None
        for val in df[column].unique():
            try:
                float_val = float(val)
            except (TypeError, ValueError):
                str_values.append(val)
        if (float_val is not None) and (not np.isnan(float_val)) and len(str_values) == 1:
            nan_label = str_values[0]
            columns_nan_labels[column] = nan_label
        elif (float_val is not None) and (not np.isnan(float_val)) and not str_values:
            columns_nan_labels[column] = None

    return columns_nan_labels


def nan_labels_to_float(
    df: pd.DataFrame,
    columns_nan_labels: dict,
    exclude_columns: set = set()
) -> pd.DataFrame:
    """
    Replace str nan labels in float/int columns with actual np.NaN
    and casting the column to float type.
    """
    df_with_nan = df.copy()
    for column, label in columns_nan_labels.items():
        if column not in exclude_columns:
            df_with_nan[column].replace(label, np.NaN, inplace=True)
            df_with_nan[column] = df_with_nan[column].astype(float)

            logger.info(
                f"Column '{column}' contains unique "
                f"non-numeric value: '{label}'. "
                "It will be treated as null label "
                "and replaced with nulls."
            )

    return df_with_nan


def fillnan(df, str_columns, float_columns, categ_columns):
    for c in str_columns | categ_columns:
        df[c] = df[c].fillna("NaN")

    return df


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


def slugify_parameters(exclude_params=(), turn_on=True):
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
                    updated_kwargs[key] = slugify(value, lowercase=turn_on)
            return function(**updated_kwargs)

        return inner_wrapper

    return wrapper


def inverse_dict(dictionary: Dict) -> Dict:
    """
    Swap keys and values in the dictionary
    """
    return dict(zip(dictionary.values(), dictionary.keys()))


def clean_up_metadata(metadata: Dict):
    """
    Clean up the metadata,
    remove the sensitive information (credentials to the remote storage) from the metadata
    """
    for key, value in list(metadata.items()):
        if key == "credentials":
            del metadata[key]
        elif isinstance(value, dict):
            clean_up_metadata(value)
        else:
            continue
    return metadata


def trim_string(col):
    if isinstance(col.dtype, str):
        return col.str.slice(stop=10 * 1024)
    else:
        return col


def check_if_features_assigned(dataset_pickle_path: str):
    """
    Check if features are assigned in the dataset
    """
    features = fetch_config(dataset_pickle_path).features
    if len(features) == 0:
        logger.info("No features to train VAE on")
        return False
    return True


def fetch_config(config_pickle_path: str):
    """
    Fetch the configuration from the disk
    """
    with open(config_pickle_path, "rb") as f:
        return pkl.load(f)


def fetch_unique_root(table_name: str, metadata_path: str):
    """
    Construct the unique constant substring for use in the name of the experiment and log file
    """
    unique_name = str()
    if table_name:
        unique_name = table_name
    if metadata_path:
        unique_name = os.path.basename(metadata_path)
    return slugify(unique_name)


def set_log_path(type_of_process: str, table_name: Optional[str], metadata_path: Optional[str]):
    """
    Set the log path for storing the logs of main processes
    """
    logs_dir_name = "model_artifacts/tmp_store/logs"
    os.makedirs(logs_dir_name, exist_ok=True)
    unique_name = fetch_unique_root(table_name, metadata_path)
    unique_name = f"{unique_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    file_name_without_extension = f"logs_{type_of_process}_{unique_name}"
    file_path = os.path.join(
        logs_dir_name, f"{slugify(file_name_without_extension)}.log"
    )
    os.environ["SUCCESS_LOG_FILE"] = file_path


def fetch_log_message(message):
    """
    Fetch the log message
    """
    record = message.record
    log_message = (f'{record["time"]} | {record["level"]} | '
                   f'{record["file"]}:{record["function"]}:{record["line"]} - {record["message"]}')
    return log_message


def file_sink(message):
    """
    Save logs to the log file
    """
    with open(os.getenv("SUCCESS_LOG_FILE"), "a") as log_file:
        log_message = fetch_log_message(message)
        log_file.write(log_message + "\n")


def console_sink(record):
    """
    Redirect logs to the console
    """
    sys.stderr.write(record)


def setup_logger():
    """
    Set up loggers with the specified level
    """
    logger.remove()
    logger.add(console_sink, colorize=True, level=os.getenv("LOGURU_LEVEL"))
    logger.add(file_sink, level=os.getenv("LOGURU_LEVEL"))


def check_if_logs_available():
    """
    Check if the logs are available and
    write a message to the log file if not
    """
    path_to_logs = os.getenv("SUCCESS_LOG_FILE")
    if not os.path.exists(path_to_logs):
        with open(path_to_logs, "a") as file:
            file.write("No logs available\n")


def get_initial_table_name(table_name) -> str:
    """"
    Get the initial table name without the suffix "_pk" or "_fk"
    """
    return re.sub(r"_pk$|_fk$", "", table_name)
