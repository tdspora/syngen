import os
import sys
import re
from typing import List, Dict, Optional, Union, Set, Callable, Literal
from dateutil import parser
from datetime import datetime, timedelta
import time

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

# IANA timezone names - "2023-07-02T10:18:44.000000 America/New_York"
# Zulu time (UTC) represented by 'Z' - "2023-07-02T10:18:44Z"
# Numeric offsets (like +02:00, -05:00, etc.) - "2023-07-02T10:18:44 +02:00"
# Timezone abbreviations (like EST, PST, GMT, etc.) - "2023-07-02T10:18:44 EST"
TIMEZONE_REGEX = re.compile(r"""
        (?P<iana_name>
            [A-Za-z_]+
            (?:/[A-Za-z0-9_/-]+)+
        )|
        (?P<offset_zulu>
            Z
        )|
        (?P<offset_numeric>
        [+-]                         # Match a '+' or '-' for timezone offset
        (?:
        \d{2}:\d{2}$                 # Matches "+HH:MM" or "-HH:MM"
        |
        \d{4}$                       # Matches "+HHMM" or "-HHMM"
        )
        )|
        (?P<tz_abbr>
            \b
            (?!AM\b|PM\b|YT\b)
            \b
            (?:[A-Z]{2,5})
            \b
        )
    """, re.VERBOSE | re.ASCII)


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
        dt = parser.parse(
            dt,
            dayfirst=is_format_first(date_format, "d"),
            yearfirst=is_format_first(date_format, "y")
        ).replace(tzinfo=None)
        delta = dt - datetime(1970, 1, 1)
        return delta.total_seconds()
    except parser._parser.ParserError as e:
        year = re.match(r"\d+", e.args[0][5:]).group(0)
        if int(year) > 9999:
            return MAX_ALLOWED_TIME_MS
        elif int(year) < 1:
            return MIN_ALLOWED_TIME_MS


def timestamp_to_datetime(timestamp: int, delta=False):
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


def fetch_timezone(date_string: str) -> Union[str, float]:
    """
    Attempts to find and extract a timezone string from a date string.

    Args:
        date_string: The input string that might contain a date and timezone.

    Returns:
        timezone_string:
        The matched timezone string (e.g., "America/New_York", "+05:30", "EST").
    """
    if not isinstance(date_string, str):
        return np.NaN

    match = TIMEZONE_REGEX.search(date_string)

    if match:
        if match.group("iana_name"):
            return match.group("iana_name")
        elif match.group("offset_zulu"):
            return match.group("offset_zulu")
        elif match.group("offset_numeric"):
            return match.group("offset_numeric")
        elif match.group("tz_abbr"):
            return match.group("tz_abbr")

    return np.NaN


def get_nan_labels(df: pd.DataFrame, excluded_columns: Set[str]) -> Dict:
    """
    Get labels that represent nan values in float/int columns
    """
    columns_nan_labels = {}
    object_columns = df.select_dtypes(include=[pd.StringDtype(), "object"]).columns
    columns = set(object_columns) - excluded_columns
    for column in columns:
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

    return columns_nan_labels


def nan_labels_to_float(
    df: pd.DataFrame,
    columns_nan_labels: dict,
    exclude_columns: set = set(),
    process="training"
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
            if process == "training":
                logger.info(
                    f"Column '{column}' contains unique "
                    f"non-numeric value: '{label}'. "
                    "It will be treated as null label "
                    "and replaced with nulls "
                    "during the training process."
                )

    return df_with_nan


def fillnan(df, str_columns, float_columns, categorical_columns):
    for c in str_columns | categorical_columns:
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
    forbidden_keys = ["credentials", "fernet_key"]
    for key, value in list(metadata.items()):
        if key in forbidden_keys:
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


def fetch_unique_root(table_name: Optional[str], metadata_path: Optional[str]):
    """
    Construct the unique constant substring for use in the name of the experiment and log file
    """
    unique_name = str()
    if table_name:
        unique_name = table_name
    if metadata_path:
        unique_name = os.path.basename(metadata_path)
    return slugify(unique_name)


def get_log_path(table_name: Optional[str], metadata_path: Optional[str], type_of_process: str):
    """
    Get the log path for storing the logs of main processes
    """
    logs_dir_name = "model_artifacts/system_store/logs"
    unique_name = fetch_unique_root(table_name, metadata_path)
    unique_name = f"{unique_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    file_name_without_extension = f"logs_{type_of_process}_{unique_name}"
    file_path = os.path.join(
        logs_dir_name, f"{slugify(file_name_without_extension)}.log"
    )
    return file_path


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
    path_to_logs = os.getenv("SUCCESS_LOG_FILE")
    if path_to_logs is not None:
        with open(path_to_logs, "a") as log_file:
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


def setup_log_process(
    type_of_process: Literal["train", "infer"],
    log_level: str,
    table_name: Optional[str],
    metadata_path: Optional[str]
):
    """
    Set up the logging process with the specified level
    """
    os.environ["LOGURU_LEVEL"] = log_level
    os.makedirs("model_artifacts/system_store/logs", exist_ok=True)
    os.environ["SUCCESS_LOG_FILE"] = get_log_path(
        table_name=table_name,
        metadata_path=metadata_path,
        type_of_process=type_of_process
    )
    setup_logger()


def get_initial_table_name(table_name) -> str:
    """"
    Get the initial table name without the suffix "_pk" or "_fk"
    """
    return re.sub(r"_pk$|_fk$", "", table_name)


def timing(func):
    """
    Decorator that logs the execution time of the function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.trace(
            f"Function '{func.__name__}' executed in "
            f"{elapsed_time:.2f} seconds."
        )
        return result
    return wrapper


def validate_parameter_reports(report_types: list, full_list: list) -> Callable:
    """
    Validate the values of the parameter 'reports'
    """
    def validator(ctx, param, value) -> List[str]:
        input_values = set(value)
        valid_values: List = ["none", "all"]
        valid_values.extend(report_types)

        if not input_values.issubset(set(valid_values)):
            raise ValueError(
                f"Invalid input: Acceptable values for the parameter '--reports' are "
                f"{', '.join(valid_values)}."
            )
        if "none" in input_values and "all" in input_values:
            raise ValueError(
                "Invalid input: The '--reports' parameter cannot be set to both 'none' and 'all'. "
                "Please provide only one of these options."
            )

        if "none" in input_values or "all" in input_values:
            if len(input_values) > 1:
                raise ValueError(
                    "Invalid input: When '--reports' option is set to 'none' or 'all', "
                    "no other values should be provided."
                )
            if value[0] == "all":
                return full_list
            if value[0] == "none":
                return list()

        return list(input_values)
    return validator


class ValidationError(Exception):
    """
    The exception class for handling validation errors
    """
    def __init__(
        self,
        message: str
    ):
        super().__init__(message)
        self.message = message
