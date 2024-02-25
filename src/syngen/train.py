from typing import Optional, List, Tuple, Dict
import os
import json
from json import JSONDecodeError
from collections import Counter
import numpy as np

import click
from loguru import logger
from flatten_json import flatten
import pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from syngen.ml.worker import Worker
from syngen.ml.utils import (
    setup_logger,
    create_log_file
)
from syngen.ml.data_loaders import MetadataLoader, DataLoader


@click.command()
@click.option("--metadata_path", type=str, default=None, help="Path to the metadata file")
@click.option(
    "--source",
    type=str,
    default=None,
    help="Path to the table that you want to use as a reference",
)
@click.option(
    "--table_name",
    type=str,
    default=None,
    help="Arbitrary string to name the directories",
)
@click.option(
    "--epochs",
    default=10,
    type=click.IntRange(1),
    help="Number of trained epochs. If absent, it's defaulted to 10",
)
@click.option(
    "--drop_null",
    default=False,
    type=click.BOOL,
    help="Flag which set whether to drop rows with at least one missing value. "
    "If absent, it's defaulted to False",
)
@click.option(
    "--row_limit",
    default=None,
    type=click.IntRange(1),
    help="Number of rows to train over. A number less than the original table "
         "length will randomly subset the specified rows number",
)
@click.option(
    "--print_report",
    default=False,
    type=click.BOOL,
    help="Whether to print quality report. Might require significant time "
    "for big generated tables (>1000 rows). If absent, it's defaulted to False",
)
@click.option(
    "--log_level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level which will be used in the process. "
         "If absent, it's defaulted to 'INFO'",
)
@click.option(
    "--batch_size",
    default=32,
    type=click.IntRange(1),
    help="Number of rows that goes in one batch. "
         "This parameter can help to control memory consumption.",
)
def launch_train(
    metadata_path: Optional[str],
    source: Optional[str],
    table_name: Optional[str],
    epochs: int,
    drop_null: bool,
    row_limit: Optional[int],
    print_report: bool,
    log_level: str,
    batch_size: int = 32,
):
    """
    Launch the work of training process

    Parameters
    ----------
    metadata_path
    source
    table_name
    epochs
    drop_null
    row_limit
    print_report
    log_level
    batch_size
    -------

    """
    os.environ["LOGURU_LEVEL"] = log_level
    create_log_file(type_of_process="train", table_name=table_name, metadata_path=metadata_path)
    setup_logger()
    if not metadata_path and not source and not table_name:
        raise AttributeError(
            "It seems that the information of 'metadata_path' or 'table_name' "
            "and 'source' is absent. Please provide either the information of "
            "'metadata_path' or the information of 'source' and 'table_name'"
        )
    elif metadata_path and table_name and source:
        logger.warning(
            "The information of 'metadata_path' was provided. "
            "In this case the information of 'table_name' and 'source' will be ignored"
        )
        table_name = None
    elif metadata_path and source:
        logger.warning(
            "The information of 'metadata_path' was provided. "
            "In this case the information of 'source' will be ignored"
        )
    elif metadata_path and table_name:
        logger.warning(
            "The information of 'metadata_path' was provided. "
            "In this case the information of 'table_name' will be ignored"
        )
        table_name = None
    elif source and not table_name:
        raise AttributeError(
            "It seems that the information of 'metadata_path' or 'table_name' is absent. "
            "Please provide either the information of 'metadata_path' or "
            "the information of 'source' and 'table_name'"
        )
    elif table_name and not source:
        raise AttributeError(
            "It seems that the information of 'metadata_path' or 'source' is absent. "
            "Please provide either the information of 'metadata_path' or "
            "the information of 'source' and 'table_name'"
        )
    logger.info(
        "The training process will be executed according to the information mentioned "
        "in 'train_settings' in the metadata file. If appropriate information is absent "
        "from the metadata file, then the values of parameters sent through CLI will be used. "
        "Otherwise, the values of parameters will be defaulted"
    )
    settings = {
        "source": source,
        "epochs": epochs,
        "drop_null": drop_null,
        "row_limit": row_limit,
        "batch_size": batch_size,
        "print_report": print_report,
    }

    if metadata_path:
        preprocess_data(metadata_path)

    worker = Worker(
        table_name=table_name,
        metadata_path=metadata_path,
        settings=settings,
        log_level=log_level,
        type_of_process="train",
    )

    worker.launch_train()


def get_json_columns(data: pd.DataFrame) -> List[str]:
    """
    Get the list of columns which contain JSON data
    """
    json_columns = list()
    for column in data.columns.to_list():
        try:
            if pd.isnull(data[column]).all():
                continue
            data[column].dropna().apply(lambda v: json.loads(v))
            json_columns.append(column)
        except (TypeError, JSONDecodeError):
            continue
    return json_columns


def get_flattened_df(data: pd.DataFrame, json_columns: List) -> Tuple[pd.DataFrame, Dict]:
    """
    Flatten the JSON columns in the dataframe
    """
    df_list = list()
    flattening_mapping = dict()
    for column in json_columns:
        flattened_data = pd.DataFrame(
            [
                    flatten(json.loads(i), ".")
                    for i in data[column]
            ]
        )
        flattening_mapping[column] = flattened_data.columns.to_list()
        df_list.append(flattened_data)
    flattened_data = pd.concat([data, *df_list], axis=1)
    flattened_data.drop(columns=flattening_mapping.keys(), inplace=True)
    flattened_df = flattened_data.T.loc[~flattened_data.T.index.duplicated(), :].T
    flattened_df = flattened_df.applymap(lambda x: np.NaN if x in [list(), dict()] else x)
    return flattened_df, flattening_mapping


def run_script():
    path_to_script = f"{os.getcwd()}/model_artifacts/script.py"
    if os.path.exists(path_to_script):
        os.system(f"python3 {path_to_script}")


def save_flatten_metadata(table_name: str, flattening_mapping: Dict, duplicated_columns: List):
    """
    Save the metadata of the flattening process
    """
    os.makedirs(f"{os.getcwd()}/model_artifacts", exist_ok=True)
    with open(f"{os.getcwd()}/model_artifacts/flatten_metadata.json", "a") as f:
        metadata = {
            table_name: {
                "flattening_mapping": flattening_mapping,
                "duplicated_columns": duplicated_columns
            }
        }
        json.dump(metadata, f)


def handle_json_columns(path_to_metadata: str):
    """
    Preprocess the data contained JSON columns before the training process
    """
    metadata = MetadataLoader(path_to_metadata).load_data()
    for table, settings in metadata.items():
        if table == "global":
            continue
        source = settings.get("train_settings", {}).get("source", "")
        data, schema = DataLoader(source).load_data()
        if json_columns := get_json_columns(data):
            logger.info(f"The table '{table}' contains JSON columns: {', '.join(json_columns)}")
            logger.info(f"Flattening the JSON columns in the table - '{table}'")
            flattened_data, flattening_mapping = get_flattened_df(data, json_columns)
            DataLoader(source).save_data(source, flattened_data)
            duplicated_columns = [
                key
                for key, value in dict(Counter(flattened_data.columns.to_list())).items()
                if value > 1
            ]
            save_flatten_metadata(table, flattening_mapping, duplicated_columns)
            logger.info(f"The table '{table}' has been successfully flattened")


def preprocess_data(path_to_metadata: str):
    """
    Preprocess the data before the training process
    """
    run_script()
    handle_json_columns(path_to_metadata)


if __name__ == "__main__":
    launch_train()
