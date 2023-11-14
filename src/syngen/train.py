from typing import Optional
import os

import click
from loguru import logger
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from syngen.ml.worker import Worker
from syngen.ml.utils import (
    setup_logger,
    create_log_file
)


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
    worker = Worker(
        table_name=table_name,
        metadata_path=metadata_path,
        settings=settings,
        log_level=log_level,
        type_of_process="train",
    )

    worker.launch_train()


def preprocess_data():
    """
    Preprocess the data before the training process
    """
    path_to_script = f"{os.getcwd()}/model_artifacts/script.py"
    if os.path.exists(path_to_script):
        os.system(f"python3 {path_to_script}")


if __name__ == "__main__":
    preprocess_data()
    launch_train()
