import os
import traceback
from typing import Optional, List

import click
from loguru import logger
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from syngen.ml.worker import Worker
from syngen.ml.utils import (
    setup_log_process,
    check_if_logs_available,
    validate_parameter_reports
)
from syngen.ml.validation_schema import ReportTypes


validate_reports = validate_parameter_reports(
    report_types=ReportTypes().train_report_types,
    full_list=ReportTypes().full_list_of_train_report_types
)


@click.command()
@click.option(
    "--metadata_path",
    type=str,
    default=None,
    help="Path to the metadata file"
)
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
    "--reports",
    default=("none",),
    type=click.UNPROCESSED,
    multiple=True,
    callback=validate_reports,
    help="Controls the generation of quality reports. "
    "Might require significant time for big generated tables (>1000 rows). "
    "If set to 'sample', generates a sample report. "
    "If set to 'accuracy', generates an accuracy report. "
    "If set to 'metrics_only', outputs the metrics information only to standard output "
    "without generation of a report. "
    "If set to 'all', generates both accuracy and sample report. "
    "If it's absent or set to 'none', no reports are generated.",
)
@click.option(
    "--log_level",
    default="INFO",
    type=click.Choice(["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
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
@click.option(
    "--fernet_key",
    default=None,
    type=str,
    help="The value of the Fernet key to encrypt and decrypt "
         "the sensitive data stored on the disk",
)
def launch_train(
    metadata_path: Optional[str],
    source: Optional[str],
    table_name: Optional[str],
    epochs: int,
    drop_null: bool,
    row_limit: Optional[int],
    reports: List[str],
    log_level: str,
    batch_size: int,
    fernet_key: Optional[str]
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
    reports
    log_level
    batch_size
    fernet_key
    -------

    """
    setup_log_process(
        type_of_process="train",
        log_level=log_level,
        table_name=table_name,
        metadata_path=metadata_path
    )
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
        "reports": reports,
    }

    encryption_settings = {
        "fernet_key": fernet_key,
    }

    worker = Worker(
        table_name=table_name,
        metadata_path=metadata_path,
        settings=settings,
        log_level=log_level,
        type_of_process="train",
        encryption_settings=encryption_settings
    )

    worker.launch_train()


if __name__ == "__main__":
    try:
        launch_train()
    except Exception as e:
        log_file = os.getenv("SUCCESS_LOG_FILE")
        if not os.path.exists(log_file):
            logger.error(
                f"Training failed on running stage. "
                f"The traceback of the error - {traceback.format_exc()}"
            )
        raise e
    finally:
        check_if_logs_available()
