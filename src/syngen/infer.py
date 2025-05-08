import os
from typing import Optional, List
import traceback

import click
from loguru import logger

from syngen.ml.worker import Worker
from syngen.ml.utils import (
    setup_log_process,
    check_if_logs_available,
    validate_parameter_reports
)
from syngen.ml.validation_schema import ReportTypes


validate_reports = validate_parameter_reports(
    report_types=ReportTypes().infer_report_types,
    full_list=ReportTypes().full_list_of_infer_report_types
)


@click.command()
@click.option(
    "--metadata_path",
    type=str,
    default=None,
    help="Path to the metadata file"
)
@click.option(
    "--size",
    default=100,
    type=click.IntRange(1),
    help="Desired number of rows to generate. If absent, it's defaulted to 100",
)
@click.option(
    "--table_name",
    default=None,
    type=str,
    help="Name of the table, same as in training",
)
@click.option(
    "--run_parallel",
    default=False,
    type=click.BOOL,
    help="The flag which set whether to use multiprocessing (feasible for tables > 5000 rows)."
    "If absent, it's defaulted to False",
)
@click.option(
    "--batch_size",
    default=None,
    type=click.IntRange(1),
    help="If specified, the generation is split into batches. This can save the RAM",
)
@click.option(
    "--random_seed",
    default=None,
    type=click.IntRange(0),
    help="Set any int in case you want reproducible results. To reproduce generated data again, "
    "use the same int in this command.",
)
@click.option(
    "--reports",
    default=("none",),
    type=click.UNPROCESSED,
    multiple=True,
    callback=validate_reports,
    help="Controls the generation of quality reports. "
    "Might require significant time for big generated tables (>10000 rows). "
    "If set to 'accuracy', generates an accuracy report. "
    "If set to 'metrics_only', outputs the metrics information "
    "only to standard output without generation of a report. "
    "If set to 'all', generates an accuracy report. "
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
    "--fernet_key",
    default=None,
    type=str,
    help="The value of the Fernet key to decrypt the sensitive data stored on the disk",
)
def launch_infer(
    metadata_path: Optional[str],
    size: Optional[int],
    table_name: Optional[str],
    run_parallel: bool,
    batch_size: Optional[int],
    reports: List[str],
    random_seed: Optional[int],
    log_level: str,
    fernet_key: Optional[str]
):
    """
    Launch the work of infer process
    Parameters
    ----------
    metadata_path
    size
    table_name
    run_parallel
    batch_size
    reports
    random_seed
    log_level
    fernet_key
    -------

    """
    setup_log_process(
        type_of_process="infer",
        log_level=log_level,
        table_name=table_name,
        metadata_path=metadata_path
    )
    if not metadata_path and not table_name:
        raise AttributeError(
            "It seems that the information of 'metadata_path' or 'table_name' is absent. "
            "Please provide either the information of 'metadata_path' or the information "
            "of 'table_name'"
        )
    if metadata_path and table_name:
        logger.warning(
            "The information of 'metadata_path' was provided. "
            "In this case the information of 'table_name' will be ignored"
        )
        table_name = None
    logger.info(
        "The inference process will be executed according to the information mentioned "
        "in 'infer_settings' in the metadata file. If appropriate information is absent "
        "from the metadata file, then the values of parameters sent through CLI will be used. "
        "Otherwise, the values of parameters will be defaulted"
    )
    settings = {
        "size": size,
        "run_parallel": run_parallel,
        "batch_size": batch_size,
        "reports": reports,
        "random_seed": random_seed
    }

    encryption_settings = {
        "fernet_key": fernet_key
    }

    worker = Worker(
        table_name=table_name,
        metadata_path=metadata_path,
        settings=settings,
        log_level=log_level,
        type_of_process="infer",
        encryption_settings=encryption_settings
    )

    worker.launch_infer()


if __name__ == "__main__":
    try:
        launch_infer()
    except Exception as e:
        log_file = os.getenv("SUCCESS_LOG_FILE")
        if not os.path.exists(log_file):
            logger.error(
                f"Generation failed on running stage. "
                f"The traceback of the error - {traceback.format_exc()}"
            )
        raise e
    finally:
        check_if_logs_available()
