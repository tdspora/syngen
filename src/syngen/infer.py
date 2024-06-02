import os
from typing import Optional
import traceback

import click
from loguru import logger

from syngen.ml.worker import Worker
from syngen.ml.utils import (
    setup_logger,
    set_log_path,
    check_if_logs_available
)


@click.command()
@click.option("--metadata_path", type=str, default=None, help="Path to the metadata file")
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
    "--print_report",
    default=False,
    type=click.BOOL,
    help="Whether to print quality report. Might require significant time "
    "for big generated tables (>1000 rows). If absent, it's defaulted to False",
)
@click.option(
    "--log_level",
    default="INFO",
    type=click.Choice(["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level which will be used in the process. "
         "If absent, it's defaulted to 'INFO'",
)
def launch_infer(
    metadata_path: Optional[str],
    size: Optional[int],
    table_name: Optional[str],
    run_parallel: bool,
    batch_size: Optional[int],
    print_report: bool,
    random_seed: Optional[int],
    log_level: str,
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
    print_report
    random_seed
    log_level
    -------

    """
    os.environ["LOGURU_LEVEL"] = log_level
    set_log_path(type_of_process="infer", table_name=table_name, metadata_path=metadata_path)
    setup_logger()
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
        "print_report": print_report,
        "random_seed": random_seed,
        "get_infer_metrics": False
    }
    worker = Worker(
        table_name=table_name,
        metadata_path=metadata_path,
        settings=settings,
        log_level=log_level,
        type_of_process="infer",
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
