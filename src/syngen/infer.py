from typing import Optional, List, Union, Tuple

import click
from loguru import logger

from syngen.ml.worker import Worker
from syngen.ml.utils import (
    setup_log_process,
    get_reports,
    fetch_env_variables
)
from syngen.ml.validation_schema import ReportTypes


def validate_required_parameters(
        metadata_path: Optional[str] = None,
        table_name: Optional[str] = None
):
    if not metadata_path and not table_name:
        raise AttributeError(
            "It seems that the information of 'metadata_path' or 'table_name' is absent. "
            "Please provide either the information of 'metadata_path' or the information "
            "of 'table_name'."
        )
    if metadata_path and table_name:
        logger.warning(
            "The information of 'metadata_path' was provided. "
            "In this case the information of 'table_name' will be ignored."
        )


def launch_infer(
    metadata_path: Optional[str] = None,
    size: Optional[int] = 100,
    table_name: Optional[str] = None,
    run_parallel: bool = False,
    batch_size: Optional[int] = None,
    reports: Union[List[str], Tuple[str], str] = "none",
    random_seed: Optional[int] = None,
    log_level: str = "INFO",
    fernet_key: Optional[str] = None
):
    setup_log_process(
        type_of_process="infer",
        log_level=log_level,
        table_name=table_name,
        metadata_path=metadata_path
    )

    encryption_settings = fetch_env_variables({"fernet_key": fernet_key})

    worker = Worker(
        table_name=table_name,
        metadata_path=metadata_path,
        settings={
            "size": size,
            "run_parallel": run_parallel,
            "batch_size": batch_size,
            "reports": get_reports(
                value=reports,
                report_types=ReportTypes(),
                type_of_process="infer"
            ),
            "random_seed": random_seed
        },
        log_level=log_level,
        type_of_process="infer",
        encryption_settings=encryption_settings
    )

    logger.info(
        "The inference process will be executed according to the information mentioned "
        "in 'infer_settings' in the metadata file. If appropriate information is absent "
        "from the metadata file, then the values of parameters sent through CLI will be used. "
        "Otherwise, the values of parameters will be defaulted."
    )

    worker.launch_infer()


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
    help="The name of the environment variable that kept the value of the Fernet key "
         "to decrypt the sensitive data stored on the disk",
)
def cli_launch_infer(
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
    validate_required_parameters(metadata_path=metadata_path, table_name=table_name)
    launch_infer(
        metadata_path=metadata_path,
        size=size,
        table_name=table_name,
        run_parallel=run_parallel,
        batch_size=batch_size,
        reports=reports,
        random_seed=random_seed,
        log_level=log_level,
        fernet_key=fernet_key
    )


if __name__ == "__main__":
    cli_launch_infer()
