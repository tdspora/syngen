import os
from typing import Optional

import click
from loguru import logger
from datetime import datetime

from syngen.ml.worker import Worker
from syngen.ml.utils import setup_logger, create_log_file
from syngen.ml.mlflow.mlflow_tracker import MlflowTracker


@click.command()
@click.option("--metadata_path", type=str, default=None, help="Path to the metadata file")
@click.option("--size", default=100, type=click.IntRange(1),
              help="Desired number of rows to generate. If absent, it's defaulted to 100")
@click.option("--table_name", default=None, type=str, help="Name of the table, same as in training")
@click.option("--run_parallel", default=False, type=click.BOOL,
              help="The flag which set whether to use multiprocessing (feasible for tables > 5000 rows)."
                   "If absent, it's defaulted to False")
@click.option("--batch_size", default=None, type=click.IntRange(1),
              help="If specified, the generation is split into batches. This can save the RAM")
@click.option("--random_seed", default=None, type=click.IntRange(0),
              help="Set any int in case you want reproducible results. To reproduce generated data again, "
                   "use the same int in this command.")
@click.option("--print_report", default=False, type=click.BOOL,
              help="Whether to print quality report. Might require significant time "
                   "for big generated tables (>1000 rows). If absent, it's defaulted to False")
@click.option("--log_level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
              help="Set the logging level which will be used in the process. If absent, it's defaulted to 'INFO'")
def launch_infer(
        metadata_path: Optional[str],
        size: Optional[int],
        table_name: Optional[str],
        run_parallel: bool,
        batch_size: Optional[int],
        print_report: bool,
        random_seed: Optional[int],
        log_level: str
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
    create_log_file(type_of_process="infer", table_name=table_name, metadata_path=metadata_path)
    setup_logger()
    if not metadata_path and not table_name:
        raise AttributeError("It seems that the information of 'metadata_path' or 'table_name' is absent. "
                             "Please provide either the information of 'metadata_path' or the information "
                             "of 'table_name'")
    if metadata_path and table_name:
        logger.warning("The information of 'metadata_path' was provided. "
                       "In this case the information of 'table_name' will be ignored")
        table_name = None
    logger.info(
        "The inference process will be executed according to the information mentioned in 'infer_settings' "
        "in the metadata file. If appropriate information is absent from the metadata file, then the values "
        "of parameters sent through CLI will be used. Otherwise, the values of parameters will be defaulted"
    )
    settings = {
        "size": size,
        "run_parallel": run_parallel,
        "batch_size": batch_size,
        "print_report": print_report,
        "random_seed": random_seed
    }
    worker = Worker(
        table_name=table_name,
        metadata_path=metadata_path,
        settings=settings,
        log_level=log_level,
        type="infer"
    )
    mlflow_exp_name = f"exp at {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}" if table_name else \
        f"{metadata_path.split('/')[-1].split('.')[0]}_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}"
    set_mlflow(mlflow_exp_name)
    worker.launch_infer()

def set_mlflow(exp_name: str = None):
    try:
        response = os.system("ping -c 1 " + os.environ.get("MLFLOW_TRACKING_URI")[7:-6])
        if response == 0:
            tracker = MlflowTracker(exp_name, True)
        else:
            tracker = MlflowTracker(exp_name, False)
            logger.warning("MLFlow server is not reachable, so the tracking will not be performed")
        tracker.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        tracker.set_experiment(exp_name)
    except Exception as e:
        logger.warning(f"MLFlow server is not reachable. {e}")


if __name__ == "__main__":
    launch_infer()
