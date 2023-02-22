from typing import Optional

import click
from loguru import logger

from syngen.ml.worker import Worker


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
def launch_infer(
        metadata_path: Optional[str],
        size: Optional[int],
        table_name: Optional[str],
        run_parallel: bool,
        batch_size: Optional[int],
        print_report: bool,
        random_seed: Optional[int]
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
    -------

    """
    if not metadata_path and not table_name:
        raise AttributeError("It seems that the information of metadata_path or table_name is absent. "
                             "Please provide either the information of metadata_path or the information of table_name.")
    if metadata_path:
        if table_name:
            logger.warning("The information of metadata_path was provided. "
                           "In this case the information of table_name will be ignored.")
            table_name = None
        if not metadata_path.endswith(('.yaml', '.yml')):
            raise NotImplementedError("This format for metadata_path is not supported. "
                                      "Please provide metadata_path in '.yaml' or in '.yml' format")
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
        settings=settings
    )
    worker.launch_infer()


if __name__ == "__main__":
    launch_infer()
