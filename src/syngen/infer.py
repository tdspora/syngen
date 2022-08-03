import click
from typing import Optional

from syngen.ml.config import InferConfig
from syngen.ml.reporters import Report, AccuracyReporter
from syngen.ml.strategies import InferStrategy
from syngen.ml.train_chain import VaeInferHandler
from syngen.ml.data_loaders import MetadataLoader
from syngen.ml.vae import VanillaVAEWrapper


def get_metadata(config):
    if config.metadata_path:
        metadata = MetadataLoader().load_data(config.metadata_path)
        return metadata
    elif config.table_name:
        metadata = {"table_name": config.table_name}
        return metadata
    else:
        raise AttributeError("Either table name or path to metadata MUST be provided")


def set_handler(config):
    return VaeInferHandler(
        metadata=get_metadata(config),
        paths=config.set_paths(),
        wrapper_name=VanillaVAEWrapper.__name__,
        random_seed=config.random_seed,
        keys_mode=config.keys_mode
    )


def set_reporters(config):
    """
    Set up reporter which used in order to create the report during infer process
    """
    accuracy_reporter = AccuracyReporter(
        metadata={"table_name": config.table_name},
        paths=config.set_paths()
    )
    Report().register_reporter(accuracy_reporter)


def infer(config: InferConfig):
    """
    Launch the infer strategy
    """
    set_handler(config)

    set_reporters(config)

    infer_strategy = InferStrategy(
        size=config.size,
        run_parallel=config.run_parallel,
        keys_mode=config.keys_mode,
        metadata_path=config.metadata_path,
        print_report=config.print_report,
        batch_size=config.batch_size,
        handler=set_handler(config)
    )

    infer_strategy.run()


@click.command()
@click.argument("size", type=int)
@click.argument("table_name", type=str)
@click.option("--run_parallel", default=True, type=bool)
@click.option("--batch_size", default=None, type=int)
@click.option("--keys_mode", default=None, type=str,
              help ="When generating data for multiple tables use keys_mode = multable.")
@click.option("--metadata_path", default=None, type=str)
@click.option("--random_seed", default=None, type=int,
              help="Set any int in case you want reproducible results. To reproduce generated data again, "
                   "use the same int in this command.")
@click.option("--print_report", default=False, type=bool,
              help="Whether to print quality report. Might require significant time for big generated tables "
                   "(>1000 rows)")
def infer_model(
        size: int,
        table_name: str,
        run_parallel: bool,
        batch_size: Optional[int],
        keys_mode: Optional[bool],
        metadata_path: Optional[str],
        random_seed: Optional[int],
        print_report: bool):
    """
    Launch the work of infer process

    Parameters
    ----------
    size
    table_name
    run_parallel
    batch_size
    keys_mode
    metadata_path
    random_seed
    print_report
    """
    infer_config = InferConfig(
        size=size,
        table_name=table_name,
        run_parallel=run_parallel,
        batch_size=batch_size,
        keys_mode=keys_mode,
        metadata_path=metadata_path,
        random_seed=random_seed,
        print_report=print_report
    )

    infer(infer_config)


if __name__ == "__main__":
    infer_model()
