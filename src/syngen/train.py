import click
from loguru import logger

from syngen.ml.config import TrainConfig
from syngen.ml.data_loaders import DataLoader, MetadataLoader
from syngen.ml.strategies import TrainStrategy
from syngen.ml.train_chain import RootHandler, VaeTrainHandler
from syngen.ml.vae import VanillaVAEWrapper


def get_metadata(metadata_path, table_name):
    """
    Return metadata
    """
    if metadata_path:
        metadata = MetadataLoader().load_data(metadata_path)
        return metadata
    elif table_name:
        metadata = {"table_name": table_name}
        return metadata
    else:
        raise AttributeError("Either table name or path to metadata MUST be provided")


def set_handler(
        metadata: dict,
        paths: dict,
        wrapper_name: str,
        keys_mode: bool):
    """
    Set up the handler which used in training process
    Parameters
    ----------
    metadata
    paths
    wrapper_name
    keys_mode
    """

    root_handler = RootHandler(
        metadata=metadata,
        paths=paths
    )
    vae_handler = VaeTrainHandler(
        metadata=metadata,
        paths=paths,
        wrapper_name=wrapper_name,
        keys_mode=keys_mode
    )

    root_handler.set_next(vae_handler)
    return root_handler


def train(config: TrainConfig):
    """
    Launch the train strategy
    """
    data = DataLoader().load_data(config.path)

    metadata = get_metadata(config.metadata_path, config.table_name)

    paths = config.set_paths()

    handler = set_handler(
        metadata=metadata,
        paths=paths,
        wrapper_name=VanillaVAEWrapper.__name__,
        keys_mode=config.keys_mode
    )

    strategy = TrainStrategy(
        keys_mode=config.keys_mode,
        paths=paths,
        handler=handler
    )

    logger.info(f"Generator: {'vae'}, mode: {'train'}")

    strategy.run(
        data,
        epochs=config.epochs,
        row_subset=config.row_limit,
        batch_size=config.batch_size,
        dropna=config.dropna,
        keys_mode=config.keys_mode
    )


@click.command()
@click.argument("path")
@click.option("--epochs", default=10, help="Epochs.")
@click.option("--dropna", default=False, type=bool)
@click.option("--row_limit", default=None, type=int)
@click.option("--table_name", default=None, type=str)
@click.option("--metadata_path", default=None, type=str)
@click.option("--keys_mode", default=False, type=bool,
              help="When generating data for multiple tables use keys_mode = multable.")
def train_model(
    path: str,
    keys_mode: bool,
    epochs: int,
    dropna: bool,
    row_limit: int,
    table_name: str,
    metadata_path: str,
    batch_size: int = 32,
):
    """
    Launch the work of training process
    Parameters
    ----------
    path
    keys_mode
    epochs
    dropna
    row_limit
    table_name
    metadata_path
    batch_size
    -------

    """
    cli_config = TrainConfig(
        path=path,
        keys_mode=keys_mode,
        epochs=epochs,
        dropna=dropna,
        row_limit=row_limit,
        table_name=table_name,
        metadata_path=metadata_path,
        batch_size=batch_size
    )

    train(cli_config)


if __name__ == "__main__":
    train_model()
