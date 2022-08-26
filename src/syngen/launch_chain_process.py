import click
from typing import Optional

from syngen.ml.yaml_handler import YAMLHandler


@click.command()
@click.option("--metadata_path", default=None, type=str)
def launch_chain_process(metadata_path: Optional[str]):
    if metadata_path is None:
        raise AttributeError("The metadata_path seems to be absent. Please provide metadata_path")
    if not metadata_path.endswith('.yaml'):
        raise NotImplementedError("This format for metadata_path is not supported. "
                                  "Please provide metadata_path in '.yaml' format")
    YAMLHandler(metadata_path=metadata_path).run()


if __name__ == "__main__":
    launch_chain_process()
