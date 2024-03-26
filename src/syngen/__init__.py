import os
import argparse

from syngen.train import preprocess_data  # noqa: F401


base_dir = os.path.dirname(__file__)
version_file = os.path.join(base_dir, "VERSION")

with open(version_file) as f:
    __version__ = f.read().strip()


def main():
    parser = argparse.ArgumentParser(prog="syngen")
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + __version__
    )
    args = parser.parse_args()
    return args
