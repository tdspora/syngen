from setuptools import setup, find_packages

from syngen.ml.utils import fetch_version_syngen


setup(
    name="syngen",
    version=fetch_version_syngen(),
    packages=find_packages(),
    include_package_data=True
)
