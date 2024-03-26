from setuptools import setup, find_packages

from syngen import __version__


setup(
    name="syngen",
    version=__version__,
    packages=find_packages(),
    include_package_data=True
)
