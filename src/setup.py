from setuptools import setup, find_packages

with open("../src/syngen/VERSION", "r") as file:
    version_info = file.read()
setup(name="syngen", version=version_info, packages=find_packages())
