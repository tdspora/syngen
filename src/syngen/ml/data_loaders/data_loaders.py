from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import pandavro as pdx
import yaml
from yaml import Loader
from loguru import logger

from syngen.ml.validation_schema import validate_schema, configuration_schema


class BaseDataLoader(ABC):
    """
    Abstract class for data loader
    """

    @abstractmethod
    def load_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        pass


class DataLoader(BaseDataLoader):
    """
    Base class for loading and saving data either in csv or in avro format
    """

    def __init__(self, path: str):
        if not path:
            raise ValueError("It seems that the information of source is absent.")
        self.path = path
        self.file_loader = self.__get_file_loader()

    def __get_file_loader(self):
        path = Path(self.path)
        if path.suffix == '.avro':
            return AvroLoader()
        elif path.suffix == '.csv':
            return CSVLoader()
        else:
            raise NotImplementedError("File format not supported")

    def load_data(self) -> pd.DataFrame:
        df = self.file_loader.load_data(self.path)
        if df.shape[0] < 1:
            raise ValueError("Empty file was provided. Unable to train.")
        return df

    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        self.file_loader.save_data(path, df)


class CSVLoader(BaseDataLoader):
    """
    Class for loading and saving data in csv format
    """

    @staticmethod
    def _load_data(path, **kwargs) -> pd.DataFrame:
        df = pd.read_csv(path, **kwargs).iloc[:, :]
        df.columns = df.columns.str.replace(':', '')
        return df

    def load_data(self, path, **kwargs):
        return self._load_data(path, **kwargs)

    @staticmethod
    def _save_data(path: Optional[str], df: pd.DataFrame, **kwargs):
        if df is not None:
            df.to_csv(path, **kwargs)

    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        self._save_data(path, df, **kwargs)


class AvroLoader(BaseDataLoader):
    """
    Class for loading and saving data in avro format
    """

    def load_data(self, path, **kwargs) -> pd.DataFrame:
        return pdx.from_avro(path, **kwargs)

    @staticmethod
    def save_data(path: str, df: pd.DataFrame, **kwargs):
        if df is not None:
            pdx.to_avro(path, df, **kwargs)


class MetadataLoader(BaseDataLoader):
    """
    Metadata class for loading and saving metadata in YAML format
    """
    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        self.metadata_loader = self.get_metadata_loader()

    def get_metadata_loader(self):
        if self.metadata_path is not None:
            path = Path(self.metadata_path)
            if path.suffix in ['.yaml', '.yml']:
                return YAMLLoader()
            else:
                raise NotImplementedError("File format not supported")

    def load_data(self) -> dict:
        return self.metadata_loader.load_data(self.metadata_path)

    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        self.metadata_loader.save_data(path, df, **kwargs)


class YAMLLoader(BaseDataLoader):
    """
    Class for loading and saving data in YAML format
    """
    def load_data(self, metadata_path: str) -> dict:
        with open(metadata_path, "r") as metadata_file:
            metadata = yaml.load(metadata_file, Loader=Loader)
            validate_schema(configuration_schema, metadata)
        return metadata

    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        raise NotImplementedError("Saving YAML files is not supported")
