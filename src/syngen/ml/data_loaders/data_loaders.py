from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple

import pandas as pd
from pandas.errors import ParserError
import pandavro as pdx
import yaml
from yaml import Loader
from avro.datafile import DataFileReader
from avro.io import DatumReader
from loguru import logger

from syngen.ml.validation_schema import validate_schema, configuration_schema
from syngen.ml.schema_convertor import AvroSchemaConvertor


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

    def load_data(self) -> Tuple[pd.DataFrame, Optional[Dict]]:
        df, schema = self.file_loader.load_data(self.path)
        if df.shape[0] < 1:
            raise ValueError("Empty file was provided. Unable to train.")
        return df, schema

    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        self.file_loader.save_data(path, df)


class CSVLoader(BaseDataLoader):
    """
    Class for loading and saving data in csv format
    """

    @staticmethod
    def _load_data(path, **kwargs) -> Tuple[pd.DataFrame, None]:
        df = pd.DataFrame()
        try:
            df = pd.read_csv(path, engine="python", **kwargs).iloc[:, :]
        except ParserError:
            df = pd.read_csv(path, engine="c", **kwargs).iloc[:, :]
        finally:
            df.columns = df.columns.str.replace(':', '')
            return df, None

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

    @staticmethod
    def _load_df(path) -> pd.DataFrame:
        """
        Load data in Avro format
        :param path: the path to the the file
        :return: dataframe
        """
        return pdx.from_avro(path)

    def load_data(self, path: str, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """
        Load data in Avro format from AWS S3, Azure Storage or locally

        :param
        path:
        str which should be should be in the next format for the connection to AWS S3: "s3://path/to/bucket"
        str which should be should be in the next format for the connection to Azure Storage: "azure://{container_name}/{blob_name}"
        """
        try:
            with open(path, 'rb') as f:
                df = self._load_df(f)
                schema = self._load_schema(f)
                return df, schema
        except FileNotFoundError as error:
            message = f"It seems that the path to the table isn't valid.\n" \
                      f"The details of the error - {error}.\n" \
                      f"Please, check the path to the table"
            logger.error(message)
            raise FileNotFoundError(message)

    @staticmethod
    def save_data(path: str, df: pd.DataFrame, **kwargs):
        if df is not None:
            pdx.to_avro(path, df, **kwargs)

    @staticmethod
    def _load_schema(f) -> Dict[str, str]:
        """
        Load schema of the metadata of the table in Avro format
        :param f: object of the class 'smart_open.Reader'
        :return: dictionary where key is the name of the column, value is the data type of the column
        """
        reader = DataFileReader(f, DatumReader())
        meta = eval(reader.meta['avro.schema'].decode())
        schema = {field["name"]: field["type"] for field in meta.get("fields", {})}
        return AvroSchemaConvertor(schema).converted_schema


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
