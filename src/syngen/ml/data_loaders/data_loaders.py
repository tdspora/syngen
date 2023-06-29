import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
import pickle
import csv
import copy
import inspect

import pandas as pd
import pandavro as pdx
import yaml
from yaml import Loader
from avro.datafile import DataFileReader
from avro.io import DatumReader

from syngen.ml.validation_schema import validate_schema
from syngen.ml.convertor import CSVConvertor, AvroConvertor
from syngen.ml.utils import trim_string
from syngen.ml.custom_logger import custom_logger


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

    def __init__(self, path: str, format: Dict = None):
        if not path:
            raise ValueError("It seems that the information of source is absent")
        self.path = path
        self.format = format
        self.file_loader = self.__get_file_loader()
        self.has_existed_path = self.check_if_path_exists()

    def check_if_path_exists(self):
        if os.path.exists(self.path):
            return True
        return False

    def __get_file_loader(self):
        path = Path(self.path)
        if path.suffix == '.avro':
            return AvroLoader()
        elif path.suffix == '.csv':
            return CSVLoader()
        elif path.suffix == ".pkl":
            return BinaryLoader()
        else:
            raise NotImplementedError(f"File format not supported for extension {path.suffix}")

    def load_data(self) -> Tuple[pd.DataFrame, Dict]:
        try:
            df, schema = self.file_loader.load_data(self.path, format=self.format)
            if df.shape[0] < 1:
                raise ValueError("Empty file was provided. Unable to train")
            return df, schema
        except UnicodeDecodeError as error:
            message = f"It seems that the content of the data in the path - '{self.path}' " \
                      f"doesn't have the encoding UTF-8. The details of the error - {error}.\n" \
                      f"Please, use the data in UTF-8 encoding"
            custom_logger.error(message)
            raise ValueError(message)

    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        self.file_loader.save_data(path, df, format=self.format)


class CSVLoader:
    """
    Class for loading and saving data in CSV format.
    """
    
    RFC_4180 = {
        'sep': ',',
        'quotechar': '"',
        'quoting': csv.QUOTE_MINIMAL,
        'escapechar': '\\'
    }

    @staticmethod
    def _get_quoting(quoting: Optional[str]) -> int:
        quoting_map = {
            'minimal': csv.QUOTE_MINIMAL,
            'all': csv.QUOTE_ALL,
            'non-numeric': csv.QUOTE_NONNUMERIC,
            'none': csv.QUOTE_NONE
        }
        return quoting_map.get(quoting.lower(), csv.QUOTE_NONE) if quoting else csv.QUOTE_NONE

    @staticmethod
    def _get_csv_params(engine: str = 'c', **kwargs):
        params = copy.deepcopy(CSVLoader.RFC_4180)
        params.update(kwargs)
        format_params = kwargs.get('format')
        
        if format_params:
            params['quoting'] = CSVLoader._get_quoting(format_params.pop('quoting', None))
            engine = format_params.pop('engine', engine)
            params.update(format_params)
        if 'format' in kwargs.keys():
            params.pop('format')
            
        return engine, params

    @staticmethod
    def _load_data(path, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        engine, params = CSVLoader._get_csv_params(**kwargs)
        
        try:
            df = pd.read_csv(path, engine=engine, **params).apply(trim_string, axis=0)
        except FileNotFoundError as error:
            message = (f"It seems that the path to the table isn't valid.\n"
                       f"The details of the error - {error}.\n"
                       f"Please, check the path to the table")
            custom_logger.error(message)
            raise FileNotFoundError(message)
        
        return df, CSVConvertor({"fields": {}, "format": "CSV"}, df).schema


    def load_data(self, path, **kwargs):
        return self._load_data(path, **kwargs)

    @staticmethod
    def _save_data(path: Optional[str], df: pd.DataFrame, **kwargs):
        """
        Save the provided DataFrame to a CSV file.
        
        :param path: The file path to save the DataFrame.
        :param df: The DataFrame to be saved.
        :param kwargs: Additional keyword arguments to be passed to the to_csv method.
        """
        if df is not None:
            # Extract valid parameters
            valid_parameters = inspect.signature(pd.DataFrame.to_csv).parameters
            
            # Filter out any keyword arguments that are not valid parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_parameters}
            
            # Save the DataFrame to a CSV file
            df.to_csv(path, **filtered_kwargs, index=False)
        

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
        str which should be in the next format for the connection to AWS S3: "s3://path/to/bucket"
        str which should be in the next format for the connection to Azure Storage: "azure://{container_name}/{blob_name}"
        """
        try:
            with open(path, 'rb') as f:
                df = self._load_df(f)
                schema, preprocessed_df = self._load_schema(f, df)
                return preprocessed_df, schema
        except FileNotFoundError as error:
            message = f"It seems that the path to the table isn't valid.\n" \
                      f"The details of the error - {error}.\n" \
                      f"Please, check the path to the table"
            custom_logger.error(message)
            raise FileNotFoundError(message)

    @staticmethod
    def save_data(path: str, df: pd.DataFrame, **kwargs):
        if df is not None:
            pdx.to_avro(path, df, **kwargs)

    @staticmethod
    def _load_schema(f, df) -> Tuple[Dict[str, str], pd.DataFrame]:
        """
        Load schema of the metadata of the table in Avro format and preprocess dataframe
        :param f: object of the class 'smart_open.Reader'
        :return: dictionary where key is the name of the column, value is the data type of the column
        """
        reader = DataFileReader(f, DatumReader())
        meta = eval(reader.meta['avro.schema'].decode())
        schema = {field["name"]: field["type"] for field in meta.get("fields", {})}
        convertor = AvroConvertor(schema, df)
        schema, preprocessed_df = convertor.converted_schema, convertor.preprocessed_df
        return schema, preprocessed_df


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
                raise NotImplementedError("The format of metadata isn't supported")

    def load_data(self) -> dict:
        return self.metadata_loader.load_data(self.metadata_path)

    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        self.metadata_loader.save_data(path, df, **kwargs)


class YAMLLoader(BaseDataLoader):
    """
    Class for loading and saving data in YAML format
    """
    def load_data(self, metadata_path: str) -> dict:
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = yaml.load(metadata_file, Loader=Loader)
            validate_schema(metadata)
            parameters = ["train_settings", "infer_settings", "keys"]
            metadata = self.replace_none_values_of_metadata_settings(parameters, metadata)
        return metadata

    @staticmethod
    def replace_none_values_of_metadata_settings(parameters, metadata: dict):
        """
        Replace None values for parameters in the metadata
        """
        metadata["global"] = dict() if metadata.get("global") is None else metadata.get("global")
        if metadata["global"]:
            for settings in metadata["global"].keys():
                metadata["global"][settings] = dict() if metadata["global"].get(settings) is None \
                        else metadata["global"].get(settings)
        for key in metadata.keys():
            if key == "global":
                continue
            for parameter in parameters:
                if metadata.get(key).get(parameter) is None:
                    metadata[key][parameter] = {}
        return metadata

    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        raise NotImplementedError("Saving YAML files is not supported")


class BinaryLoader(BaseDataLoader):
    """
    Class for loading and saving data using byte stream
    """

    def load_data(self, path: str, **kwargs) -> Tuple[pd.DataFrame, None]:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data, None

    def save_data(self, path: str, data, **kwargs):
        with open(path, "wb") as f:
            pickle.dump(data, f)
