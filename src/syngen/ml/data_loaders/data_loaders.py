import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List
import pickle
import csv
import inspect

import pandas as pd
import pandas.errors
import pandavro as pdx
import yaml
from yaml import SafeLoader
from yaml.scanner import ScannerError
from avro.datafile import DataFileReader
from avro.io import DatumReader
from avro.errors import InvalidAvroBinaryEncoding
from loguru import logger

from syngen.ml.validation_schema import SUPPORTED_EXCEL_EXTENSIONS
from syngen.ml.convertor import CSVConvertor, AvroConvertor
from syngen.ml.utils import trim_string
from syngen.ml.context import get_context, global_context
from syngen.ml.validation_schema import (
    ExcelFormatSettingsSchema,
    CSVFormatSettingsSchema,
)

DELIMITERS = {"\\t": "\t"}


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
            raise ValueError("It seems that the information of source is absent")
        self.path = path
        self.file_loader = self._get_file_loader()
        self.has_existed_path = self.__check_if_path_exists()
        self.has_existed_destination = self.__check_if_path_exists(type_of_path="destination")

    def __check_if_path_exists(self, type_of_path="source"):
        if (type_of_path == "source" and os.path.exists(self.path)) or (
            type_of_path == "destination" and os.path.exists(os.path.dirname(self.path))
        ):
            return True
        return False

    def _get_file_loader(self):
        path = Path(self.path)
        if path.suffix == ".avro":
            return AvroLoader()
        elif path.suffix in [".csv", ".txt"]:
            return CSVLoader()
        elif path.suffix == ".tsv":
            return CSVLoader(sep="\t")
        elif path.suffix == ".psv":
            return CSVLoader(sep="|")
        elif path.suffix == ".pkl":
            return BinaryLoader()
        elif path.suffix in SUPPORTED_EXCEL_EXTENSIONS:
            return ExcelLoader()
        else:
            raise NotImplementedError(f"File format not supported for extension {path.suffix}")

    def load_data(self, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        try:
            df, schema = self.file_loader.load_data(self.path, **kwargs)
            return df, schema
        except UnicodeDecodeError as error:
            message = (
                f"It seems that the content of the data in the path - '{self.path}' "
                f"doesn't have the encoding UTF-8. The details of the error - {error}.\n"
                f"Please, use the data in UTF-8 encoding"
            )
            logger.error(message, error)
            raise error
        except pandas.errors.EmptyDataError as error:
            message = (
                f"The empty file was provided. Unable to load data from the path - '{self.path}'. "
                f"The details of the error - {error}"
            )
            logger.error(message, error)
            raise error

    def save_data(self, path: Optional[str], df: pd.DataFrame, **kwargs):
        if df is not None:
            self.file_loader.save_data(path, df, **kwargs)

    def get_columns(self) -> List[str]:
        return self.file_loader.get_columns(self.path)


class CSVLoader:
    """
    Class for loading and saving data in CSV format.
    """

    def __init__(self, **kwargs):
        self.format = get_context().get_config()
        self.format.update(kwargs)
        self.format = {
            k: v
            for k, v in self.format.items()
            if k in CSVFormatSettingsSchema._declared_fields.keys()
        }

    @staticmethod
    def _get_quoting(quoting: Optional[str]) -> int:
        quoting_map = {
            "minimal": csv.QUOTE_MINIMAL,
            "all": csv.QUOTE_ALL,
            "non-numeric": csv.QUOTE_NONNUMERIC,
            "none": csv.QUOTE_NONE,
        }
        if isinstance(quoting, int):
            return quoting
        else:
            return (
                quoting_map.get(quoting.lower(), csv.QUOTE_MINIMAL)
                if quoting
                else csv.QUOTE_MINIMAL
            )

    @staticmethod
    def _get_csv_params(**kwargs):
        params = {}
        format_params = kwargs.get("format")

        if format_params:
            params.update(format_params)
            quoting = format_params.get("quoting", None)
            params["quoting"] = CSVLoader._get_quoting(quoting)

        return params

    @staticmethod
    def _load_data(path, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        params = CSVLoader._get_csv_params(**kwargs)
        try:
            df = pd.read_csv(path, **params).apply(trim_string, axis=0)
            if all([isinstance(column, int) for column in df.columns]):
                df.rename(
                    columns={
                        k: f"column_{v}" for k, v in zip(df.columns, list(range(len(df.columns))))
                    },
                    inplace=True,
                )
            sep = params.get("sep", ",")
            if len(sep) > 1:
                params["sep"] = ","
            params["skiprows"] = None
            global_context(params)
        except FileNotFoundError as error:
            message = (
                f"It seems that the path to the table isn't valid.\n"
                f"The details of the error - {error}.\n"
                f"Please, check the path to the table"
            )
            logger.error(message)
            raise FileNotFoundError(message)

        return df, CSVConvertor({"fields": {}, "format": "CSV"}, df).schema

    def load_data(self, path, **kwargs):
        return self._load_data(path, format=self.format, **kwargs)

    def get_columns(self, path: str) -> List[str]:
        return self._get_columns(path)

    @staticmethod
    def _get_columns(path) -> List[str]:
        """
        Get the column names of the table located in the path
        """
        try:
            head_df = pd.read_csv(path, nrows=0)
            return list(head_df.columns)
        except pd.errors.EmptyDataError as error:
            logger.error(
                f"The empty file was provided. Unable to train this table located "
                f"in the path - '{path}'. The details of the error - {error}"
            )
            raise error

    @staticmethod
    def _save_data(path: Optional[str], df: pd.DataFrame, **kwargs):
        """
        Save the provided DataFrame to a CSV file.
        :param path: The file path to save the DataFrame.
        :param df: The DataFrame to be saved.
        :param kwargs: Additional keyword arguments to be passed to the to_csv method.
        """
        format_params = CSVLoader._get_csv_params(**kwargs)
        if df is not None:
            # Extract valid parameters
            valid_parameters = inspect.signature(pd.DataFrame.to_csv).parameters

            # Filter out any keyword arguments that are not valid parameters
            filtered_kwargs = {k: v for k, v in format_params.items() if k in valid_parameters}
            for k, v in filtered_kwargs.items():
                if isinstance(v, str) and v in DELIMITERS.keys():
                    filtered_kwargs[k] = v.replace(v, DELIMITERS[v])

            if "header" in filtered_kwargs and filtered_kwargs.get("header", None) is None:
                filtered_kwargs["header"] = False
            else:
                filtered_kwargs["header"] = True

            if "sep" in filtered_kwargs and len(filtered_kwargs.get("sep", None)) > 1:
                filtered_kwargs["sep"] = ","
                logger.warning(
                    "As the length of the value of the parameter 'separator' is more than "
                    "1 character, the 'separator' will be set to ',' in accordance with "
                    "the standard 'RFC 4180'"
                )
            if (
                "na_values" in format_params
                and format_params.get("na_values", [])
                and df.isnull().values.any()
            ):
                filtered_kwargs["na_rep"] = format_params["na_values"][0]
                logger.warning(
                    "Since the 'na_values' parameter is not empty, "
                    "the missing values will be filled with "
                    "the first value from the 'na_values' parameter"
                )

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
        :param path: the path to the file
        :return: dataframe
        """
        return pdx.from_avro(path)

    def load_data(self, path: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Load data in Avro format
        """
        try:
            with open(path, "rb") as f:
                df = self._load_df(f)
                schema = self._load_schema(f)
                return self._preprocess_schema_and_df(schema, df)
        except FileNotFoundError as error:
            message = (
                f"It seems that the path to the table isn't valid.\n"
                f"The details of the error - {error}.\n"
                f"Please, check the path to the table"
            )
            logger.error(message)
            raise FileNotFoundError(message)

    @staticmethod
    def _save_df(path: str, df: pd.DataFrame):
        """
        Save data in Avro Format
        """
        if df is not None:
            pdx.to_avro(path, df)

    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        self._save_df(path, df)

    @staticmethod
    def _load_schema(f) -> Dict[str, str]:
        """
        Load schema of the metadata of the table in Avro format
        :param f: object of the class 'smart_open.Reader'
        :return: dictionary where key is the name of the column,
        value is the data type of the column
        """
        reader = DataFileReader(f, DatumReader())
        meta = eval(reader.meta["avro.schema"].decode())
        schema = {field["name"]: field["type"] for field in meta.get("fields", {})}
        return schema

    @staticmethod
    def _preprocess_schema_and_df(
        schema: Dict[str, str], df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Preprocess schema and dataframe
        """
        convertor = AvroConvertor(schema, df)
        schema, preprocessed_df = convertor.converted_schema, convertor.preprocessed_df
        return preprocessed_df, schema

    def _get_columns(self, f) -> List[str]:
        """
        Get the column names of the table located in the path
        """
        schema = self._load_schema(f)
        return list(schema.keys())

    def get_columns(self, path: str) -> List[str]:
        try:
            with open(path, "rb") as f:
                return self._get_columns(f)
        except InvalidAvroBinaryEncoding as error:
            logger.error(
                f"The empty file was provided. Unable to train this table "
                f"located in the path - '{path}'. "
                f"The details of the error - {error}"
            )
            raise error


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
            if path.suffix in [".yaml", ".yml"]:
                return YAMLLoader()
            else:
                raise NotImplementedError("The format of metadata isn't supported")

    def load_data(self) -> dict:
        return self.metadata_loader.load_data(self.metadata_path)

    def save_data(self, path: str, metadata: Dict, **kwargs):
        self.metadata_loader.save_data(path, metadata, **kwargs)


class YAMLLoader(BaseDataLoader):
    """
    Class for loading and saving data in YAML format
    """

    _metadata_sections = ["train_settings", "infer_settings", "keys"]

    def _load_data(self, metadata_file) -> Dict:
        try:
            metadata = yaml.load(metadata_file, Loader=SafeLoader)
            metadata = self.replace_none_values_of_metadata_settings(
                self._metadata_sections, metadata
            )
            return metadata
        except ScannerError as error:
            message = (
                f"It seems that the metadata file in YAML format isn't valid.\n"
                f"The details of the error - {str(error)}.\n"
                f"Please check the metadata file in YAML format."
            )
            logger.error(error)
            raise ValueError(message)

    def load_data(self, metadata_path: str) -> Dict:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return self._load_data(f)

    @staticmethod
    def replace_none_values_of_metadata_settings(parameters, metadata: dict):
        """
        Replace None values for parameters in the metadata
        """
        metadata["global"] = dict() if metadata.get("global") is None else metadata.get("global")
        if metadata["global"]:
            for settings in metadata["global"].keys():
                metadata["global"][settings] = (
                    dict()
                    if metadata["global"].get(settings) is None
                    else metadata["global"].get(settings)
                )
        for key in metadata.keys():
            if key == "global":
                continue
            for parameter in parameters:
                if metadata.get(key).get(parameter) is None:
                    metadata[key][parameter] = {}
        return metadata

    def save_data(self, path: str, metadata: Dict, **kwargs):
        with open(path, "w") as f:
            self._save_data(metadata, f)

    @staticmethod
    def _save_data(metadata: Dict, f):
        yaml.dump(metadata, f)


class BinaryLoader(BaseDataLoader):
    """
    Class for loading and saving data using byte stream
    """

    @staticmethod
    def _load_data(f) -> Tuple[pd.DataFrame, None]:
        return pickle.load(f), None

    def load_data(self, path: str) -> Tuple[pd.DataFrame, None]:
        with open(path, "rb") as f:
            return self._load_data(f)

    @staticmethod
    def _save_data(data, f):
        pickle.dump(data, f)

    def save_data(self, path: str, data, **kwargs):
        with open(path, "wb") as f:
            self._save_data(data, f)


class ExcelLoader:
    """
    Class for loading and saving data in Excel format
    """

    def __init__(self):
        self.format = get_context().get_config()
        self.sheet_name = self.format.get("sheet_name", 0)
        self.format = {
            k: v
            for k, v in self.format.items()
            if k in ExcelFormatSettingsSchema._declared_fields.keys()
        }

    def _load_data(self, path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load data in Excel format
        """
        try:
            df = pd.read_excel(path, sheet_name=self.sheet_name)
            if isinstance(self.sheet_name, list) or self.sheet_name is None:
                dfs = [df for sheet_name, df in df.items()]
                df = pd.concat(dfs, ignore_index=True)
            global_context({})
            return df, CSVConvertor({"fields": {}, "format": "CSV"}, df).schema
        except FileNotFoundError as error:
            message = (
                f"It seems that the path to the table isn't valid.\n"
                f"The details of the error - {error}.\n"
                f"Please, check the path to the table"
            )
            logger.error(message, error)
            raise error

    def load_data(self, path):
        return self._load_data(path)

    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        self._save_data(path, df)

    @staticmethod
    def _save_data(path: str, df: pd.DataFrame):
        """
        Save provided data frame in Excel format
        """
        if df is not None:
            df.to_excel(path, index=False)

    def get_columns(self, path: str) -> List[str]:
        return self._get_columns(path)

    @staticmethod
    def _get_columns(path) -> List[str]:
        """
        Get the column names of the table located in the path
        """
        head_df = pd.read_excel(path, nrows=0)
        return list(head_df.columns)
