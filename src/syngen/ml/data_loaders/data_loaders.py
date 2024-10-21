import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List
import pickle as pkl
import csv
import inspect

import pandas as pd
import pandas.errors
import pandavro as pdx
import yaml
from yaml import SafeLoader
from yaml.scanner import ScannerError
from avro.errors import InvalidAvroBinaryEncoding
from loguru import logger
import fastavro

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
    def __init__(self, path: str):
        if not path:
            raise ValueError("It seems that the information of source is absent")
        self.path = path

    @abstractmethod
    def load_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        pass


class DataLoader(BaseDataLoader):
    """
    Base class for loading and saving data
    """

    def __init__(self, path: str):
        super().__init__(path)
        self.file_loader = self._get_file_loader()
        self.has_existed_path = self.__check_if_path_exists()
        self.has_existed_destination = self.__check_if_path_exists(type_of_path="destination")

    @property
    def original_schema(self):
        return (
            self.file_loader.load_original_schema()
            if hasattr(self.file_loader, "load_original_schema")
            else None
        )

    def __check_if_path_exists(self, type_of_path="source"):
        if (type_of_path == "source" and os.path.exists(self.path)) or (
            type_of_path == "destination" and os.path.exists(os.path.dirname(self.path))
        ):
            return True
        return False

    def _get_file_loader(self):
        path = Path(self.path)
        if path.suffix == ".avro":
            return AvroLoader(self.path)
        elif path.suffix in [".csv", ".txt"]:
            return CSVLoader(self.path)
        elif path.suffix == ".tsv":
            return CSVLoader(self.path, sep="\t")
        elif path.suffix == ".psv":
            return CSVLoader(self.path, sep="|")
        elif path.suffix == ".pkl":
            return BinaryLoader(self.path)
        elif path.suffix in SUPPORTED_EXCEL_EXTENSIONS:
            return ExcelLoader(self.path)
        else:
            raise NotImplementedError(f"File format not supported for extension {path.suffix}")

    def load_data(self, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        try:
            df, schema = self.file_loader.load_data(**kwargs)
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

    def save_data(self, data, **kwargs):
        if data is not None:
            self.file_loader.save_data(data, **kwargs)

    def get_columns(self, **kwargs) -> List[str]:
        return self.file_loader.get_columns(**kwargs)


class CSVLoader(BaseDataLoader):
    """
    Class for loading and saving data in '.csv' format.
    """

    def __init__(self, path: str, **kwargs):
        super().__init__(path)
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

    def _fetch_data(self, **params):
        return pd.read_csv(self.path, **params).apply(trim_string, axis=0)

    def _load_data(self, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        params = CSVLoader._get_csv_params(**kwargs)
        try:
            df = self._fetch_data(**params)
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

        return df, CSVConvertor(df).schema

    def load_data(self, **kwargs):
        return self._load_data(format=self.format, **kwargs)

    def __get_columns(self, **kwargs):
        head_df = pd.read_csv(self.path, **kwargs, nrows=0)
        return list(head_df.columns)

    def get_columns(self, **kwargs) -> List[str]:
        return self._get_columns(**kwargs)

    def _get_columns(self, **kwargs) -> List[str]:
        """
        Get the column names of the table located in the path
        """
        try:
            return self.__get_columns(**kwargs)
        except pd.errors.EmptyDataError as error:
            logger.error(
                f"The empty file was provided. Unable to train this table located "
                f"in the path - '{self.path}'. The details of the error - {error}"
            )
            raise error

    def _write_data(self, df, **kwargs):
        """
        Save the dataframe in '.csv' format
        """
        df.to_csv(self.path, **kwargs, index=False)

    def _save_data(self, df: pd.DataFrame, **kwargs):
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

            self._write_data(df, **filtered_kwargs)

    def save_data(self, df: pd.DataFrame, **kwargs):
        self._save_data(df, **kwargs)


class AvroLoader(BaseDataLoader):
    """
    Class for loading and saving data in '.avro' format
    """

    def _load_data(self) -> pd.DataFrame:
        """
        Load data in '.avro' format
        """
        with open(self.path, "rb") as f:
            return pdx.from_avro(f)

    @staticmethod
    def _get_preprocessed_schema(schema: Dict) -> Dict:
        """
        Get the preprocessed schema
        """
        if schema is not None:
            return {
                field["name"]: field["type"]
                for field
                in schema.get("fields", {})
            }

    def load_data(self, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """
        Load data in '.avro' format
        """
        try:
            df = self._load_data()
            schema = self.load_schema()
            return self._preprocess_schema_and_df(schema, df)
        except FileNotFoundError as error:
            message = (
                f"It seems that the path to the table isn't valid.\n"
                f"The details of the error - {error}.\n"
                f"Please, check the path to the table"
            )
            logger.error(message)
            raise FileNotFoundError(message)

    def _save_data(self, df: pd.DataFrame, schema: Optional[Dict]):
        """
        Save data in Avro Format
        """
        if df is not None:
            pdx.to_avro(self.path, df, schema)

    def save_data(self, df: pd.DataFrame, schema: Optional[Dict] = None, **kwargs):
        if schema is not None:
            logger.trace(f"The data will be saved with the schema: {schema}")
            preprocessed_schema = (
                self._get_preprocessed_schema(schema) if schema is not None else schema
            )
            df = AvroConvertor(preprocessed_schema, df).preprocessed_df
        self._save_data(df, schema)

    def __load_original_schema(self):
        with open(self.path, "rb") as f:
            return self._load_original_schema(f)

    @staticmethod
    def _load_original_schema(f):
        """
        Load schema of the metadata of the table in '.avro' format
        """
        reader = fastavro.reader(f)
        return reader.writer_schema

    def load_original_schema(self) -> Dict:
        return self.__load_original_schema()

    def load_schema(self) -> Dict[str, str]:
        """
        Load the custom schema of the metadata of the table in '.avro' format
        used by the algorithm
        """
        original_schema = self.load_original_schema()
        return self._get_preprocessed_schema(original_schema)

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

    def _get_columns(self) -> List[str]:
        """
        Get the column names of the table located in the path
        """
        schema = self.load_schema()
        return list(schema.keys())

    def get_columns(self, **kwargs) -> List[str]:
        try:
            return self._get_columns()
        except InvalidAvroBinaryEncoding as error:
            logger.error(
                f"The empty file was provided. Unable to train this table "
                f"located in the path - '{self.path}'. "
                f"The details of the error - {error}"
            )
            raise error


class MetadataLoader(BaseDataLoader):
    """
    Metadata class for loading and saving metadata in YAML format
    """

    def __init__(self, path: str):
        super().__init__(path)
        self.metadata_loader = self.get_metadata_loader()

    def get_metadata_loader(self):
        if self.path is not None:
            path = Path(self.path)
            if path.suffix in [".yaml", ".yml"]:
                return YAMLLoader(self.path)
            else:
                raise NotImplementedError("The format of metadata isn't supported")

    def load_data(self) -> Dict:
        return self.metadata_loader.load_data()

    def save_data(self, metadata: Dict, **kwargs):
        self.metadata_loader.save_data(metadata, **kwargs)


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

    def load_data(self) -> Dict:
        with open(self.path, "r", encoding="utf-8") as f:
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

    def save_data(self, metadata: Dict, **kwargs):
        with open(self.path, "w") as f:
            self._save_data(metadata, f)

    @staticmethod
    def _save_data(metadata: Dict, f):
        yaml.dump(metadata, f)


class BinaryLoader(BaseDataLoader):
    """
    Class for loading and saving data using byte stream
    """

    def _load_data(self) -> pd.DataFrame:
        """
        Load data in Binary format
        """
        with open(self.path, "rb") as f:
            return pkl.load(f)

    def load_data(self) -> Tuple[pd.DataFrame, None]:
        return self._load_data(), None

    def _save_data(self, data):
        with open(self.path, "wb") as f:
            pkl.dump(data, f)

    def save_data(self, data, **kwargs):
        """
        Save data in Binary format
        """
        self._save_data(data)


class ExcelLoader(BaseDataLoader):
    """
    Class for loading and saving data in Excel format
    """

    def __init__(self, path: str):
        super().__init__(path)
        self.format = get_context().get_config()
        self.sheet_name = self.format.get("sheet_name", 0)
        self.format = {
            k: v
            for k, v in self.format.items()
            if k in ExcelFormatSettingsSchema._declared_fields.keys()
        }

    def _fetch_data(self) -> pd.DataFrame:
        return pd.read_excel(self.path, sheet_name=self.sheet_name)

    def _load_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Load data in Excel format
        """
        try:
            df = self._fetch_data()
            if isinstance(self.sheet_name, list) or self.sheet_name is None:
                dfs = [df for sheet_name, df in df.items()]
                df = pd.concat(dfs, ignore_index=True)
            global_context({})
            return df, CSVConvertor(df).schema
        except FileNotFoundError as error:
            message = (
                f"It seems that the path to the table isn't valid.\n"
                f"The details of the error - {error}.\n"
                f"Please, check the path to the table"
            )
            logger.error(message, error)
            raise error

    def load_data(self) -> Tuple[pd.DataFrame, Dict]:
        return self._load_data()

    def save_data(self, df: pd.DataFrame, **kwargs):
        self._save_data(df)

    def _save_data(self, df: pd.DataFrame):
        """
        Save provided data frame in Excel format
        """
        if df is not None:
            df.to_excel(self.path, index=False, engine="openpyxl")

    def _get_columns(self, **kwargs) -> List[str]:
        """
        Get the column names of the table located in the path
        """
        head_df = pd.read_excel(self.path, **kwargs, nrows=0)
        return list(head_df.columns)

    def get_columns(self, **kwargs) -> List[str]:
        return self._get_columns(**kwargs)
