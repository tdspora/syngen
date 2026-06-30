import os
from copy import deepcopy
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List, Literal
import pickle as pkl
from dataclasses import dataclass

import pandas as pd
import pandas.errors
import pandavro as pdx
import yaml
from yaml import SafeLoader
from yaml.scanner import ScannerError
from avro.errors import InvalidAvroBinaryEncoding
from loguru import logger
import fastavro
from cryptography.fernet import Fernet

from syngen.ml.validation_schema import SUPPORTED_EXCEL_EXTENSIONS
from syngen.ml.convertor import CSVConvertor, AvroConvertor
from syngen.ml.utils import trim_string, fetch_env_variables
from syngen.ml.format_settings import set_format_settings, CSVFormatSettings, ExcelFormatSettings
from syngen.ml.validation_schema import ReportTypes


class BaseDataLoader(ABC):
    """
    Abstract class for data loader
    """
    def __init__(self, path: str):
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
    def __init__(
        self,
        path: str,
        table_name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        sensitive: bool = False
    ):
        super().__init__(path)
        self.fernet_key = (
            metadata.get(table_name)
            .get("encryption", {})
            .get("fernet_key")
        ) if (metadata is not None and table_name is not None) else None
        if sensitive and self.fernet_key and not self.path.endswith(".dat"):
            logger.warning(
                f"The provided Fernet key will be ignored because encryption and decryption "
                f"are not required for the data in the specified path: '{self.path}'"
            )
        self.sensitive = (
            True if sensitive and self.fernet_key and self.path.endswith(".dat") else False
        ) or self.path.endswith(".dat")
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
        if self.sensitive:
            return DataEncryptor(self.path, self.fernet_key)
        elif path.suffix == ".avro":
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

    def get_columns(self) -> List[str]:
        return self.file_loader.get_columns()


class CSVLoader(BaseDataLoader):
    """
    Class for loading and saving data in '.csv' format.
    """

    def __init__(self, path: str, **kwargs):
        super().__init__(path)
        format_settings = dict(kwargs)
        format_settings.update(CSVFormatSettings().format_settings)
        set_format_settings(format_settings)

    def _fetch_data(self):
        return (
            pd.read_csv(self.path, **CSVFormatSettings().load_format_settings)
            .apply(trim_string, axis=0)
        )

    def _load_data(self) -> Tuple[pd.DataFrame, Dict]:
        try:
            df = self._fetch_data()
            if all([isinstance(column, int) for column in df.columns]):
                df.rename(
                    columns={
                        k: f"column_{v}" for k, v in zip(df.columns, list(range(len(df.columns))))
                    },
                    inplace=True,
                )
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
        return self._load_data()

    def __get_columns(self):
        head_df = pd.read_csv(self.path, **CSVFormatSettings().load_format_settings, nrows=0)
        columns = list(head_df.columns)
        if all(isinstance(c, int) for c in columns):
            columns = [f"column_{i}" for i in range(len(columns))]
        return columns

    def get_columns(self) -> List[str]:
        return self._get_columns()

    def _get_columns(self) -> List[str]:
        """
        Get the column names of the table located in the path
        """
        try:
            return self.__get_columns()
        except pd.errors.EmptyDataError as error:
            logger.error(
                f"The empty file was provided. Unable to train this table located "
                f"in the path - '{self.path}'. The details of the error - {error}"
            )
            raise error

    def _write_data(self, df):
        """
        Save the dataframe in '.csv' format
        """
        df.to_csv(self.path, **CSVFormatSettings().save_format_settings, index=False)

    def _save_data(self, df: pd.DataFrame, **kwargs):
        """
        Save the provided DataFrame to a CSV file.
        """
        if df is not None:
            self._write_data(df)

    def save_data(self, df: pd.DataFrame, **kwargs):
        self._save_data(df)


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
    def _get_preprocessed_schema(schema: Optional[Dict]) -> Optional[Dict]:
        """
        Get the preprocessed schema
        """
        if schema is not None:
            return {
                field["name"]: field["type"]
                for field
                in schema.get("fields", {})
            }
        return schema

    def load_data(self, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """
        Load data in '.avro' format
        """
        try:
            df = self._load_data()
            schema = self.load_schema()
            return self._get_schema_and_df(schema, df)
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

    @staticmethod
    def _normalize_nullable_values_for_avro(
        df: pd.DataFrame,
        schema: Optional[Dict]
    ) -> pd.DataFrame:
        """
        Ensure Avro unions with null receive real Python None.
        To keep writer-schema compatibility, convert missing markers
        to `None` for fields whose schema explicitly allows `null`.
        """
        if schema is None:
            return df

        fields = schema.get("fields")

        df = df.copy()
        for field in fields:
            name = field.get("name")
            field_type = field.get("type")
            if not name or name not in df.columns:
                continue

            # Identify unions (e.g. ["null", "string"]) that allow null.
            union_types = field_type if isinstance(field_type, list) else None
            allows_null = bool(union_types) and any(t == "null" for t in union_types)
            if not allows_null:
                continue

            series = df[name]
            # Convert pandas missing markers to None, so the Avro union selects the null branch.
            series = series.astype(object).where(~series.isna(), None)
            df[name] = series

        return df

    @staticmethod
    def _extend_schema(schema: Dict, df: pd.DataFrame) -> Dict:
        """
        Return schema extended with inferred Avro fields for any columns in df
        that are absent from schema's field list.
        """
        existing_names = {f["name"] for f in schema.get("fields", [])}
        new_cols = [col for col in df.columns if col not in existing_names]
        if not new_cols:
            return schema
        inferred = pdx.schema_infer(df[new_cols])
        extended = deepcopy(schema)
        extended["fields"] = schema["fields"] + inferred["fields"]
        return extended

    def save_data(self, df: pd.DataFrame, schema: Optional[Dict], **kwargs):
        if schema is not None:
            schema = self._extend_schema(schema, df)
            logger.trace(f"The data will be saved with the schema: {schema}")

        preprocessed_schema = self._get_preprocessed_schema(schema)
        df = AvroConvertor(preprocessed_schema, df).preprocessed_df
        df = self._normalize_nullable_values_for_avro(df, schema)
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
    def _get_schema_and_df(
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

    def get_columns(self) -> List[str]:
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


@dataclass
class YAMLLoader(BaseDataLoader):
    """
    Class for loading and saving data in YAML format
    """
    metadata_sections = ["train_settings", "infer_settings", "format", "keys"]
    train_reports = ReportTypes().full_list_of_train_report_types
    infer_reports = ReportTypes().full_list_of_infer_report_types

    def __init__(self, path: str):
        super().__init__(path)

    def _normalize_reports(self, settings: dict, type_of_process: Literal["train", "infer"]):
        """
        Cast the value of the parameter 'reports' to the list
        """
        reports = settings.get(f"{type_of_process}_settings", {}).get("reports", [])
        if isinstance(reports, str):
            if reports not in ["none", "all"]:
                settings[f"{type_of_process}_settings"]["reports"] = [reports]
            elif reports == "none":
                settings[f"{type_of_process}_settings"]["reports"] = []
            elif reports == "all" and type_of_process == "train":
                settings[f"{type_of_process}_settings"]["reports"] = self.train_reports
            elif reports == "all" and type_of_process == "infer":
                settings[f"{type_of_process}_settings"]["reports"] = self.infer_reports

    def _normalize_parameter_reports(self, metadata: dict) -> dict:
        for table, settings in metadata.items():
            self._normalize_reports(settings, "train")
            self._normalize_reports(settings, "infer")
        return metadata

    def _fetch_encryption_settings(self, metadata: dict) -> dict:
        """
        Fetch the encryption settings, expecially 'fernet_key' parameter,
        from environment variables
        """
        errors = list()
        for table, settings in metadata.items():
            try:
                if "encryption" not in settings:
                    settings["encryption"] = {}
                else:
                    settings["encryption"] = fetch_env_variables(settings["encryption"])
            except ValueError as error:
                errors.append(str(error))
                continue
        if errors:
            errors = " ".join(errors)
            logger.error(errors)
            raise ValueError(errors)
        else:
            return metadata

    def _load_data(self, metadata_file) -> Dict:
        try:
            metadata = yaml.load(metadata_file, Loader=SafeLoader)
            metadata = self._replace_none_values_of_metadata_settings(metadata)
            metadata = self._fetch_encryption_settings(metadata)
            metadata = self._normalize_parameter_reports(metadata)

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

    def _replace_none_values_of_metadata_settings(self, metadata: dict):
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
            for parameter in self.metadata_sections:
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

    def get_columns(self) -> List[str]:
        data, _ = self.load_data()
        return data.columns.tolist()

    def load_data(self) -> Tuple[pd.DataFrame, None]:
        return self._load_data(), None

    def _save_data(self, data):
        with open(self.path, "wb") as f:
            pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

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
        excel_settings = ExcelFormatSettings()
        self.sheet_name = excel_settings.sheet_name
        self.format = excel_settings.load_format_settings

    def _fetch_data(self) -> pd.DataFrame:
        return pd.read_excel(self.path, sheet_name=self.sheet_name)

    def _load_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Load data in Excel format
        """
        try:
            data = self._fetch_data()
            if isinstance(self.sheet_name, list) or self.sheet_name is None:
                data = pd.concat(data.values(), ignore_index=True)
            return data, CSVConvertor(data).schema
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

    def _get_columns(self) -> List[str]:
        """
        Get the column names of the table located in the path
        """
        head_df = pd.read_excel(self.path, sheet_name=self.sheet_name, nrows=0)
        return list(head_df.columns)

    def get_columns(self) -> List[str]:
        return self._get_columns()


class DataEncryptor(BaseDataLoader):
    """
    A class to handle encryption and decryption of data using a Fernet key
    """

    def __init__(self, path: str, fernet_key: Optional[str]):
        """
        Initialize the DataEncryptor with a Fernet key.
        """
        super().__init__(path)
        self.validate_fernet_key(fernet_key)
        self.fernet = Fernet(fernet_key)

    @classmethod
    def validate_fernet_key(cls, fernet_key: str):
        """
        Validate the provided Fernet key.
        A valid Fernet key is a 44-character URL-safe base64-encoded string.
        """
        if fernet_key is None or not fernet_key.strip():
            raise ValueError("It seems that the Fernet key is absent")

        error_message = "It seems that the provided Fernet key is invalid"
        try:
            Fernet(fernet_key.encode())

        except ValueError as e:
            logger.error(f"{error_message}. {str(e)}")
            raise e

    def get_columns(self) -> List[str]:
        """
        Get the column names of the table
        """
        data, _ = self.load_data()
        return data.columns.tolist()

    @staticmethod
    def _save_data(path: str, data: bytes):
        """
        Save the encrypted dataframe to the disk
        """
        with open(path, "wb") as encrypted_file:
            encrypted_file.write(data)

    def save_data(self, df: pd.DataFrame, **kwargs):
        """
        Save the encrypted dataframe to the disk
        """
        try:
            serialized_df: bytes = pkl.dumps(df, protocol=pkl.HIGHEST_PROTOCOL)
            encrypted_data = self.fernet.encrypt(serialized_df)

            # Use atomic write operation for better safety
            temp_path = f"{self.path}.tmp"
            self._save_data(temp_path, encrypted_data)
            os.replace(temp_path, self.path)

            logger.info(
                f"Data is successfully encrypted and saved to '{self.path}'."
            )
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"Encryption failed: {str(e)}")
            raise e

    def _load_data(self) -> bytes:
        """
        Read encrypted data from disk.
        """
        with open(self.path, "rb") as encrypted_file:
            return encrypted_file.read()

    def load_data(self, *args, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """
        Load the decrypted data from the disk.
        """
        try:
            encrypted_data = self._load_data()
            decrypted_data = self.fernet.decrypt(encrypted_data)
            df_decrypted = pkl.loads(decrypted_data)

            logger.info(
                f"Data stored at the path - '{self.path}' "
                f"has been successfully decrypted and loaded."
            )
            return df_decrypted, {"fields": {}, "format": "CSV"}
        except Exception as e:
            logger.error(
                f"It seems that the decryption process of the data "
                f"stored at the path - '{self.path}' failed - {str(e)}"
            )
            raise e
