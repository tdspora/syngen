from typing import Dict, List, Tuple, Optional
from collections import Counter
from datetime import datetime, date
import json

import pandas as pd
import numpy as np
from loguru import logger
import chardet


class Convertor:
    """
    Abstract class for converting the fetched schema in Avro, Parquet or Delta formats
    """

    _DECODABLE_MIME_TYPES = {"text/plain", None}
    _NON_DECODABLE_TEXT_MIME_TYPES = {"text/html", "text/xml", "text/x-python"}
    _NON_DECODABLE_MIME_PREFIXES = (
        "image/", "audio/", "video/", "application/", "font/"
    )


    def __init__(self, original_schema: Optional[Dict] = None, df: pd.DataFrame = None):
        self.original_schema = original_schema
        self.df = df
        self.excluded_dtypes: Tuple = (str, bytes, datetime, date, bool, list, dict)
        self.custom_schema = {}
        self.preprocessed_df = self.df.copy()

    def _check_dtype_or_nan(self, included_dtypes: Tuple, excluded_dtypes: Tuple = ()):
        """
        Check if the value is of the specified data types or 'np.NaN'
        """
        return (
            lambda x: (isinstance(x, included_dtypes) and not isinstance(x, excluded_dtypes))
            or (not isinstance(x, self.excluded_dtypes) and np.isnan(x))
        )

    def _cast_columns_to_schema_types(self):
        """
        Cast DataFrame column types to match the schema
        """
        type_map = {
            "string": "string",
            "double": "float64",
            "float": "float64",
            "decimal": "float64"
        }
        for column, data_type in self.custom_schema.get("fields", {}).items():
            if data_type == "null":
                if self.preprocessed_df[column].isnull().all():
                    continue
                else:
                    raise ValueError(
                        f"It seems that the data type - '{data_type}' "
                        f"isn't correct for the column - '{column}' as it's not empty"
                    )

            if data_type in type_map:
                self.preprocessed_df[column] = (
                    self.preprocessed_df[column].astype(type_map[data_type])
                )
            elif data_type == "boolean":
                # Preserve booleans as booleans (Avro schema boolean should not be coerced to int).
                # If there are missing values, use pandas nullable boolean dtype.
                if any(self.preprocessed_df[column].isnull()):
                    self.preprocessed_df[column] = self.preprocessed_df[column].astype("boolean")
                else:
                    self.preprocessed_df[column] = self.preprocessed_df[column].astype(bool)
            elif data_type == "int":
                if self.preprocessed_df[column].isnull().any():
                    self.preprocessed_df[column] = self.preprocessed_df[column].astype("float64")
                else:
                    self.preprocessed_df[column] = self.preprocessed_df[column].astype("int64")

    def _cast_columns_if_schema_is_not_provided(self):
        """
        Cast DataFrame column types if the schema is not provided
        just based on the values contained in columns
        """

        for column in self.preprocessed_df.columns:
            # Check for boolean first (bool is subclass of int in Python)
            if self.preprocessed_df[column].map(
                self._check_dtype_or_nan(included_dtypes=(bool,))
            ).all():
                # Keep boolean columns as-is, don't convert to int
                continue
            elif self.preprocessed_df[column].map(
                self._check_dtype_or_nan(included_dtypes=(int,), excluded_dtypes=(bool,))
            ).all():
                if self.preprocessed_df[column].isnull().any():
                    self.preprocessed_df[column] = self.preprocessed_df[column].astype("float64")
                else:
                    self.preprocessed_df[column] = self.preprocessed_df[column].astype('int64')
            elif self.preprocessed_df[column].map(
                self._check_dtype_or_nan(included_dtypes=(int, float))
            ).all():
                self.preprocessed_df[column] = self.preprocessed_df[column].astype(float)
            elif self.preprocessed_df[column].map(
                self._check_dtype_or_nan(included_dtypes=(str, bytes))
            ).all():
                self.preprocessed_df[column] = (
                    self.preprocessed_df[column].astype(pd.StringDtype())
                )

    def _get_serializable_columns(self):
        """
        Get the columns with complex data types (dict, list, tuple, 'numpy.ndarray') or binary)
        that require serialization to JSON strings
        """

        return [
            column
            for column, data_type in self.custom_schema.get("fields", {}).items()
            if data_type in ("complex", "binary")
        ]

    def _set_none_values_to_nan(self):
        """
        Set 'None' values contained in columns as 'np.NaN'
        """
        df_object_subset = self.preprocessed_df.select_dtypes(["object"])
        for column in df_object_subset:
            self.preprocessed_df[column] = [
                np.NaN
                if i is None
                else i
                for i in self.preprocessed_df[column]
            ]

    def _cast_values_to_string(self):
        """
        Cast the values contained in columns with the data type 'object' to 'string'
        """
        for column in self.preprocessed_df.select_dtypes(["object"]):
            self.preprocessed_df[column] = [
                i
                if (not isinstance(i, self.excluded_dtypes) and pd.isna(i))
                or isinstance(i, self.excluded_dtypes)
                else str(i)
                for i in self.preprocessed_df[column]
            ]

    def _serialize_complex_value(self, x):
        """
        Serialize complex value (dict, list, tuple, 'numpy.ndarray') to JSON string
        """
        if isinstance(x, np.ndarray):
            return json.dumps(x.tolist())
        elif isinstance(x, (dict, list, tuple)):
            return json.dumps(x)
        return x

    def _get_encoding_info(self, value) -> dict:
        """
        Detect the mime type and the character encoding of a single binary value.
        """
        result = chardet.detect(value)
        return {
            "mime_type": result["mime_type"],
            "encoding": result["encoding"],
        }

    def _collect_encoding_infos(self, column: str) -> List[dict]:
        """
        Run `chardet` on every non-null binary value in `column` and return the
        list of mime/encoding records.
        """
        return [
            self._get_encoding_info(value)
            for value in self.preprocessed_df[column]
            if isinstance(value, (bytes, bytearray))
        ]

    def _validate_binary_encoding_infos(
        self, column: str, encoding_infos: List[dict]
    ) -> None:
        """
        Validate that every value in a binary column is a decodable plain-text
        payload. Raise ValueError if any value is detected as a file format
        (image / audio / video / archive / font / executable / structured text
        such as HTML / XML / Python source) or as plain text whose character
        encoding could not be determined.
        """
        non_decodable_mime_types = sorted({
            info["mime_type"]
            for info in encoding_infos
            if info["mime_type"] in self._NON_DECODABLE_TEXT_MIME_TYPES
            or (
                info["mime_type"] is not None
                and info["mime_type"].startswith(self._NON_DECODABLE_MIME_PREFIXES)
            )
        })
        if non_decodable_mime_types:
            non_decodable_mime_types = "', '".join(non_decodable_mime_types)
            message = (
                f"The binary column '{column}' contains values whose content "
                f"cannot be processed as plain text. Detected MIME type(s): "
                f"'{non_decodable_mime_types}'. Only binary columns whose values "
                f"are plain-text payloads ('text/plain') are supported."
            )
            logger.error(message)
            raise ValueError(message)

        if any(
            info["mime_type"] == "text/plain" and info["encoding"] is None
            for info in encoding_infos
        ):
            message = (
                f"The binary column '{column}' contains plain-text values "
                f"whose character encoding could not be determined. Please "
                f"ensure every value uses a standard encoding before running training."
            )
            logger.error(message)
            raise ValueError(message)

        plain_text_encodings = sorted({
            info["encoding"]
            for info in encoding_infos
            if info["mime_type"] == "text/plain" and info["encoding"] is not None
        })
        if len(plain_text_encodings) > 1:
            encodings_str = "', '".join(plain_text_encodings)
            message = (
                f"The binary column '{column}' contains plain-text values with "
                f"multiple different character encodings: '{encodings_str}'. "
                f"All values in a binary column must use the same character encoding. "
                f"Please ensure every value is encoded consistently before running training."
            )
            logger.error(message)
            raise ValueError(message)
        self.custom_schema.setdefault("encoding", {})[column] = plain_text_encodings[0]

    def _cast_binary_column(self, column: str) -> None:
        """
        Decode every binary value in `column` to a string using the most
        popular detected character encoding, and record that encoding under
        `custom_schema["encoding"][column]` so downstream consumers
        can re-encode the synthetic output to bytes.
        """
        encoding_infos = self._collect_encoding_infos(column)
        self._validate_binary_encoding_infos(column, encoding_infos)
        encoding = self.custom_schema["encoding"][column]

        def _decode(value):
            if not isinstance(value, (bytes, bytearray)) or encoding is None:
                return value
            try:
                return value.decode(encoding)
            except (UnicodeDecodeError, TypeError) as exc:
                message = (
                    f"Failed to decode a value in the binary column '{column}' "
                    f"using the most popular detected encoding '{encoding}'. "
                    f"Underlying error: {exc}."
                )
                logger.error(message)
                raise ValueError(message) from exc

        self.preprocessed_df[column] = self.preprocessed_df[column].map(_decode)

    def _cast_values_to_json(self):
        """
        Cast the values contained in columns with complex data types
        (dict, list, tuple, 'numpy.ndarray') to JSON strings. Binary columns
        are decoded to strings using the most popular detected encoding.
        """
        for column in self._get_serializable_columns():
            if column not in self.preprocessed_df.columns:
                continue
            data_type = self.custom_schema["fields"].get(column)
            if data_type == "binary":
                self._cast_binary_column(column)
            else:
                self.preprocessed_df[column] = (
                    self.preprocessed_df[column].map(self._serialize_complex_value)
                )


    def _preprocess_df(self):
        """
        Get the preprocessed data frame, update data types of columns
        """
        if not self.preprocessed_df.empty:
            try:
                self._set_none_values_to_nan()
                if getattr(self, "complex_types", None) is not None:
                    self._cast_values_to_json()
                self._cast_values_to_string()
                if self.custom_schema.get("format") == "CSV":
                    self._cast_columns_if_schema_is_not_provided()
                else:
                    self._cast_columns_to_schema_types()
            except Exception as e:
                logger.error(e)
                raise e


class CSVConvertor(Convertor):
    """
    Class for supporting the custom schema for csv files
    """

    def __init__(self, df):
        super().__init__(original_schema=None, df=df)
        self.custom_schema = {"fields": {}, "format": "CSV"}
        self._preprocess_df()


class AvroConvertor(Convertor):
    """
    Class for converting the fetched avro schema
    """

    def __init__(self, original_schema: Optional[Dict] = None, df: pd.DataFrame = None):
        super().__init__(original_schema, df)
        self.original_schema = original_schema or {}
        self.complex_types = {"array", "map", "record", "enum", "fixed"}
        self.custom_schema = self._get_custom_schema()
        self._preprocess_df()

    def _get_custom_schema(self) -> Dict:
        """
        Convert the schema of Avro file to the unified format
        """
        def _extract_type_names(avro_type) -> set[str]:
            if isinstance(avro_type, list):
                names: set[str] = set()
                for t in avro_type:
                    names |= _extract_type_names(t)
                return names
            if isinstance(avro_type, dict):
                return _extract_type_names(avro_type.get("type"))
            if avro_type is None:
                return {"null"}
            return {avro_type}

        custom_schema = dict()
        custom_schema["fields"] = dict()
        for column, data_type in self.original_schema.items():
            fields = custom_schema["fields"]
            type_names = _extract_type_names(data_type)

            if "boolean" in type_names:
                fields[column] = "boolean"
            elif type_names & {"int", "long"}:
                fields[column] = "int"
            elif type_names & {"float", "double"}:
                fields[column] = "float"
            elif type_names & {"string", "bytes"}.union(self.complex_types):
                fields[column] = "string"
            elif type_names == {"null"}:
                fields[column] = "null"
            else:
                message = (
                    f"It seems that the column - '{column}' has unsupported data type - "
                    f"'{data_type}'"
                )
                logger.error(message)
                raise ValueError(message)
        custom_schema["format"] = "Avro"
        return custom_schema
