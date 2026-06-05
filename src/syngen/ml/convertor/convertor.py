from typing import Dict, List, Tuple, Optional
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

    def __init__(
        self,
        original_schema: Optional[Dict] = None,
        df: pd.DataFrame = None,
        serialize_complex_types: bool = True
    ):
        self.original_schema = original_schema
        self.df = df
        self.excluded_dtypes: Tuple = (str, bytes, datetime, date, bool, list, dict)
        self.custom_schema = {}
        self.preprocessed_df = self.df.copy()
        self.serialize_complex_types = serialize_complex_types

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

    def _get_serializable_columns_mapping(self):
        """
        Get the mapping of columns and their data types with complex data types
        (dict, list, tuple, 'numpy.ndarray') or binary)
        that require serialization to JSON strings
        """
        return {
            column: data_type
            for column, data_type in self.custom_schema.get("fields", {}).items()
            if data_type.startswith("complex") or data_type == "binary"
        }

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

    def _serialize_complex_value(self, x, column_name):
        """
        Serialize complex value (dict, list, tuple, 'numpy.ndarray') to JSON string.
        Put the column name as a key for list, tuple and 'numpy.ndarray' types
        to be able to flatten it later.
        """
        if isinstance(x, np.ndarray):
            return json.dumps(dict({f"{column_name}": x.tolist()}))
        elif isinstance(x, dict):
            return json.dumps(x)
        elif isinstance(x, (list, tuple)):
            return json.dumps(dict({f"{column_name}": x}))
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
        Run `chardet` on every non-null binary value in `column`
        and return the list of `mime/encoding` records.
        """
        return [
            self._get_encoding_info(value)
            for value in self.preprocessed_df[column]
            if isinstance(value, (bytes, bytearray))
        ]

    def _validate_binary_encoding_infos(
        self, column: str, encoding_infos: List[dict]
    ) -> bool:
        """
        Validate that every value in a binary column is a decodable plain-text
        payload. Returns `True` when decoding can proceed (single encoding detected
        and stored in custom_schema). Returns `False` and emits a warning when
        decoding should be skipped:
          - non-decodable MIME type (`image/*`, `audio/*`, `video/*`, `application/*`,
            `font/*`, `text/html`, `text/xml`, `text/x-python`);
          - encoding could not be determined (`chardet` returns `None`);
          - multiple different encodings found across values.
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
            mime_str = "', '".join(non_decodable_mime_types)
            logger.warning(
                f"The binary column '{column}' contains values with non-decodable "
                f"MIME type(s): '{mime_str}'. "
                f"Decoding will be skipped; binary values will be placed as-is."
            )
            return False

        if any(
            info["mime_type"] in self._DECODABLE_MIME_TYPES and info["encoding"] is None
            for info in encoding_infos
        ):
            logger.warning(
                f"The binary column '{column}' contains values "
                f"whose character encoding could not be determined. "
                f"Decoding will be skipped; binary values will be placed as-is."
            )
            return False

        plain_text_encodings = sorted({
            info["encoding"]
            for info in encoding_infos
            if info["mime_type"] in self._DECODABLE_MIME_TYPES and info["encoding"] is not None
        })
        if len(plain_text_encodings) > 1:
            encodings_str = "', '".join(plain_text_encodings)
            logger.warning(
                f"The binary column '{column}' contains plain-text values with "
                f"multiple different character encodings: '{encodings_str}'. "
                f"Decoding will be skipped; binary values will be placed as-is."
            )
            return False

        if not plain_text_encodings:
            return False

        self.custom_schema.setdefault("encoding", {})[column] = plain_text_encodings[0]
        return True

    def _cast_binary_column(self, column: str) -> None:
        """
        Decode every binary value in `column` to a string using the detected
        character encoding, and record that encoding under
        `custom_schema["encoding"][column]` so downstream consumers
        can re-encode the synthetic output to bytes.

        When validation cannot determine a safe single encoding (non-decodable
        MIME, unknown encoding, or mixed encodings), decoding is skipped and
        the raw bytes are left as-is in the column.
        """
        encoding_infos = self._collect_encoding_infos(column)
        can_decode = self._validate_binary_encoding_infos(column, encoding_infos)
        if not can_decode:
            return
        encoding = self.custom_schema["encoding"][column]

        def _decode(value):
            try:
                return value.decode(encoding)
            except (UnicodeDecodeError, TypeError) as exc:
                message = (
                    f"Failed to decode a value in the binary column '{column}' "
                    f"using the encoding '{encoding}'. Underlying error: {exc}."
                )
                logger.error(message)
                raise ValueError(message) from exc

        self.preprocessed_df[column] = self.preprocessed_df[column].map(_decode)

    def _to_tuples_recursive(self, x):
        """
        Recursively convert nested lists into tuples (for PyArrow `map` data type).
        """
        # Handle null/NaN safely (scalars only)
        if self._is_null(x):
            return x

        # If it's a list-like collection, recurse into each element
        if isinstance(x, (list, np.ndarray)) and all(not isinstance(item, list) for item in x):
            return tuple(self._to_tuples_recursive(i) for i in x)

        # Base case: primitive value (str, int, float, None, ...)
        return x

    @staticmethod
    def _is_null(x):
        """Safely check for null without triggering ambiguous array truth errors."""
        if x is None:
            return True
        # Only call pd.isna on scalars; arrays/lists raise ValueError
        if np.isscalar(x):
            return pd.isna(x)
        return False

    def _cast_values_to_json(self):
        """
        Cast the values contained in columns with complex data types
        (`dict`, `list`, `tuple`, `numpy.ndarray`) to JSON strings. Binary columns
        are decoded to strings using the most popular detected encoding.
        """
        serializable_columns = self._get_serializable_columns_mapping().keys()
        for column in serializable_columns:
            if column not in self.preprocessed_df.columns:
                continue
            data_type = self.custom_schema["fields"].get(column)
            if data_type == "binary":
                self._cast_binary_column(column)
            else:
                self.preprocessed_df[column] = (
                    self.preprocessed_df[column].map(
                        lambda x: self._serialize_complex_value(x, column)
                    )
                )

    def _deserialize_values_from_json(self):
        """
        Deserialize JSON values to convert them to the original format
        """
        serializable_columns_mapping = self._get_serializable_columns_mapping()
        for column, data_type in serializable_columns_mapping.items():
            if column not in self.preprocessed_df.columns:
                continue
            if "struct" in data_type:
                self.preprocessed_df[column] = self.preprocessed_df[column].map(
                    lambda x: json.loads(x) if not pd.isna(x) else x
                )
            if "map" in data_type:
                self.preprocessed_df[column] = (
                    self.preprocessed_df[column].map(
                        lambda x: json.loads(x).get(f"{column}") if not pd.isna(x) else x
                    )
                )
                logger.warning(
                    f"Restoring the potentially nested structure of the column - '{column}' "
                    f"with the `map` data type. "
                    "The process of restoring the original structure of the column "
                    "is based on the assumption that the column contains lists of primitive values. "
                    "If this is not the case, there might be issues with the "
                    "consistency between the data and the provided schema."
                )
                self.preprocessed_df[column] = self.preprocessed_df[column].map(
                    lambda x: self._to_tuples_recursive(x) if not pd.isna(x) else x
                )
            if "list" in data_type:
                self.preprocessed_df[column] = (
                    self.preprocessed_df[column].map(
                        lambda x: json.loads(x).get(f"{column}") if not pd.isna(x) else x
                    )
                )

    def _postprocess_encoded_columns(self):
        """
        Postprocess encoded columns to return them to the original format
        """
        if self.custom_schema.get("encoding"):
            for column, encoding in self.custom_schema["encoding"].items():
                self.preprocessed_df[column] = self.preprocessed_df[column].map(
                    lambda x: x.encode(encoding) if not pd.isna(x) else x
                )

    def _preprocess_df(self):
        """
        Get the preprocessed data frame, update data types of columns
        """
        if not self.preprocessed_df.empty:
            try:
                self._set_none_values_to_nan()
                if self.custom_schema.get("format") == "CSV":
                    self._cast_values_to_string()
                    self._cast_columns_if_schema_is_not_provided()
                else:
                    self._cast_columns_to_schema_types()
                    self._cast_values_to_string()
                    if (
                        any(
                            "complex" in data_type
                            for data_type in self.custom_schema.get("fields", {}).values()
                        )
                        and self.serialize_complex_types is True
                    ):
                        self._cast_values_to_json()
                    if (
                        any(
                            "complex" in data_type
                            for data_type in self.custom_schema.get("fields", {}).values()
                        )
                        and self.serialize_complex_types is False
                    ):
                        self._postprocess_encoded_columns()
                        self._deserialize_values_from_json()
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

    def __init__(
        self,
        original_schema: Optional[Dict] = None,
        df: pd.DataFrame = None,
        serialize_complex_types: bool = True
    ):
        super().__init__(original_schema, df, serialize_complex_types)
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
