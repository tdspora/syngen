from typing import Dict, Tuple
from datetime import datetime, date
import json

import pandas as pd
import numpy as np
from loguru import logger


class Convertor:
    """
    Abstract class for converting the fetched schema in Avro, Parquet or Delta formats
    """

    def __init__(self, schema: Dict, df: pd.DataFrame):
        self.schema = schema
        self.df = df
        self.excluded_dtypes: Tuple = (str, bytes, datetime, date, bool, list, dict)
        self.converted_schema = {}
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
        for column, data_type in self.converted_schema.get("fields", {}).items():
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
                # Preserve booleans as booleans.
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
        Cast DataFrame column types if the schema is not provided (the '.csv' format), 
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

    def _get_complex_columns(self):
        return [
            column
            for column, data_type in self.converted_schema.get("fields", {}).items()
            if data_type == "complex"
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

    @staticmethod
    def _serialize_complex_value(x):
        """
        Serialize complex value (dict, list, tuple, 'numpy.ndarray') to JSON string
        """
        if isinstance(x, np.ndarray):
            return json.dumps(x.tolist())
        elif isinstance(x, (dict, list, tuple)):
            return json.dumps(x)
        return x

    def _cast_values_to_json(self):
        """
        Cast the values contained in columns with complex data types
        (dict, list, tuple, 'numpy.ndarray') to JSON strings
        """
        for column in self._get_complex_columns():
            if column in self.preprocessed_df.columns:
                self.preprocessed_df[column] = (
                    self.preprocessed_df[column].apply(self._serialize_complex_value)
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
                if self.schema.get("format") == "CSV":
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
    schema: Dict = {"fields": {}, "format": "CSV"}

    def __init__(self, df):
        schema = {"fields": {}, "format": "CSV"}
        super().__init__(schema, df)
        self._preprocess_df()


class AvroConvertor(Convertor):
    """
    Class for converting the fetched avro schema
    """

    def __init__(self, schema, df):
        super().__init__(schema, df)
        self.schema = schema or {}
        self.complex_types = {"array", "map", "record", "enum", "fixed"}
        self.converted_schema = self._get_converted_schema()
        self._preprocess_df()

    def _get_converted_schema(self) -> Dict:
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

        converted_schema = dict()
        converted_schema["fields"] = dict()
        for column, data_type in self.schema.items():
            fields = converted_schema["fields"]
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
        converted_schema["format"] = "Avro"
        return converted_schema
