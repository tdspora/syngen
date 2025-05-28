from typing import Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, date

import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class Convertor:
    """
    Abstract class for converting fetched schema in Avro, Parquet or Delta formats
    """
    schema: Dict
    df: pd.DataFrame
    excluded_dtypes: Tuple = (str, bytes, datetime, date)

    def _check_dtype_or_nan(self, dtypes: Tuple):
        """
        Check if the value is of the specified data types or 'np.NaN'
        """
        return (
            lambda x: isinstance(x, dtypes)
            or (not isinstance(x, self.excluded_dtypes) and np.isnan(x))
        )

    def _update_data_types(self, schema: Dict, df: pd.DataFrame):
        """
        Update data types related to the fetched schema
        """
        for column, data_type in schema.get("fields", {}).items():
            if data_type in ["binary", "date", "string"]:
                df[column] = df[column].astype("string")
            elif data_type == "int":
                if any(df[column].isnull()):
                    df[column] = df[column].astype("float64")
                else:
                    df[column] = df[column].astype("int64")
            elif data_type == "null":
                if df[column].isnull().all():
                    continue
                else:
                    raise ValueError(
                        f"It seems that the data type - '{data_type}' "
                        f"isn\'t correct for the column - '{column}' as it's not empty"
                    )

        if not schema.get("fields"):
            for column in df.columns:
                if df[column].apply(lambda x: isinstance(x, int)).all():
                    df[column] = df[column].astype(int)
                elif df[column].apply(self._check_dtype_or_nan(dtypes=(int, float))).all():
                    df[column] = df[column].astype(float)
                elif df[column].apply(self._check_dtype_or_nan(dtypes=(str, bytes))).all():
                    df[column] = df[column].astype(pd.StringDtype())

    @staticmethod
    def _set_none_values_to_nan(df: pd.DataFrame):
        """
        Set 'None' values contained in columns as 'np.NaN'
        """
        df_object_subset = df.select_dtypes(["object"])
        for column in df_object_subset:
            df[column] = [
                np.NaN
                if i is None
                else i
                for i in df[column]
            ]
        return df

    def _cast_values_to_string(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cast the values contained in columns with the data type 'object'
        to 'string'
        """
        df_object_subset = df.select_dtypes(["object"])
        for column in df_object_subset:
            df[column] = [
                i
                if not isinstance(i, self.excluded_dtypes) and np.isnan(i)
                else str(i)
                for i in df[column]
            ]
        return df

    def _preprocess_df(self, schema: Dict, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data frame, update data types of columns
        """
        if not df.empty:
            try:
                df = self._set_none_values_to_nan(df)
                df = self._cast_values_to_string(df)
                self._update_data_types(schema, df)
                return df
            except Exception as e:
                logger.error(e)
                raise e
        else:
            return df


class CSVConvertor(Convertor):
    """
    Class for supporting custom schema for csv files
    """
    schema: Dict = {"fields": {}, "format": "CSV"}

    def __init__(self, df):
        schema = {"fields": {}, "format": "CSV"}
        super().__init__(schema, df)
        self.preprocessed_df = self._preprocess_df(schema, df)


class AvroConvertor(Convertor):
    """
    Class for converting fetched avro schema
    """

    def __init__(self, schema, df):
        super().__init__(schema, df)
        self.converted_schema = self._convert_schema(schema)
        self.preprocessed_df = self._preprocess_df(self.converted_schema, df)

    @staticmethod
    def _convert_schema(schema) -> Dict:
        """
        Convert the schema of Avro file to unified format, preprocess dataframe
        """
        converted_schema = dict()
        converted_schema["fields"] = dict()
        schema = schema if schema else dict()
        for column, data_type in schema.items():
            fields = converted_schema["fields"]
            if "int" in data_type or "long" in data_type or "boolean" in data_type:
                fields[column] = "int"
            elif "float" in data_type or "double" in data_type:
                fields[column] = "float"
            elif "string" in data_type:
                fields[column] = "string"
            elif "bytes" in data_type:
                fields[column] = "string"
            elif "null" in data_type:
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
