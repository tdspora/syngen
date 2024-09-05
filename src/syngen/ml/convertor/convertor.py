from typing import Dict, Tuple
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from loguru import logger


class Convertor(ABC):
    """
    Abstract class for converting fetched schema in Avro, Parquet or Delta formats
    """

    def __init__(self, schema, df):
        self.converted_schema, self.preprocessed_df = self._convert_schema_and_df(schema, df)

    @abstractmethod
    def _convert_schema_and_df(self, schema: Dict, df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        """
        Convert the schema of file to unified format, preprocess dataframe
        """
        pass

    @staticmethod
    def _update_data_types(schema: Dict, df: pd.DataFrame):
        """
        Update data types related to the fetched schema
        """
        for column, data_type in schema.get("fields", {}).items():
            if data_type in ["binary", "date", "string"]:
                print("!!!!!!!!!!!!!!!!!!")
                print(column)
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

    @staticmethod
    def _cast_values_to_string(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cast the values contained in columns with the data type 'object'
        to 'string'
        """
        df_object_subset = df.select_dtypes(["object"])
        for column in df_object_subset:
            df[column] = [
                i
                if not isinstance(i, str) and np.isnan(i)
                else str(i)
                for i in df[column]
            ]
        return df

    def _preprocess_df(self, schema: Dict, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data frame, update data types of columns
        """
        if not df.empty and schema["format"] != "CSV":
            try:
                self._update_data_types(schema, df)
            except Exception as e:
                logger.error(e)
                raise e
            else:
                df = self._set_none_values_to_nan(df)
                df = self._cast_values_to_string(df)
                return df
        else:
            return df


class CSVConvertor(Convertor):
    """
    Class for supporting custom schema for csv files
    """

    df: pd.DataFrame()
    schema = {"fields": {}, "format": "CSV"}

    def __init__(self, schema, df):
        super().__init__(schema, df)

    def _convert_schema_and_df(self, schema, df) -> Tuple[Dict, pd.DataFrame]:
        preprocessed_df = self._preprocess_df(schema, df)
        return schema, preprocessed_df


class AvroConvertor(Convertor):
    """
    Class for converting fetched avro schema
    """

    def __init__(self, schema, df):
        super().__init__(schema, df)

    def _convert_schema_and_df(self, schema, df) -> Tuple[Dict, pd.DataFrame]:
        """
        Convert the schema of Avro file to unified format, preprocess dataframe
        """
        converted_schema = dict()
        converted_schema["fields"] = dict()
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
        preprocessed_df = self._preprocess_df(converted_schema, df)
        return converted_schema, preprocessed_df
