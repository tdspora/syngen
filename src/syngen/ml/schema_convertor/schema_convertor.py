from typing import Dict
from abc import ABC

from loguru import logger


class SchemaConvertor(ABC):
    """Abstract class for converting fetched schema in Avro, Parquet or Delta formats"""
    def __init__(self, schema):
        self.converted_schema = self._convert_schema(schema)

    @staticmethod
    def _convert_schema(schema):
        pass


class AvroSchemaConvertor(SchemaConvertor):
    """
    Class for converting fetched avro schema
    """
    def __init__(self, schema):
        super().__init__(schema)

    @staticmethod
    def _convert_schema(schema) -> Dict:
        converted_schema = dict()
        converted_schema["fields"] = dict()
        for column, data_type in schema.items():
            fields = converted_schema["fields"]
            if 'int' in data_type or 'long' in data_type:
                fields[column] = 'int'
            elif 'float' in data_type or 'double' in data_type:
                fields[column] = 'float'
            elif 'string' in data_type:
                fields[column] = 'string'
            elif 'bytes' in data_type or 'boolean' in data_type:
                fields[column] = 'binary'
            else:
                message = "Not supported data type"
                logger.error(message)
                raise ValueError(message)
        converted_schema["format"] = "Avro"
        return converted_schema
