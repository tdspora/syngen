from typing import Dict
import re

from loguru import logger


class AvroSchemaConvertor:
    def __init__(self, schema):
        self.converted_schema = self._convert_schema(schema)

    @staticmethod
    def _convert_schema(schema) -> Dict:
        for column, data_type in schema.items():
            if 'int' in data_type or 'long' in data_type:
                schema[column] = 'int'
            elif 'float' in data_type or 'double' in data_type:
                schema[column] = 'float'
            elif 'string' in data_type:
                schema[column] = 'string'
            elif 'bytes' in data_type or 'boolean' in data_type:
                schema[column] = 'binary'
            else:
                message = "Not supported data type"
                logger.error(message)
                raise ValueError(message)
        return schema
