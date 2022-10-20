from typing import Dict

from schema import Schema, Optional, And, Or, SchemaError
from loguru import logger


configuration_schema = Schema({
    str: {
        Optional("train_settings"): {
            Optional("epochs"): And(int, lambda n: n >= 1),
            Optional("drop_null"): bool,
            Optional("row_limit"): And(int, lambda n: n >= 1)
        },
        Optional("infer_settings"): {
            Optional("size"): And(int, lambda n: n >= 1),
            Optional("run_parallel"): bool,
            Optional("random_seed"): int,
            Optional("print_report"): bool
        },
        "source": str,
        "keys": {
            str: {
                "type": Or("FK", "PK"),
                "columns": [str],
                Optional("references"): {
                    "table": str,
                    "columns": [str]
                }
            }
        }
    }
}
)


def validate_schema(configuration_schema: Schema, metadata: Dict):
    """
    Validate the metadata file regarding the configuration_schema
    :param configuration_schema: Schema
    :param metadata: Dict
    """
    try:
        configuration_schema.validate(metadata)
        logger.info("The schema of metadata file is valid")
    except SchemaError as err:
        logger.error(
            f"It seems that the schema of metadata file isn't valid. "
            f"The details of validation error - {err}")
        raise err
