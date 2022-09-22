from typing import Dict

from schema import Schema, Optional, Or, SchemaError
from loguru import logger


config_schema = Schema({
    "configuration": {
        "tables": [
            {
                str: {
                    "table_name": str,
                    Optional("train_settings"): {
                        Optional("epochs"): int,
                        Optional("dropna"): bool,
                        Optional("row_limit"): int
                    },
                    Optional("infer_settings"): {
                        Optional("size"): int,
                        Optional("run_parallel"): bool,
                        Optional("random_seed"): int,
                        Optional("print_report"): bool
                    },
                    "path": str,
                    "keys": [
                        {
                            str: {
                                "type": Or("FK", "PK"),
                                "columns": [str],
                                "references": {
                                    "table": str,
                                    "columns": [str]
                                }
                            }
                        }
                    ]
                }
            }
        ]
    }
}
)


def validate_schema(metadata: Dict):
    try:
        config_schema.validate(metadata)
        logger.info("The schema of metadata file is valid")
    except SchemaError as err:
        logger.error(
            f"It seems that the schema of metadata file isn't valid. "
            f"The details of validation error - {err}")
        raise err

