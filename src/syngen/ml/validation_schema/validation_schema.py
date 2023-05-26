from typing import Dict, List

from schema import Schema, Optional, And, Or, SchemaError

from syngen.ml.custom_logger import custom_logger


def keys_schema(types_of_keys: List[str]):
    schema = {
        str: {
            "type": Or(*types_of_keys),
            "columns": [str],
            Optional("joined_sample"): bool
        }
    }
    if schema[str]["type"] == "FK":
        schema[str]["references"] = {"table": str, "columns": [str]}
    else:
        schema[str][Optional("references")] = {"table": str, "columns": [str]}
    return schema

configuration_schema = Schema({
    str: {
        Optional("train_settings"): Or(
            {
                Optional("epochs"): And(int, lambda n: n >= 1),
                Optional("drop_null"): bool,
                Optional("row_limit"): And(int, lambda n: n >= 1),
                Optional("batch_size"): And(int, lambda n: n >= 1),
                Optional("print_report"): bool,
                Optional("column_types"): {
                    Optional("categorical"): [str]
                }
            },
            None
        ),
        Optional("infer_settings"): Or(
            {
                Optional("size"): And(int, lambda n: n >= 1),
                Optional("run_parallel"): bool,
                Optional("batch_size"): And(int, lambda n: n >= 1),
                Optional("random_seed"): And(int, lambda n: n >= 0),
                Optional("print_report"): bool
            },
            None
        ),
        "source": str,
        Optional("keys"): Or(keys_schema(types_of_keys=["FK", "PK", "UQ"]), None)
    }
})


def validate_schema(configuration_schema: Schema, metadata: Dict):
    """
    Validate the metadata file regarding the configuration_schema
    :param configuration_schema: Schema
    :param metadata: Dict
    """
    try:
        configuration_schema.validate(metadata)
        custom_logger.info("The schema of metadata file is valid")
    except SchemaError as err:
        custom_logger.error(
            f"It seems that the schema of metadata file isn't valid. "
            f"The details of validation error - {err}")
        raise err
