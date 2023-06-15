from typing import Dict, List

from schema import Schema, Optional, And, Or, SchemaError

from syngen.ml.custom_logger import custom_logger


def build_keys_schema(types_of_keys: List[str]):
    """
    Build the schema for the field 'keys' in metadata file
    """
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


def build_configuration_schema() -> Schema:
    """
    Build the configuration_schema for validation of metadata file
    """
    training_settings = {
        Optional("epochs"): And(int, lambda n: n >= 1),
        Optional("drop_null"): bool,
        Optional("row_limit"): And(int, lambda n: n >= 1),
        Optional("batch_size"): And(int, lambda n: n >= 1),
        Optional("print_report"): bool,
    }
    infer_settings = {
        Optional("size"): And(int, lambda n: n >= 1),
        Optional("run_parallel"): bool,
        Optional("batch_size"): And(int, lambda n: n >= 1),
        Optional("random_seed"): And(int, lambda n: n >= 0),
        Optional("print_report"): bool
    }
    config_schema = Schema({
        Optional("global"): {
            Optional("train_settings"): Or(training_settings, None),
            Optional("infer_settings"): Or(infer_settings, None),
        },
        str: {
            Optional("train_settings"): Or(
                {
                    **training_settings,
                    Optional("column_types"): {
                        Optional("categorical"): [str]
                    }
                },
                None
            ),
            Optional("infer_settings"): Or(infer_settings, None),
            "source": str,
            Optional("keys"): Or(build_keys_schema(types_of_keys=["FK", "PK", "UQ"]), None),
        }
    })

    return config_schema


def validate_schema(metadata: Dict):
    """
    Validate the metadata file regarding the configuration_schema
    """
    try:
        build_configuration_schema().validate(metadata)
        custom_logger.info("The schema of metadata file is valid")
    except SchemaError as err:
        custom_logger.error(
            f"It seems that the schema of metadata file isn't valid. "
            f"The details of validation error - {err}")
        raise err
