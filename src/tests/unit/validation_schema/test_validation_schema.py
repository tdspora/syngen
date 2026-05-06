import pytest

from marshmallow import ValidationError

from syngen.ml.validation_schema import ValidationMetadataSchema
from syngen.ml.data_loaders import MetadataLoader

from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


def test_validation_of_metadata_file(rp_logger, caplog):
    rp_logger.info("Test the validation of the schema of the valid metadata file")
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "type_of_process, reports", [
        ("train", []),
        ("infer", []),
        ("train", ["accuracy", "sample"]),
        ("infer", ["accuracy"]),
        ("train", ["accuracy"]),
        ("infer", ["accuracy"]),
        ("train", ["sample"]),
        ("train", ["metrics_only"]),
        ("infer", ["metrics_only"]),
        ("train", ["accuracy", "metrics_only"]),
        ("infer", ["accuracy", "metrics_only"]),
        ("train", ["sample", "metrics_only"])
    ]
)
def test_validation_of_metadata_file_with_diff_types_of_reports(
    type_of_process, reports, rp_logger, caplog
):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        f"with reports - {', '.join(reports)} during the {type_of_process} process"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["global"]["train_settings"]["reports"] = reports
    with caplog.at_level(level="DEBUG"):
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_with_source_contained_path_to_excel_table(
    rp_logger, caplog
):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        "contained the parameter 'source' which is the path to the excel table"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file_for_excel_table.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_without_global_settings(rp_logger, caplog):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        "with the absent 'global' settings"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/"
        f"valid_metadata_file_with_absent_global_settings.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_only_with_required_fields(rp_logger, caplog):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        "with only provided 'source' fields during the training process "
        "when the 'loader' isn't provided."
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/"
        f"valid_metadata_file_only_with_required_fields.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "wrong_setting, expected_error",
    [
        (
            {"source": 0},
            'The error(s) found in - "fk_test": {\n    "train_settings": '
            '{\n        "source": '
            '[\n            "Not a valid string."\n        ]\n    }\n}'
        ),
        (
            {"epochs": 0},
            'The error(s) found in - "fk_test": {\n    "train_settings": '
            '{\n        "epochs": '
            '[\n            "Must be greater than or equal to 1."\n        ]\n    }\n}',
        ),
        (
            {"epochs": "not a valid type of a value"},
            'The error(s) found in - "fk_test": {\n    "train_settings": '
            '{\n        "epochs": '
            '[\n            "Not a valid integer."\n        ]\n    }\n}',
        ),
        (
            {"drop_null": "not a valid type of a value"},
            'The error(s) found in - "fk_test": {\n    "train_settings": '
            '{\n        "drop_null": '
            '[\n            "Not a valid boolean."\n        ]\n    }\n}',
        ),
        (
            {"row_limit": 0},
            'The error(s) found in - "fk_test": {\n    "train_settings": '
            '{\n        "row_limit": '
            '[\n            "Must be greater than or equal to 1."\n        ]\n    }\n}',
        ),
        (
            {"row_limit": "not a valid type of a value"},
            'The error(s) found in - "fk_test": {\n    "train_settings": '
            '{\n        "row_limit": '
            '[\n            "Not a valid integer."\n        ]\n    }\n}',
        ),
        (
            {"batch_size": 0},
            'The error(s) found in - "fk_test": {\n    "train_settings": '
            '{\n        "batch_size": '
            '[\n            "Must be greater than or equal to 1."\n        ]\n    }\n}',
        ),
        (
            {"batch_size": "not a valid type of a value"},
            'The error(s) found in - "fk_test": {\n    "train_settings": '
            '{\n        "batch_size": '
            '[\n            "Not a valid integer."\n        ]\n    }\n}',
        ),
        (
            {"reports": "not a valid type of a value"},
            'The error(s) found in - "fk_test": {\n    "train_settings": '
            '{\n        "reports": '
            '[\n            "Invalid value."\n        ]\n    }\n}',
        ),
        (
            {"column_types": {"invalid_type": ["column_1", "column_2"]}},
            'The error(s) found in - "fk_test": {\n    "train_settings": '
            '{\n        "column_types": {\n            "invalid_type": '
            '{\n                "key": '
            '[\n                    "Must be one of: categorical."'
            '\n                ]\n            }\n        }\n    }\n}',
        ),
    ],
)
def test_validation_of_metadata_file_with_invalid_training_settings(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info(
        "Test the validation of the schema of the metadata with the invalid training settings"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["fk_test"]["train_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert (
        str(error.value)
        == f"Validation error(s) found in the schema of the metadata. {expected_error}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "wrong_setting, expected_error",
    [
        (
            {"epochs": 0},
            'The error(s) found in - "global": {\n    "train_settings": '
            '{\n        "epochs": [\n            "Must be greater than or equal to 1."'
            '\n        ]\n    }\n}'
        ),
        (
            {"epochs": "not a valid type of a value"},
            'The error(s) found in - "global": {\n    "train_settings": '
            '{\n        "epochs": [\n            "Not a valid integer."'
            '\n        ]\n    }\n}',
        ),
        (
            {"drop_null": "not a valid type of a value"},
            'The error(s) found in - "global": {\n    "train_settings": '
            '{\n        "drop_null": [\n            "Not a valid boolean."'
            '\n        ]\n    }\n}',
        ),
        (
            {"row_limit": 0},
            'The error(s) found in - "global": {\n    "train_settings": '
            '{\n        "row_limit": [\n            "Must be greater than or equal to 1."'
            '\n        ]\n    }\n}',
        ),
        (
            {"row_limit": "not a valid type of a value"},
            'The error(s) found in - "global": {\n    "train_settings": '
            '{\n        "row_limit": [\n            "Not a valid integer."'
            '\n        ]\n    }\n}',
        ),
        (
            {"batch_size": 0},
            'The error(s) found in - "global": {\n    "train_settings": '
            '{\n        "batch_size": [\n            "Must be greater than or equal to 1."'
            '\n        ]\n    }\n}',
        ),
        (
            {"batch_size": "not a valid type of a value"},
            'The error(s) found in - "global": {\n    "train_settings": '
            '{\n        "batch_size": [\n            "Not a valid integer."'
            '\n        ]\n    }\n}',
        ),
        (
            {"reports": "not a valid type of a value"},
            'The error(s) found in - "global": {\n    "train_settings": '
            '{\n        "reports": [\n            "Invalid value."'
            '\n        ]\n    }\n}',
        ),
    ],
)
def test_validation_of_metadata_file_with_invalid_global_training_settings(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info(
        "Test the validation of the schema of the metadata "
        "with the invalid global training settings"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["global"]["train_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert (
        str(error.value)
        == f"Validation error(s) found in the schema of the metadata. {expected_error}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "wrong_setting, expected_error",
    [
        (
            {"destination": 0},
            'The error(s) found in - "fk_test": {\n    "infer_settings": '
            '{\n        "destination": [\n            "Not a valid string."'
            '\n        ]\n    }\n}',
        ),
        (
            {"size": 0},
            'The error(s) found in - "fk_test": {\n    "infer_settings": '
            '{\n        "size": [\n            "Must be greater than or equal to 1."'
            '\n        ]\n    }\n}',
        ),
        (
            {"size": "not a valid type of a value"},
            'The error(s) found in - "fk_test": {\n    "infer_settings": '
            '{\n        "size": [\n            "Not a valid integer."'
            '\n        ]\n    }\n}',
        ),
        (
            {"run_parallel": "not a valid type of a value"},
            'The error(s) found in - "fk_test": {\n    "infer_settings": '
            '{\n        "run_parallel": [\n            "Not a valid boolean."'
            '\n        ]\n    }\n}',
        ),
        (
            {"random_seed": -1},
            'The error(s) found in - "fk_test": {\n    "infer_settings": '
            '{\n        "random_seed": [\n            "Must be greater than or equal to 0."'
            '\n        ]\n    }\n}',
        ),
        (
            {"random_seed": "not a valid type of a value"},
            'The error(s) found in - "fk_test": {\n    "infer_settings": '
            '{\n        "random_seed": [\n            "Not a valid integer."'
            '\n        ]\n    }\n}',
        ),
        (
            {"reports": "not a valid type of a value"},
            'The error(s) found in - "fk_test": {\n    "infer_settings": '
            '{\n        "reports": [\n            "Invalid value."'
            '\n        ]\n    }\n}',
        ),
    ],
)
def test_validation_of_metadata_file_with_invalid_infer_settings(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info("Test the validation of the metadata with the invalid infer settings")
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["fk_test"]["infer_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="infer"
        ).validate_schema()
    assert (
        str(error.value)
        == f"Validation error(s) found in the schema of the metadata. {expected_error}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "wrong_setting, expected_error",
    [
        (
            {"size": 0},
            'The error(s) found in - "global": {\n    "infer_settings": '
            '{\n        "size": [\n            "Must be greater than or equal to 1."'
            '\n        ]\n    }\n}',
        ),
        (
            {"size": "not a valid type of a value"},
            'The error(s) found in - "global": {\n    "infer_settings": '
            '{\n        "size": [\n            "Not a valid integer."'
            '\n        ]\n    }\n}',
        ),
        (
            {"run_parallel": "not a valid type of a value"},
            'The error(s) found in - "global": {\n    "infer_settings": '
            '{\n        "run_parallel": [\n            "Not a valid boolean."'
            '\n        ]\n    }\n}',
        ),
        (
            {"random_seed": -1},
            'The error(s) found in - "global": {\n    "infer_settings": '
            '{\n        "random_seed": [\n            "Must be greater than or equal to 0."'
            '\n        ]\n    }\n}',
        ),
        (
            {"random_seed": "not a valid type of a value"},
            'The error(s) found in - "global": {\n    "infer_settings": '
            '{\n        "random_seed": [\n            "Not a valid integer."'
            '\n        ]\n    }\n}',
        ),
        (
            {"reports": "not a valid type of a value"},
            'The error(s) found in - "global": {\n    "infer_settings": '
            '{\n        "reports": [\n            "Invalid value."'
            '\n        ]\n    }\n}',
        ),
    ],
)
def test_validation_of_metadata_file_with_invalid_global_infer_settings(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info(
        "Test the validation of the schema of the metadata with the invalid global infer settings"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["global"]["infer_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="infer"
        ).validate_schema()
    assert (
        str(error.value)
        == f"Validation error(s) found in the schema of the metadata. {expected_error}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "wrong_setting, expected_error",
    [
        (
            {"fernet_key": 1},
            'The error(s) found in - "global": {\n    "encryption": '
            '{\n        "fernet_key": [\n            "Not a valid string."'
            '\n        ]\n    }\n}'
        )
    ],
)
def test_validation_of_metadata_file_with_invalid_global_encryption_settings(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info(
        "Test the validation of the schema of the metadata "
        "with the invalid global encryption settings"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["global"]["encryption"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert (
        str(error.value)
        == f"Validation error(s) found in the schema of the metadata. {expected_error}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "wrong_setting, expected_error",
    [
        (
            {"fernet_key": 1},
            'The error(s) found in - "fk_test": {\n    "encryption": '
            '{\n        "fernet_key": [\n            "Not a valid string."'
            '\n        ]\n    }\n}'
        )
    ],
)
def test_validation_of_metadata_file_with_invalid_encryption_settings(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info(
        "Test the validation of the schema of the metadata with the invalid encryption settings"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["fk_test"]["encryption"] = wrong_setting
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert (
        str(error.value)
        == f"Validation error(s) found in the schema of the metadata. {expected_error}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "wrong_setting, expected_error",
    [
        (
            {"sep": 0},
            'The error(s) found in - "fk_test": {\n    "sep": '
            '[\n        "Not a valid string."\n    ]\n}'
        ),
        (
            {"quotechar": 0},
            'The error(s) found in - "fk_test": {\n    "quotechar": '
            '[\n        "Not a valid string."\n    ]\n}',
        ),
        (
            {"quotechar": "value with more than one character"},
            'The error(s) found in - "fk_test": {\n    "quotechar": '
            '[\n        "Length must be 1."\n    ]\n}',
        ),
        (
            {"quoting": 0},
            'The error(s) found in - "fk_test": {\n    "quoting": '
            '[\n        "Not a valid string."\n    ]\n}',
        ),
        (
            {"quoting": "not a valid value"},
            'The error(s) found in - "fk_test": {\n    "quoting": '
            '[\n        "Must be one of: minimal, all, non-numeric, none."\n    ]\n}',
        ),
        (
            {"escapechar": 0},
            'The error(s) found in - "fk_test": {\n    "escapechar": '
            '[\n        "Not a valid string."\n    ]\n}',
        ),
        (
            {"escapechar": "value with more than one character"},
            'The error(s) found in - "fk_test": {\n    "escapechar": '
            '[\n        "Length must be 1."\n    ]\n}',
        ),
        (
            {"encoding": 0},
            'The error(s) found in - "fk_test": {\n    "encoding": '
            '[\n        "Not a valid string."\n    ]\n}',
        ),
        (
            {"header": "not a valid type of a value"},
            'The error(s) found in - "fk_test": {\n    "header": '
            '[\n        "Invalid value."\n    ]\n}',
        ),
        (
            {"skiprows": "not a valid type of a value"},
            'The error(s) found in - "fk_test": {\n    "skiprows": '
            '[\n        "Invalid value."\n    ]\n}',
        ),
        (
            {"on_bad_lines": 0},
            'The error(s) found in - "fk_test": {\n    "on_bad_lines": '
            '[\n        "Not a valid string."\n    ]\n}',
        ),
        (
            {"on_bad_lines": "not a valid value"},
            'The error(s) found in - "fk_test": {\n    "on_bad_lines": '
            '[\n        "Must be one of: error, warn, skip."\n    ]\n}',
        ),
        (
            {"engine": 0},
            'The error(s) found in - "fk_test": {\n    "engine": '
            '[\n        "Not a valid string."\n    ]\n}',
        ),
        (
            {"engine": "not a valid value"},
            'The error(s) found in - "fk_test": {\n    "engine": '
            '[\n        "Must be one of: c, python."\n    ]\n}',
        ),
        (
            {"sheet_name": 0},
            'The error(s) found in - "fk_test": {\n    "sheet_name": '
            '[\n        "Unknown field."\n    ]\n}',
        ),
    ],
)
def test_validation_of_metadata_file_with_invalid_format_settings_for_csv_table(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info(
        "Test the validation of the schema of the metadata "
        "with the format settings set to the CSV file"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["fk_test"]["format"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert (
        str(error.value)
        == f"Validation error(s) found in the schema of the metadata. {expected_error}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "wrong_setting, expected_error",
    [
        (
            {"sep": ","},
            'The error(s) found in - "pk_test": {\n    "sep": '
            '[\n        "Unknown field."\n    ]\n}'
        ),
        (
            {"quotechar": '"'},
            'The error(s) found in - "pk_test": {\n    "quotechar": '
            '[\n        "Unknown field."\n    ]\n}',
        ),
        (
            {"quoting": "non-numeric"},
            'The error(s) found in - "pk_test": {\n    "quoting": '
            '[\n        "Unknown field."\n    ]\n}',
        ),
        (
            {"escapechar": "\\"},
            'The error(s) found in - "pk_test": {\n    "escapechar": '
            '[\n        "Unknown field."\n    ]\n}',
        ),
        (
            {"encoding": "ascii"},
            'The error(s) found in - "pk_test": {\n    "encoding": '
            '[\n        "Unknown field."\n    ]\n}',
        ),
        (
            {"header": 0},
            'The error(s) found in - "pk_test": {\n    "header": '
            '[\n        "Unknown field."\n    ]\n}',
        ),
        (
            {"skiprows": 0},
            'The error(s) found in - "pk_test": {\n    "skiprows": '
            '[\n        "Unknown field."\n    ]\n}',
        ),
        (
            {"on_bad_lines": "skip"},
            'The error(s) found in - "pk_test": {\n    "on_bad_lines": '
            '[\n        "Unknown field."\n    ]\n}',
        ),
        (
            {"engine": "python"},
            'The error(s) found in - "pk_test": {\n    "engine": '
            '[\n        "Unknown field."\n    ]\n}'
        ),
    ],
)
def test_validation_of_metadata_file_with_invalid_format_settings_for_excel_table(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info(
        "Test the validation of the schema of the metadata "
        "with the format settings set to the Excel table"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file_for_excel_table.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["pk_test"]["format"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert (
        str(error.value)
        == f"Validation error(s) found in the schema of the metadata. {expected_error}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_without_source_fields_in_train_without_loader(rp_logger):
    rp_logger.info(
        "Test the validation of the schema of the metadata file "
        "without 'source' field in the training process if the loader isn't provided."
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/metadata_file_without_required_fields.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert str(error.value) == (
        'Validation error(s) found in the schema of the metadata. '
        'The error(s) found in - "pk_test": {\n    "train_settings": '
        '{\n        "source": [\n            "Missing data for required field."'
        '\n        ]\n    }\n}'
        'The error(s) found in - "fk_test": {\n    "train_settings": '
        '{\n        "source": [\n            "Missing data for required field."'
        '\n        ]\n    }\n}'
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_without_source_fields_in_train_with_loader(
    caplog, rp_logger
):
    rp_logger.info(
        "Test the validation of the schema of the metadata file "
        "without 'source' field in the training process if the loader is provided."
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/metadata_file_without_required_fields.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=False,
            process="train"
        ).validate_schema()
    assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_with_source_and_loader_in_train(rp_logger):
    rp_logger.info(
        "Test the validation of the schema of the metadata file "
        "with the provided 'source' and 'loader' in the training process."
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=False,
            process="train"
        ).validate_schema()
    assert str(error.value) == (
        'Validation error(s) found in the schema of the metadata. '
        'The error(s) found in - "pk_test": {\n    "train_settings": '
        '{\n        "_schema": [\n            '
        '"The \'source\' field is not allowed when the \'loader\' parameter is provided. '
        'Please, review your metadata file."\n        ]\n    }\n}'
        'The error(s) found in - "fk_test": {\n    "train_settings": '
        '{\n        "_schema": [\n            '
        '"The \'source\' field is not allowed when the \'loader\' parameter is provided. '
        'Please, review your metadata file."\n        ]\n    }\n}'
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("validation_of_source", [True, False])
def test_validation_of_metadata_file_without_source_fields_in_infer(
    validation_of_source, caplog, rp_logger
):
    rp_logger.info(
        "Test the validation of the schema of the metadata file "
        "without 'source' field in the inference process"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/metadata_file_without_required_fields.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=validation_of_source,
            process="infer"
        ).validate_schema()
    assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_without_training_settings_during_train_process_with_loader(
    rp_logger, caplog
):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        "during the training process without provided 'training_settings' "
        "in case the loader is provided"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/metadata_file_without_training_settings.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=False,
            process="train"
        ).validate_schema()
    assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_without_training_settings_during_train_process_without_loader(
    rp_logger
):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        "during the training process without provided 'training_settings' "
        "in case the loader isn't provided"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/metadata_file_without_training_settings.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with pytest.raises(ValidationError) as e:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert str(e.value) == (
        'Validation error(s) found in the schema of the metadata. '
        'The error(s) found in - "pk_test": {\n    "train_settings": '
        '{\n        "source": '
        '[\n            "Missing data for required field."\n        ]\n    }\n}'
        'The error(s) found in - "fk_test": {\n    "train_settings": '
        '{\n        "source": '
        '[\n            "Missing data for required field."\n        ]\n    }\n}'
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("validation_of_source", [True, False])
def test_validation_of_metadata_file_without_training_settings_during_infer_process(
    validation_of_source, rp_logger, caplog
):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        "during the inference process without provided 'training_settings' "
        "in case the loader is provided"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/metadata_file_without_training_settings.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=validation_of_source,
            process="infer"
        ).validate_schema()
    assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_with_invalid_PK_key(rp_logger):
    rp_logger.info(
        "Test the validation of the schema of the metadata file "
        "with the invalid PK which contained 'references' section"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/metadata_file_with_invalid_PK_key.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert str(error.value) == (
        'Validation error(s) found in the schema of the metadata. '
        'The error(s) found in - "pk_test": {\n    "keys": '
        '{\n        "pk_test_pk_id": {\n            "value": '
        '{\n                "_schema": '
        '[\n                    "The \'references\' field is only allowed when \'type\' is \'FK\'"'
        '\n                ]\n            }\n        }\n    }\n}'
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_with_invalid_UQ_key(rp_logger):
    rp_logger.info(
        "Test the validation of the schema of the metadata file "
        "with the invalid UQ contained 'references' section"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/metadata_file_with_invalid_UQ_key.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert str(error.value) == (
        'Validation error(s) found in the schema of the metadata. '
        'The error(s) found in - "fk_test": '
        '{\n    "keys": {\n        "fk_test_uq_name": {\n            "value": '
        '{\n                "_schema": '
        '[\n                    "The \'references\' field is only allowed when \'type\' is \'FK\'"'
        '\n                ]\n            }\n        }\n    }\n}'
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_with_invalid_FK_key(rp_logger):
    rp_logger.info(
        "Test the validation of the schema of the metadata file "
        "with the invalid FK which doesn't contain 'references' section"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/metadata_file_with_invalid_FK_key.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert str(error.value) == (
        'Validation error(s) found in the schema of the metadata. '
        'The error(s) found in - "fk_test": {\n    "keys": '
        '{\n        "fk_test_fk_id": {\n            "value": '
        '{\n                "_schema": '
        '[\n                    "The \'references\' field is required when \'type\' is \'FK\'"'
        '\n                ]\n            }\n        }\n    }\n}'
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "path_to_metadata, expected_error",
    [
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_absent_pk_columns.yaml",
            'The error(s) found in - "table_a": {\n    "keys": {\n        "pk_id": '
            '{\n            "value": {\n                "columns": '
            '[\n                    "Field may not be null."\n                ]'
            '\n            }\n        }\n    }\n}',
        ),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_absent_fk_columns.yaml",
            'The error(s) found in - "table_b": {\n    "keys": {\n        "fk_id": '
            '{\n            "value": {\n                "columns": '
            '[\n                    "Missing data for required field."\n                ]'
            '\n            }\n        }\n    }\n}'
        ),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_absent_ref_columns.yaml",
            'The error(s) found in - "table_b": {\n    "keys": {\n        "fk_id": '
            '{\n            "value": {\n                "references": '
            '{\n                    "columns": '
            '[\n                        "Missing data for required field."'
            '\n                    ]\n                }'
            '\n            }\n        }\n    }\n}'),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_duplicated_pk_columns.yaml",
            'The error(s) found in - "table_a": {\n    "keys": {\n        "pk_id": '
            '{\n            "value": {\n                "_schema": '
            '[\n                    "The \'columns\' field must contain unique values"'
            '\n                ]\n            }\n        }\n    }\n}',
        ),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_duplicated_fk_columns.yaml",
            'The error(s) found in - "table_b": {\n    "keys": {\n        "fk_id": '
            '{\n            "value": {\n                "_schema": '
            '[\n                    "The \'columns\' field must contain unique values"'
            '\n                ]\n            }\n        }\n    }\n}',
        ),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_absent_uq_columns.yaml",
            'The error(s) found in - "table_a": {\n    "keys": {\n        "uq_id": '
            '{\n            "value": {\n                "columns": '
            '[\n                    "Field may not be null."\n                ]'
            '\n            }\n        }\n    }\n}',
        ),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_duplicated_uq_columns.yaml",
            'The error(s) found in - "table_a": {\n    "keys": {\n        "uq_id": '
            '{\n            "value": {\n                "_schema": '
            '[\n                    "The \'columns\' field must contain unique values"'
            '\n                ]\n            }\n        }\n    }\n}',
        ),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_diff_length_of_columns.yaml",
            'The error(s) found in - "table_b": {\n    "keys": {\n        "fk_id": '
            '{\n            "value": {\n                "_schema": '
            '[\n                    '
            '"The \'columns\' field must have the same length as \'references.columns\'"'
            '\n                ]\n            }\n        }\n    }\n}',
        ),
    ],
)
def test_validation_schema_of_keys(path_to_metadata, expected_error, rp_logger):
    rp_logger.info(
        "Test the validation of the schema of the metadata file with the invalid section 'keys'"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert str(error.value) == (
        f"Validation error(s) found in the schema of the metadata. {expected_error}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_with_regex_patterns(rp_logger, caplog):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        "with 'regex_patterns' in PK and UQ keys"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file_with_regex_patterns.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_with_regex_patterns_in_pk_key(rp_logger, caplog):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        "with 'regex_patterns' added to the PK key"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["pk_test"]["keys"]["pk_test_pk_id"]["regex_patterns"] = {
        "Id": "^[A-Z]{3}-[0-9]{6}$"
    }
    with caplog.at_level(level="DEBUG"):
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_with_regex_patterns_in_uq_key(rp_logger, caplog):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        "with 'regex_patterns' added to the UQ key"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["fk_test"]["keys"]["fk_test_uq_name"]["regex_patterns"] = {
        "Name": "[A-Z][a-z]+"
    }
    with caplog.at_level(level="DEBUG"):
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_with_regex_patterns_set_to_none(rp_logger, caplog):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        "with 'regex_patterns' explicitly set to 'None'"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["pk_test"]["keys"]["pk_test_pk_id"]["regex_patterns"] = None
    with caplog.at_level(level="DEBUG"):
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_with_regex_patterns_in_fk_key(rp_logger):
    rp_logger.info(
        "Test the validation of the schema of the metadata file "
        "with 'regex_patterns' in FK key which is not allowed"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["fk_test"]["keys"]["fk_test_fk_id"]["regex_patterns"] = {"Id": "[0-9]+"}
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert str(error.value) == (
        'Validation error(s) found in the schema of the metadata. '
        'The error(s) found in - "fk_test": {\n    "keys": '
        '{\n        "fk_test_fk_id": {\n            "value": '
        '{\n                "_schema": '
        '[\n                    '
        '"The \'regex\' field is only allowed when \'type\' is \'PK\' or \'UQ\'. '
        'Got: \'FK\'."\n                ]\n            }\n        }\n    }\n}'
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_with_regex_patterns_for_nonexistent_column(rp_logger):
    rp_logger.info(
        "Test the validation of the schema of the metadata file "
        "with 'regex_patterns' referencing a column not in 'columns'"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["pk_test"]["keys"]["pk_test_pk_id"]["regex_patterns"] = {
        "nonexistent_col": "[0-9]+"
    }
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert (
        "The column 'nonexistent_col' specified in the 'regex' field does not exist "
        "in the 'columns' field. Available columns: ['Id']."
    ) in str(error.value)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_with_invalid_regex_expression(rp_logger):
    rp_logger.info(
        "Test the validation of the schema of the metadata file "
        "with an invalid regular expression in 'regex_patterns'"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["pk_test"]["keys"]["pk_test_pk_id"]["regex_patterns"] = {"Id": "[invalid"}
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert str(error.value) == (
        'Validation error(s) found in the schema of the metadata. '
        'The error(s) found in - "pk_test": {\n    "keys": '
        '{\n        "pk_test_pk_id": {\n            "value": '
        '{\n                "regex_patterns": {\n                    "Id": '
        '{\n                        "value": '
        '[\n                            '
        '"The regex pattern \'[invalid\' for the column None is not a valid regular expression. '
        'Details: unterminated character set at position 0."\n                        ]'
        '\n                    }\n                }\n            }\n        }\n    }\n}'
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validation_of_metadata_file_with_empty_string_regex_pattern(rp_logger):
    rp_logger.info(
        "Test the validation of the schema of the metadata file "
        "with an empty string as a regex pattern value"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["pk_test"]["keys"]["pk_test_pk_id"]["regex_patterns"] = {"Id": ""}
    with pytest.raises(ValidationError) as error:
        ValidationMetadataSchema(
            metadata=metadata,
            validation_of_source=True,
            process="train"
        ).validate_schema()
    assert str(error.value) == (
         'Validation error(s) found in the schema of the metadata. '
         'The error(s) found in - "pk_test": {\n    "keys": '
         '{\n        "pk_test_pk_id": {\n            "value": '
         '{\n                "regex_patterns": {\n                    "Id": '
         '{\n                        "value": '
         '[\n                            '
         '"The regex pattern for the column None must be a non-empty string. '
         'Got: \'\'."\n                        ]'
         '\n                    }\n                }\n            }\n        }\n    }\n}'
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)
