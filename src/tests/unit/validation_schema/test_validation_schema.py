import pytest

from marshmallow import ValidationError

from syngen.ml.validation_schema import ValidationSchema
from syngen.ml.data_loaders import MetadataLoader

from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


def test_valid_metadata_file(rp_logger, caplog):
    rp_logger.info("Test the validation of the schema of the valid metadata file")
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
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
def test_valid_metadata_file_with_diff_types_of_reports(
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
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
            process="train"
        ).validate_schema()
        assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_with_source_contained_path_to_excel_table(
    rp_logger, caplog
):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file contained "
        "the parameter 'source' which is the path to excel table"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/"
        "valid_metadata_file_for_excel_table.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
            process="train"
        ).validate_schema()
        assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_without_global_settings(rp_logger, caplog):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        "with absent 'global' settings"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/"
        "valid_metadata_file_with_absent_global_settings.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
            process="train"
        ).validate_schema()
        assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_only_with_required_fields(rp_logger, caplog):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        "with absent 'global' settings"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/"
        "valid_metadata_file_only_with_required_fields.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
            process="train"
        ).validate_schema()
        assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "wrong_setting, expected_error",
    [
        (
            {"source": 0},
            "The details are - {'fk_test': {'train_settings': {"
            "'source': ['Not a valid string.']}}}",
        ),
        (
            {"epochs": 0},
            "The details are - {'fk_test': {'train_settings': "
            "{'epochs': ['Must be greater than or equal to 1.']}}}",
        ),
        (
            {"epochs": "not a valid type of a value"},
            "The details are - {'fk_test': {'train_settings': "
            "{'epochs': ['Not a valid integer.']}}}",
        ),
        (
            {"drop_null": "not a valid type of a value"},
            "The details are - {'fk_test': {'train_settings': {"
            "'drop_null': ['Not a valid boolean.']}}}",
        ),
        (
            {"row_limit": 0},
            "The details are - {'fk_test': {'train_settings': "
            "{'row_limit': ['Must be greater than or equal to 1.']}}}",
        ),
        (
            {"row_limit": "not a valid type of a value"},
            "The details are - {'fk_test': {'train_settings': "
            "{'row_limit': ['Not a valid integer.']}}}",
        ),
        (
            {"batch_size": 0},
            "The details are - {'fk_test': {'train_settings': "
            "{'batch_size': ['Must be greater than or equal to 1.']}}}",
        ),
        (
            {"batch_size": "not a valid type of a value"},
            "The details are - {'fk_test': {'train_settings': "
            "{'batch_size': ['Not a valid integer.']}}}",
        ),
        (
            {"reports": "not a valid type of a value"},
            "The details are - {'fk_test': {'train_settings': "
            "{'reports': ['Invalid value.']}}}",
        ),
        (
            {"column_types": {"invalid_type": ["column_1", "column_2"]}},
            "The details are - {'fk_test': {'train_settings': {'column_types': "
            "defaultdict(<class 'dict'>, {'invalid_type': {"
            "'key': ['Must be one of: categorical.']}})}}}",
        ),
    ],
)
def test_metadata_file_with_invalid_training_settings(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info(
        "Test the validation of the schema of the metadata  with invalid training settings"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["fk_test"]["train_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
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
            "The details are - {'global': {'train_settings': "
            "{'epochs': ['Must be greater than or equal to 1.']}}}",
        ),
        (
            {"epochs": "not a valid type of a value"},
            "The details are - {'global': {'train_settings': "
            "{'epochs': ['Not a valid integer.']}}}",
        ),
        (
            {"drop_null": "not a valid type of a value"},
            "The details are - {'global': {'train_settings': {"
            "'drop_null': ['Not a valid boolean.']}}}",
        ),
        (
            {"row_limit": 0},
            "The details are - {'global': {'train_settings': "
            "{'row_limit': ['Must be greater than or equal to 1.']}}}",
        ),
        (
            {"row_limit": "not a valid type of a value"},
            "The details are - {'global': {'train_settings': "
            "{'row_limit': ['Not a valid integer.']}}}",
        ),
        (
            {"batch_size": 0},
            "The details are - {'global': {'train_settings': "
            "{'batch_size': ['Must be greater than or equal to 1.']}}}",
        ),
        (
            {"batch_size": "not a valid type of a value"},
            "The details are - {'global': {'train_settings': "
            "{'batch_size': ['Not a valid integer.']}}}",
        ),
        (
            {"reports": "not a valid type of a value"},
            "The details are - {'global': {'train_settings': "
            "{'reports': ['Invalid value.']}}}",
        ),
    ],
)
def test_metadata_file_with_invalid_global_training_settings(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info(
        "Test the validation of the schema of the metadata with invalid global training settings"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["global"]["train_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
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
            "The details are - {'fk_test': {'infer_settings': {"
            "'destination': ['Not a valid string.']}}}",
        ),
        (
            {"size": 0},
            "The details are - {'fk_test': {'infer_settings': {"
            "'size': ['Must be greater than or equal to 1.']}}}",
        ),
        (
            {"size": "not a valid type of a value"},
            "The details are - {'fk_test': {'infer_settings': {"
            "'size': ['Not a valid integer.']}}}",
        ),
        (
            {"run_parallel": "not a valid type of a value"},
            "The details are - {'fk_test': {'infer_settings': {"
            "'run_parallel': ['Not a valid boolean.']}}}",
        ),
        (
            {"random_seed": -1},
            "The details are - {'fk_test': {'infer_settings': {"
            "'random_seed': ['Must be greater than or equal to 0.']}}}",
        ),
        (
            {"random_seed": "not a valid type of a value"},
            "The details are - {'fk_test': {'infer_settings': "
            "{'random_seed': ['Not a valid integer.']}}}",
        ),
        (
            {"reports": "not a valid type of a value"},
            "The details are - {'fk_test': {'infer_settings': {"
            "'reports': ['Invalid value.']}}}",
        ),
    ],
)
def test_metadata_file_with_invalid_infer_settings(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info("Test the validation of the metadata with invalid infer settings")
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["fk_test"]["infer_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
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
            {"size": 0},
            "The details are - {'global': {'infer_settings': {"
            "'size': ['Must be greater than or equal to 1.']}}}",
        ),
        (
            {"size": "not a valid type of a value"},
            "The details are - {'global': {'infer_settings': {"
            "'size': ['Not a valid integer.']}}}",
        ),
        (
            {"run_parallel": "not a valid type of a value"},
            "The details are - {'global': {'infer_settings': {"
            "'run_parallel': ['Not a valid boolean.']}}}",
        ),
        (
            {"random_seed": -1},
            "The details are - {'global': {'infer_settings': {"
            "'random_seed': ['Must be greater than or equal to 0.']}}}",
        ),
        (
            {"random_seed": "not a valid type of a value"},
            "The details are - {'global': {'infer_settings': "
            "{'random_seed': ['Not a valid integer.']}}}",
        ),
        (
            {"reports": "not a valid type of a value"},
            "The details are - {'global': {'infer_settings': {"
            "'reports': ['Invalid value.']}}}",
        ),
    ],
)
def test_metadata_file_with_invalid_global_infer_settings(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info(
        "Test the validation of the schema of the metadata with invalid global infer settings"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["global"]["infer_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
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
            "The details are - {'global': {'encryption': {'fernet_key': ['Not a valid string.']}}}"
        )
    ],
)
def test_metadata_file_with_invalid_global_encryption_settings(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info(
        "Test the validation of the schema of the metadata with invalid global encryption settings"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["global"]["encryption"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
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
            "The details are - {'fk_test': {'encryption': "
            "{'fernet_key': ['Not a valid string.']}}}"
        )
    ],
)
def test_metadata_file_with_invalid_encryption_settings(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info(
        "Test the validation of the schema of the metadata with invalid encryption settings"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["fk_test"]["encryption"] = wrong_setting
    with pytest.raises(ValidationError) as error:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
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
        ({"sep": 0}, "The details are - {'fk_test': {'sep': ['Not a valid string.']}}"),
        (
            {"quotechar": 0},
            "The details are - {'fk_test': {'quotechar': ['Not a valid string.']}}",
        ),
        (
            {"quotechar": "value with more than one character"},
            "The details are - {'fk_test': {'quotechar': [" "'Length must be 1.']}}",
        ),
        (
            {"quoting": 0},
            "The details are - {'fk_test': {'quoting': ['Not a valid string.']}}",
        ),
        (
            {"quoting": "not a valid value"},
            "The details are - {'fk_test': {'quoting': ["
            "'Must be one of: minimal, all, non-numeric, none.']}}",
        ),
        (
            {"escapechar": 0},
            "The details are - {'fk_test': {'escapechar': ['Not a valid string.']}}",
        ),
        (
            {"escapechar": "value with more than one character"},
            "The details are - {'fk_test': {'escapechar': [" "'Length must be 1.']}}",
        ),
        (
            {"encoding": 0},
            "The details are - {'fk_test': {'encoding': ['Not a valid string.']}}",
        ),
        (
            {"header": "not a valid type of a value"},
            "The details are - {'fk_test': {" "'header': ['Invalid value.']}}",
        ),
        (
            {"skiprows": "not a valid type of a value"},
            "The details are - {'fk_test': {" "'skiprows': ['Invalid value.']}}",
        ),
        (
            {"on_bad_lines": 0},
            "The details are - {'fk_test': {'on_bad_lines': ['Not a valid string.']}}",
        ),
        (
            {"on_bad_lines": "not a valid value"},
            "The details are - {'fk_test': {'on_bad_lines': ["
            "'Must be one of: error, warn, skip.']}}",
        ),
        (
            {"engine": 0},
            "The details are - {'fk_test': {'engine': ['Not a valid string.']}}",
        ),
        (
            {"engine": "not a valid value"},
            "The details are - {'fk_test': {'engine': ['Must be one of: c, python.']}}",
        ),
        (
            {"sheet_name": 0},
            "The details are - {'fk_test': {'sheet_name': ['Unknown field.']}}",
        ),
    ],
)
def test_metadata_file_with_invalid_format_settings_for_csv_table(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info(
        "Test the validation of the schema of the metadata with format settings set to CSV file"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/valid_metadata_file.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["fk_test"]["format"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
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
        ({"sep": ","}, "The details are - {'pk_test': {'sep': ['Unknown field.']}}"),
        (
            {"quotechar": '"'},
            "The details are - {'pk_test': {'quotechar': ['Unknown field.']}}",
        ),
        (
            {"quoting": "non-numeric"},
            "The details are - {'pk_test': {'quoting': ['Unknown field.']}}",
        ),
        (
            {"escapechar": "\\"},
            "The details are - {'pk_test': {'escapechar': ['Unknown field.']}}",
        ),
        (
            {"encoding": "ascii"},
            "The details are - {'pk_test': {'encoding': ['Unknown field.']}}",
        ),
        (
            {"header": 0},
            "The details are - {'pk_test': {'header': ['Unknown field.']}}",
        ),
        (
            {"skiprows": 0},
            "The details are - {'pk_test': {'skiprows': ['Unknown field.']}}",
        ),
        (
            {"on_bad_lines": "skip"},
            "The details are - {'pk_test': {'on_bad_lines': ['Unknown field.']}}",
        ),
        (
            {"engine": "python"},
            "The details are - {'pk_test': {'engine': ['Unknown field.']}}",
        ),
    ],
)
def test_metadata_file_with_invalid_format_settings_for_excel_table(
    rp_logger, wrong_setting, expected_error
):
    rp_logger.info(
        "Test the validation of the schema of the metadata with format settings set to Excel table"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/"
        "valid_metadata_file_for_excel_table.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    metadata["pk_test"]["format"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
            process="train"
        ).validate_schema()
    assert (
        str(error.value)
        == f"Validation error(s) found in the schema of the metadata. {expected_error}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_metadata_file_with_absent_required_fields(rp_logger):
    rp_logger.info(
        "Test the validation of the schema of the metadata file with absent required fields"
    )
    path_to_metadata = (f"{DIR_NAME}/unit/validation_schema/fixtures/"
                        "metadata_file_without_required_fields.yaml")
    metadata = MetadataLoader(path_to_metadata).load_data()
    with pytest.raises(ValidationError) as error:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
            process="train"
        ).validate_schema()
    assert str(error.value) == (
        "Validation error(s) found in the schema of the metadata. "
        "The details are - {'pk_test': {'train_settings': "
        "{'source': ['Missing data for required field.']}}, 'fk_test': {"
        "'train_settings': {'source': ['Missing data for required field.']}}}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_metadata_file_with_invalid_PK_key_contained_references_section(rp_logger):
    rp_logger.info(
        "Test the validation of the schema of the metadata file with the invalid PK "
        "which contained 'references' section"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/metadata_file_with_invalid_PK_key.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with pytest.raises(ValidationError) as error:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
            process="train"
        ).validate_schema()
    assert str(error.value) == (
        "Validation error(s) found in the schema of the metadata. "
        "The details are - {'pk_test': {'keys': defaultdict(<class 'dict'>, {"
        "'pk_test_pk_id': {'value': {'_schema': [\"The 'references' field "
        "is only allowed when 'type' is 'FK'\"]}}})}}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_metadata_file_with_invalid_UQ_key_contained_references_section(rp_logger):
    rp_logger.info(
        "Test the validation of the schema of the metadata file with the invalid UQ "
        "contained 'references' section"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/metadata_file_with_invalid_UQ_key.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with pytest.raises(ValidationError) as error:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
            process="train"
        ).validate_schema()
    assert str(error.value) == (
        "Validation error(s) found in the schema of the metadata. "
        "The details are - {'fk_test': {'keys': defaultdict(<class 'dict'>, {"
        "'fk_test_uq_name': {'value': {"
        "'_schema': [\"The 'references' field is only allowed when 'type' is 'FK'\"]}}})}}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_metadata_file_with_invalid_FK_key_without_references_section(rp_logger):
    rp_logger.info(
        "Test the validation of the schema of the metadata file with the invalid FK "
        "which doesn't contain 'references' section"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/metadata_file_with_invalid_FK_key.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with pytest.raises(ValidationError) as error:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
            process="train"
        ).validate_schema()
    assert str(error.value) == (
        "Validation error(s) found in the schema of the metadata. "
        "The details are - {'fk_test': {'keys': defaultdict(<class 'dict'>, {"
        "'fk_test_fk_id': {'value': {"
        "'_schema': [\"The 'references' field is required when 'type' is 'FK'\"]}}})}}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "path_to_metadata, expected_error",
    [
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_absent_pk_columns.yaml",
            "The details are - {'table_a': {'keys': defaultdict(<class 'dict'>, {"
            "'pk_id': {'value': {'columns': ['Field may not be null.']}}})}}",
        ),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_absent_fk_columns.yaml",
            "The details are - {'table_b': {'keys': defaultdict(<class 'dict'>, {"
            "'fk_id': {'value': {'columns': ['Missing data for required field.']}}})}}",
        ),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_absent_ref_table.yaml",
            "The details are - {'table_b': {'keys': defaultdict(<class 'dict'>, {"
            "'fk_id': {'value': {'references': {'table': ["
            "'Missing data for required field.']}}}})}}",
        ),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_absent_ref_columns.yaml",
            "The details are - {'table_b': {'keys': defaultdict(<class 'dict'>, {"
            "'fk_id': {'value': {'references': {'columns': ["
            "'Missing data for required field.']}}}})}}",
        ),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_duplicated_pk_columns.yaml",
            "The details are - {'table_a': {'keys': defaultdict(<class 'dict'>, {"
            "'pk_id': {'value': {'_schema': [\"The 'columns' field "
            'must contain unique values"]}}})}}',
        ),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_duplicated_fk_columns.yaml",
            "The details are - {'table_b': {'keys': defaultdict(<class 'dict'>, {'fk_id': {"
            "'value': {'_schema': [\"The 'columns' field must contain unique values\"]}}})}}",
        ),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_absent_uq_columns.yaml",
            "The details are - {'table_a': {'keys': defaultdict(<class 'dict'>, {"
            "'pk_id': {'value': {'columns': ['Field may not be null.']}}})}}",
        ),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_duplicated_uq_columns.yaml",
            "The details are - {'table_a': {'keys': defaultdict(<class 'dict'>, {"
            "'uq_id': {'value': {'_schema': [\"The 'columns' field "
            'must contain unique values"]}}})}}',
        ),
        (
            f"{DIR_NAME}/unit/validation_schema/fixtures/"
            "metadata_file_of_related_tables_with_diff_length_of_columns.yaml",
            "The details are - {'table_b': {'keys': defaultdict(<class 'dict'>, {"
            "'fk_id': {'value': {'_schema': [\"The 'columns' field must have "
            "the same length as 'references.columns'\"]}}})}}",
        ),
    ],
)
def test_validation_schema_of_keys(rp_logger, path_to_metadata, expected_error):
    rp_logger.info(
        "Test the validation of the schema of the metadata file with invalid section 'keys'"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with pytest.raises(ValidationError) as error:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
            process="train"
        ).validate_schema()
    assert str(error.value) == (
        f"Validation error(s) found in the schema of the metadata. {expected_error}"
    )


def test_valid_metadata_file_without_sources_during_training_process_without_validation_source(
    rp_logger, caplog
):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file without provided sources "
        "during the training process in case the validation of the source is turned off"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/"
        "metadata_file_without_sources.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=False,
            process="train"
        ).validate_schema()
        assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_without_sources_during_training_process_with_validation_source(
    rp_logger
):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file without provided sources "
        "during the training process in case the validation of the source is turned on"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/"
        "metadata_file_without_sources.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with pytest.raises(ValidationError) as e:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
            process="train"
        ).validate_schema()
    assert str(e.value) == (
        "Validation error(s) found in the schema of the metadata. "
        "The details are - {'pk_test': {'train_settings': {"
        "'source': ['Missing data for required field.']}}, "
        "'fk_test': {'train_settings': {'source': ['Missing data for required field.']}}}"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_without_sources_during_infer_process_without_validation_source(
    rp_logger, caplog
):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file without provided sources "
        "during the inference process in case in case the validation of the source is turned off"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/"
        "metadata_file_without_sources.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=False,
            process="infer"
        ).validate_schema()
        assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_without_sources_during_inference_process_with_validation_source(
    rp_logger, caplog
):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file without provided sources "
        "during the training process in case the validation of the source is turned on"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/"
        "metadata_file_without_sources.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
            process="infer"
        ).validate_schema()
        assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_without_training_settings_during_train_process_without_valid_source(
    rp_logger, caplog
):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file during the training process "
        "without provided 'training_settings' in case the validation of the source is turned off"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/"
        "metadata_file_without_training_settings.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=False,
            process="train"
        ).validate_schema()
        assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_without_training_settings_during_train_process_with_valid_source(
    rp_logger
):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file during the training process "
        "without provided 'training_settings' in case the validation of the source is turned on"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/"
        "metadata_file_without_training_settings.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with pytest.raises(ValidationError) as e:
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
            process="train"
        ).validate_schema()
        assert str(e.value) == (
            "Validation error(s) found in the schema of the metadata. "
            "The details are - {'pk_test': {'train_settings': ['Field may not be null.']}, "
            "'fk_test': {'train_settings': ['Field may not be null.']}}"
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_without_training_settings_during_infer_process_without_valid_source(
    rp_logger, caplog
):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        "during the inference process without provided 'training_settings' "
        "in case the validation of the source is turned off"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/"
        "metadata_file_without_training_settings.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=False,
            process="infer"
        ).validate_schema()
        assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_without_training_settings_during_infer_process_with_valid_source(
    rp_logger, caplog
):
    rp_logger.info(
        "Test the validation of the schema of the valid metadata file "
        "during the inference process without provided 'training_settings' "
        "in case the validation of the source is turned on"
    )
    path_to_metadata = (
        f"{DIR_NAME}/unit/validation_schema/fixtures/"
        "metadata_file_without_training_settings.yaml"
    )
    metadata = MetadataLoader(path_to_metadata).load_data()
    with caplog.at_level(level="DEBUG"):
        ValidationSchema(
            metadata=metadata,
            metadata_path=path_to_metadata,
            validation_source=True,
            process="infer"
        ).validate_schema()
        assert "The schema of the metadata is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)
