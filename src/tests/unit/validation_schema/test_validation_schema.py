import pytest
from typing import Dict

import yaml
from yaml import Loader
from marshmallow import ValidationError

from syngen.ml.validation_schema import ValidationSchema

from tests.conftest import SUCCESSFUL_MESSAGE


def load_metadata_file(metadata_path) -> Dict:
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = yaml.load(metadata_file, Loader=Loader)
    return metadata


def test_valid_metadata_file(rp_logger, caplog):
    rp_logger.info("Test the validation of the schema of the valid metadata file")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_file.yaml")
    with caplog.at_level(level="INFO"):
        ValidationSchema(metadata).validate_schema()
        assert "The schema of the metadata file is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_with_source_contained_path_to_excel_table(rp_logger, caplog):
    rp_logger.info("Test the validation of the schema of the valid metadata file contained "
                   "the parameter 'source' which is the path to excel table")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_for_excel_table.yaml")
    with caplog.at_level(level="INFO"):
        ValidationSchema(metadata).validate_schema()
        assert "The schema of the metadata file is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_without_global_settings(rp_logger, caplog):
    rp_logger.info("Test the validation of the schema of the valid metadata file with absent 'global' settings")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/"
                                  "valid_metadata_file_with_absent_global_settings.yaml")
    with caplog.at_level(level="INFO"):
        ValidationSchema(metadata).validate_schema()
        assert "The schema of the metadata file is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_only_with_required_fields(rp_logger, caplog):
    rp_logger.info("Test the validation of the schema of the valid metadata file with absent 'global' settings")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/"
                                  "valid_metadata_file_only_with_required_fields.yaml")
    with caplog.at_level(level="INFO"):
        ValidationSchema(metadata).validate_schema()
        assert "The schema of the metadata file is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("wrong_setting, expected_error", [
    ({"source": 0}, "Validation error(s) found in the schema of the metadata file. "
                    "The details are - {'fk_test': {'infer_settings': {"
                    "'destination': ['Not a valid string.']}}}"),
    ({"epochs": 0}, "Validation error(s) found in the schema of the metadata file. "
                    "The details are - {'fk_test': {'train_settings': "
                    "{'epochs': ['Must be greater than or equal to 1.']}}}"),
    ({"epochs": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                "The details are - {'fk_test': {'train_settings': "
                                                "{'epochs': ['Not a valid integer.']}}}"),
    ({"drop_null": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                   "The details are - {'fk_test': {'train_settings': {"
                                                   "'drop_null': ['Not a valid boolean.']}}}"),
    ({"row_limit": 0}, "Validation error(s) found in the schema of the metadata file. "
                       "The details are - {'fk_test': {'train_settings': "
                       "{'row_limit': ['Must be greater than or equal to 1.']}}}"),
    ({"row_limit": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                   "The details are - {'fk_test': {'train_settings': "
                                                   "{'row_limit': ['Not a valid integer.']}}}"),
    ({"batch_size": 0}, "Validation error(s) found in the schema of the metadata file. "
                        "The details are - {'fk_test': {'train_settings': "
                        "{'batch_size': ['Must be greater than or equal to 1.']}}}"),
    ({"batch_size": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                    "The details are - {'fk_test': {'train_settings': "
                                                    "{'batch_size': ['Not a valid integer.']}}}"),
    ({"print_report": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                      "The details are - {'fk_test': {'train_settings': "
                                                      "{'print_report': ['Not a valid boolean.']}}}"),
    ({"column_types": {"invalid_category": ["column_1", "column_2"]}}, "Validation error(s) found in the schema of "
                                                                       "the metadata file . The details are - {"
                                                                       "'fk_test': {'train_settings': {'column_types': "
                                                                       "defaultdict(<class 'dict'>, {'invalid_type': {"
                                                                       "'key': ['Must be one of: categorical.']}})}}"
     )
])
def test_metadata_file_with_invalid_training_settings(rp_logger, wrong_setting, expected_error):
    rp_logger.info("Test the validation of the schema of the metadata  with invalid training settings")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_file.yaml")
    metadata["fk_test"]["train_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationSchema(metadata).validate_schema()
    assert str(error.value) == expected_error
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("wrong_setting, expected_error", [
    ({"epochs": 0}, "Validation error(s) found in the schema of the metadata file. "
                    "The details are - {'global': {'train_settings': "
                    "{'epochs': ['Must be greater than or equal to 1.']}}}"),
    ({"epochs": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                "The details are - {'global': {'train_settings': "
                                                "{'epochs': ['Not a valid integer.']}}}"),
    ({"drop_null": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                   "The details are - {'global': {'train_settings': {"
                                                   "'drop_null': ['Not a valid boolean.']}}}"),
    ({"row_limit": 0}, "Validation error(s) found in the schema of the metadata file. "
                       "The details are - {'global': {'train_settings': "
                       "{'row_limit': ['Must be greater than or equal to 1.']}}}"),
    ({"row_limit": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                   "The details are - {'global': {'train_settings': "
                                                   "{'row_limit': ['Not a valid integer.']}}}"),
    ({"batch_size": 0}, "Validation error(s) found in the schema of the metadata file. "
                        "The details are - {'global': {'train_settings': "
                        "{'batch_size': ['Must be greater than or equal to 1.']}}}"),
    ({"batch_size": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                    "The details are - {'global': {'train_settings': "
                                                    "{'batch_size': ['Not a valid integer.']}}}"),
    ({"print_report": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                      "The details are - {'global': {'train_settings': "
                                                      "{'print_report': ['Not a valid boolean.']}}}"
    )
])
def test_metadata_file_with_invalid_training_settings(rp_logger, wrong_setting, expected_error):
    rp_logger.info("Test the validation of the schema of the metadata with invalid global training settings")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_file.yaml")
    metadata["global"]["train_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationSchema(metadata).validate_schema()
    assert str(error.value) == expected_error
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("wrong_setting, expected_error", [
    ({"destination": 0}, "Validation error(s) found in the schema of the metadata file. "
                         "The details are - {'fk_test': {'infer_settings': {"
                         "'destination': ['Not a valid string.']}}}"),
    ({"size": 0}, "Validation error(s) found in the schema of the metadata file. "
                  "The details are - {'fk_test': {'infer_settings': {"
                  "'size': ['Must be greater than or equal to 1.']}}}"),
    ({"size": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                              "The details are - {'fk_test': {'infer_settings': {"
                                              "'size': ['Not a valid integer.']}}}"),
    ({"run_parallel": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                      "The details are - {'fk_test': {'infer_settings': {"
                                                      "'run_parallel': ['Not a valid boolean.']}}}"),
    ({"random_seed": -1}, "Validation error(s) found in the schema of the metadata file. "
                          "The details are - {'fk_test': {'infer_settings': {"
                          "'random_seed': ['Must be greater than or equal to 0.']}}}"),
    ({"random_seed": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                     "The details are - {'fk_test': {'infer_settings': "
                                                     "{'random_seed': ['Not a valid integer.']}}}"),
    ({"print_report": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                      "The details are - {'fk_test': {'infer_settings': {"
                                                      "'print_report': ['Not a valid boolean.']}}}")
])
def test_metadata_file_with_invalid_infer_settings(rp_logger, wrong_setting, expected_error):
    rp_logger.info("Test the validation of the metadata with invalid infer settings")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_file.yaml")
    metadata["fk_test"]["infer_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationSchema(metadata).validate_schema()
    assert str(error.value) == expected_error
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("wrong_setting, expected_error", [
    ({"size": 0}, "Validation error(s) found in the schema of the metadata file. "
                  "The details are - {'global': {'infer_settings': {"
                  "'size': ['Must be greater than or equal to 1.']}}}"),
    ({"size": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                              "The details are - {'global': {'infer_settings': {"
                                              "'size': ['Not a valid integer.']}}}"),
    ({"run_parallel": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                      "The details are - {'global': {'infer_settings': {"
                                                      "'run_parallel': ['Not a valid boolean.']}}}"),
    ({"random_seed": -1}, "Validation error(s) found in the schema of the metadata file. "
                          "The details are - {'global': {'infer_settings': {"
                          "'random_seed': ['Must be greater than or equal to 0.']}}}"),
    ({"random_seed": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                     "The details are - {'global': {'infer_settings': "
                                                     "{'random_seed': ['Not a valid integer.']}}}"),
    ({"print_report": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                      "The details are - {'global': {'infer_settings': {"
                                                      "'print_report': ['Not a valid boolean.']}}}")
])
def test_metadata_file_with_invalid_global_infer_settings(rp_logger, wrong_setting, expected_error):
    rp_logger.info("Test the validation of the schema of the metadata with invalid global infer settings")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_file.yaml")
    metadata["global"]["infer_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationSchema(metadata).validate_schema()
    assert str(error.value) == expected_error
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("wrong_setting, expected_error", [
    ({"sep": 0}, "Validation error(s) found in the schema of the metadata file. "
                 "The details are - {'fk_test': {'sep': ['Not a valid string.']}}"),
    ({"quotechar": 0}, "Validation error(s) found in the schema of the metadata file. "
                       "The details are - {'fk_test': {'quotechar': ['Not a valid string.']}}"),
    ({"quotechar": "value with more than one character"}, "Validation error(s) found in the schema of the metadata "
                                                          "file. The details are - {'fk_test': {'quotechar': ["
                                                          "'Length must be 1.']}}"),
    ({"quoting": 0}, "Validation error(s) found in the schema of the metadata file. "
                     "The details are - {'fk_test': {'quoting': ['Not a valid string.']}}"),
    ({"quoting": "not a valid value"}, "Validation error(s) found in the schema of the metadata file. "
                                       "The details are - {'fk_test': {'quoting': ["
                                       "'Must be one of: minimal, all, non-numeric, none.']}}"),
    ({"escapechar": 0}, "Validation error(s) found in the schema of the metadata file. "
                        "The details are - {'fk_test': {'escapechar': ['Not a valid string.']}}"),
    ({"escapechar": "value with more than one character"}, "Validation error(s) found in the schema of the metadata "
                                                           "file. The details are - {'fk_test': {'escapechar': ["
                                                           "'Length must be 1.']}}"),
    ({"encoding": 0}, "Validation error(s) found in the schema of the metadata file. "
                      "The details are - {'fk_test': {'encoding': ['Not a valid string.']}}"),
    ({"header": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                "The details are - {'fk_test': {"
                                                "'header': ['Invalid value.']}}"),
    ({"skiprows": "not a valid type of a value"}, "Validation error(s) found in the schema of the metadata file. "
                                                  "The details are - {'fk_test': {"
                                                  "'skiprows': ['Invalid value.']}}"),
    ({"on_bad_lines": 0}, "Validation error(s) found in the schema of the metadata file. "
                          "The details are - {'fk_test': {'on_bad_lines': ['Not a valid string.']}}"),
    ({"on_bad_lines": "not a valid value"}, "Validation error(s) found in the schema of the metadata file. "
                                            "The details are - {'fk_test': {'on_bad_lines': ["
                                            "'Must be one of: error, warn, skip.']}}"),
    ({"engine": 0}, "Validation error(s) found in the schema of the metadata file. "
                    "The details are - {'fk_test': {'engine': ['Not a valid string.']}}"),
    ({"engine": "not a valid value"}, "Validation error(s) found in the schema of the metadata file. "
                                      "The details are - {'fk_test': {'engine': ["
                                      "'Must be one of: c, python.']}}"),
    ({"sheet_name": 0}, "Validation error(s) found in the schema of the metadata file. "
                        "The details are - {'fk_test': {'sheet_name': ['Unknown field.']}}")
])
def test_metadata_file_with_invalid_format_settings_for_csv_table(rp_logger, wrong_setting, expected_error):
    rp_logger.info("Test the validation of the schema of the metadata with format settings set to CSV file")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_file.yaml")
    metadata["fk_test"]["format"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationSchema(metadata).validate_schema()
    assert str(error.value) == expected_error
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("wrong_setting, expected_error", [
    ({"sep": ","}, "Validation error(s) found in the schema of the metadata file. "
                   "The details are - {'pk_test': {'sep': ['Unknown field.']}}"),
    ({"quotechar": '"'}, "Validation error(s) found in the schema of the metadata file. "
                         "The details are - {'pk_test': {'quotechar': ['Unknown field.']}}"),
    ({"quoting": "non-numeric"}, "Validation error(s) found in the schema of the metadata file. "
                                 "The details are - {'pk_test': {'quoting': ['Unknown field.']}}"),
    ({"escapechar": "\\"}, "Validation error(s) found in the schema of the metadata file. "
                           "The details are - {'pk_test': {'escapechar': ['Unknown field.']}}"),
    ({"encoding": "ascii"}, "Validation error(s) found in the schema of the metadata file. "
                            "The details are - {'pk_test': {'encoding': ['Unknown field.']}}"),
    ({"header": 0}, "Validation error(s) found in the schema of the metadata file. "
                    "The details are - {'pk_test': {'header': ['Unknown field.']}}"),
    ({"skiprows": 0}, "Validation error(s) found in the schema of the metadata file. "
                      "The details are - {'pk_test': {'skiprows': ['Unknown field.']}}"),
    ({"on_bad_lines": "skip"}, "Validation error(s) found in the schema of the metadata file. "
                               "The details are - {'pk_test': {'on_bad_lines': ['Unknown field.']}}"),
    ({"engine": "python"}, "Validation error(s) found in the schema of the metadata file. "
                           "The details are - {'pk_test': {'engine': ['Unknown field.']}}")
])
def test_metadata_file_with_invalid_format_settings_for_excel_table(rp_logger, wrong_setting, expected_error):
    rp_logger.info("Test the validation of the schema of the metadata with format settings set to Excel table")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_for_excel_table.yaml")
    metadata["pk_test"]["format"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        ValidationSchema(metadata).validate_schema()
    assert str(error.value) == expected_error
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_metadata_file_with_absent_required_fields(rp_logger):
    rp_logger.info("Test the validation of the schema of the metadata file with absent required fields")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/metadata_file_without_required_fields.yaml")
    with pytest.raises(ValidationError) as error:
        ValidationSchema(metadata).validate_schema()
    assert str(error.value) == "Validation error(s) found in the schema of the metadata file. " \
                               "The details are - {'pk_test': {'train_settings': [" \
                               "'Missing data for required field.']}, 'fk_test': {" \
                               "'train_settings': {'source': ['Missing data for required field.']}}}"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_metadata_file_with_invalid_PK_key(rp_logger):
    rp_logger.info("Test the validation of the schema of the metadata file with the invalid PK key")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/metadata_file_with_invalid_PK_key.yaml")
    with pytest.raises(ValidationError) as error:
        ValidationSchema(metadata).validate_schema()
    assert str(error.value) == "Validation error(s) found in the schema of the metadata file. The details are - {" \
                               "'pk_test': {'keys': defaultdict(<class 'dict'>, {'pk_test_pk_id': {" \
                               "'value': {'_schema': [\"The 'references' field is only allowed when " \
                               "'type' is 'FK'\"]}}})}}"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_metadata_file_with_invalid_UQ_key(rp_logger):
    rp_logger.info("Test the validation of the schema of the metadata file with the invalid UQ key")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/metadata_file_with_invalid_UQ_key.yaml")
    with pytest.raises(ValidationError) as error:
        ValidationSchema(metadata).validate_schema()
    assert str(error.value) == "Validation error(s) found in the schema of the metadata file. " \
                               "The details are - {'fk_test': {'keys': defaultdict(<class 'dict'>, {" \
                               "'fk_test_uq_name': {'value': {" \
                               "'_schema': [\"The 'references' field is only allowed when 'type' is 'FK'\"]}}})}}"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_metadata_file_with_invalid_FK_key(rp_logger):
    rp_logger.info("Test the validation of the schema of the metadata file with the invalid FK key")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/metadata_file_with_invalid_FK_key.yaml")
    with pytest.raises(ValidationError) as error:
        ValidationSchema(metadata).validate_schema()
    assert str(error.value) == "Validation error(s) found in the schema of the metadata file. " \
                               "The details are - {'fk_test': {'keys': defaultdict(<class 'dict'>, {" \
                               "'fk_test_fk_id': {'value': {" \
                               "'_schema': [\"The 'references' field is required when 'type' is 'FK'\"]}}})}}"
    rp_logger.info(SUCCESSFUL_MESSAGE)
