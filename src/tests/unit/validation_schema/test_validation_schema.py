import pytest
from typing import Dict

import yaml
from yaml import Loader
from marshmallow import ValidationError

from syngen.ml.validation_schema import validate_schema

from tests.conftest import SUCCESSFUL_MESSAGE


def load_metadata_file(metadata_path) -> Dict:
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = yaml.load(metadata_file, Loader=Loader)
    return metadata

def test_valid_metadata_file(rp_logger, caplog):
    rp_logger.info("Test the validation of the valid metadata file")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_file.yaml")
    with caplog.at_level(level="INFO"):
        validate_schema(metadata)
        assert "The metadata file is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_without_global_settings(rp_logger, caplog):
    rp_logger.info("Test the validation of the valid metadata file with absent 'global' settings")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_file_with_absent_global_settings.yaml")
    with caplog.at_level(level="INFO"):
        validate_schema(metadata)
        assert "The metadata file is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_valid_metadata_file_only_with_required_fields(rp_logger, caplog):
    rp_logger.info("Test the validation of the valid metadata file with absent 'global' settings")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_file_only_with_required_fields.yaml")
    with caplog.at_level(level="INFO"):
        validate_schema(metadata)
        assert "The metadata file is valid" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("wrong_setting, expected_error", [
    ({"epochs": 0}, "Validation error(s) found in the metadata. "
                    "The details are - {'fk_test': {'train_settings': "
                    "{'epochs': ['Must be greater than or equal to 1.']}}}"),
    ({"epochs": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                "The details are - {'fk_test': {'train_settings': "
                                                "{'epochs': ['Not a valid integer.']}}}"),
    ({"drop_null": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                   "The details are - {'fk_test': {'train_settings': {"
                                                   "'drop_null': ['Not a valid boolean.']}}}"),
    ({"row_limit": 0}, "Validation error(s) found in the metadata. "
                       "The details are - {'fk_test': {'train_settings': "
                       "{'row_limit': ['Must be greater than or equal to 1.']}}}"),
    ({"row_limit": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                   "The details are - {'fk_test': {'train_settings': "
                                                   "{'row_limit': ['Not a valid integer.']}}}"),
    ({"batch_size": 0}, "Validation error(s) found in the metadata. "
                        "The details are - {'fk_test': {'train_settings': "
                        "{'batch_size': ['Must be greater than or equal to 1.']}}}"),
    ({"batch_size": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                    "The details are - {'fk_test': {'train_settings': "
                                                    "{'batch_size': ['Not a valid integer.']}}}"),
    ({"print_report": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                      "The details are - {'fk_test': {'train_settings': "
                                                      "{'print_report': ['Not a valid boolean.']}}}"),
    ({"column_types": {"invalid_category": ["column_1", "column_2"]}}, "Validation error(s) found in the metadata. "
                                                                       "The details are - {'fk_test': {'train_settings': {"
                                                                       "'column_types': defaultdict(<class 'dict'>, {"
                                                                       "'invalid_type': {'key': ['Must be one of: categorical.']}})}}"
     )
])
def test_metadata_file_with_invalid_training_settings(rp_logger, wrong_setting, expected_error):
    rp_logger.info("Test the validation of the metadata  with invalid training settings")
    metadata = load_metadata_file(r"C:\Users\Hanna_Imshenetska\Projects\syngen_open_source\tdm_syngen\src\tests\unit\validation_schema\fixtures\valid_metadata_file.yaml")
    metadata["fk_test"]["train_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        validate_schema(metadata)
    assert str(error.value) == expected_error
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("wrong_setting, expected_error", [
    ({"epochs": 0}, "Validation error(s) found in the metadata. "
                    "The details are - {'global': {'train_settings': "
                    "{'epochs': ['Must be greater than or equal to 1.']}}}"),
    ({"epochs": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                "The details are - {'global': {'train_settings': "
                                                "{'epochs': ['Not a valid integer.']}}}"),
    ({"drop_null": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                   "The details are - {'global': {'train_settings': {"
                                                   "'drop_null': ['Not a valid boolean.']}}}"),
    ({"row_limit": 0}, "Validation error(s) found in the metadata. "
                       "The details are - {'global': {'train_settings': "
                       "{'row_limit': ['Must be greater than or equal to 1.']}}}"),
    ({"row_limit": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                   "The details are - {'global': {'train_settings': "
                                                   "{'row_limit': ['Not a valid integer.']}}}"),
    ({"batch_size": 0}, "Validation error(s) found in the metadata. "
                        "The details are - {'global': {'train_settings': "
                        "{'batch_size': ['Must be greater than or equal to 1.']}}}"),
    ({"batch_size": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                    "The details are - {'global': {'train_settings': "
                                                    "{'batch_size': ['Not a valid integer.']}}}"),
    ({"print_report": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                      "The details are - {'global': {'train_settings': "
                                                      "{'print_report': ['Not a valid boolean.']}}}"
    )
])
def test_metadata_file_with_invalid_training_settings(rp_logger, wrong_setting, expected_error):
    rp_logger.info("Test the validation of the metadata with invalid global training settings")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_file.yaml")
    metadata["global"]["train_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        validate_schema(metadata)
    assert str(error.value) == expected_error
    rp_logger.info(SUCCESSFUL_MESSAGE)

@pytest.mark.parametrize("wrong_setting, expected_error", [
    ({"size": 0}, "Validation error(s) found in the metadata. "
                  "The details are - {'fk_test': {'infer_settings': {"
                  "'size': ['Must be greater than or equal to 1.']}}}"),
    ({"size": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                              "The details are - {'fk_test': {'infer_settings': {"
                                              "'size': ['Not a valid integer.']}}}"),
    ({"run_parallel": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                      "The details are - {'fk_test': {'infer_settings': {"
                                                      "'run_parallel': ['Not a valid boolean.']}}}"),
    ({"random_seed": -1}, "Validation error(s) found in the metadata. "
                          "The details are - {'fk_test': {'infer_settings': {"
                          "'random_seed': ['Must be greater than or equal to 0.']}}}"),
    ({"random_seed": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                     "The details are - {'fk_test': {'infer_settings': "
                                                     "{'random_seed': ['Not a valid integer.']}}}"),
    ({"print_report": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                      "The details are - {'fk_test': {'infer_settings': {"
                                                      "'print_report': ['Not a valid boolean.']}}}")
])
def test_metadata_file_with_invalid_infer_settings(rp_logger, wrong_setting, expected_error):
    rp_logger.info("Test the validation of the metadata with invalid infer settings")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_file.yaml")
    metadata["fk_test"]["infer_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        validate_schema(metadata)
    assert str(error.value) == expected_error
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("wrong_setting, expected_error", [
    ({"size": 0}, "Validation error(s) found in the metadata. "
                  "The details are - {'global': {'infer_settings': {"
                  "'size': ['Must be greater than or equal to 1.']}}}"),
    ({"size": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                              "The details are - {'global': {'infer_settings': {"
                                              "'size': ['Not a valid integer.']}}}"),
    ({"run_parallel": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                      "The details are - {'global': {'infer_settings': {"
                                                      "'run_parallel': ['Not a valid boolean.']}}}"),
    ({"random_seed": -1}, "Validation error(s) found in the metadata. "
                          "The details are - {'global': {'infer_settings': {"
                          "'random_seed': ['Must be greater than or equal to 0.']}}}"),
    ({"random_seed": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                     "The details are - {'global': {'infer_settings': "
                                                     "{'random_seed': ['Not a valid integer.']}}}"),
    ({"print_report": "not a valid type of a value"}, "Validation error(s) found in the metadata. "
                                                      "The details are - {'global': {'infer_settings': {"
                                                      "'print_report': ['Not a valid boolean.']}}}")
])
def test_metadata_file_with_invalid_global_infer_settings(rp_logger, wrong_setting, expected_error):
    rp_logger.info("Test the validation of the metadata with invalid global infer settings")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_file.yaml")
    metadata["global"]["infer_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        validate_schema(metadata)
    assert str(error.value) == expected_error
    rp_logger.info(SUCCESSFUL_MESSAGE)

def test_metadata_file_with_absent_required_fields(rp_logger):
    rp_logger.info("Test the validation of the metadata file with absent required fields")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/metadata_file_without_required_fields.yaml")
    with pytest.raises(ValidationError) as error:
        validate_schema(metadata)
    assert str(error.value) == "Validation error(s) found in the metadata. " \
                               "The details are - {'pk_test': {" \
                               "'source': ['Missing data for required field.']}, " \
                               "'fk_test': {'source': ['Missing data for required field.']}}"
    rp_logger.info(SUCCESSFUL_MESSAGE)

def test_metadata_file_with_invalid_PK_key(rp_logger):
    rp_logger.info("Test the validation of the metadata file with the invalid PK key")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/metadata_file_with_invalid_PK_key.yaml")
    with pytest.raises(ValidationError) as error:
        validate_schema(metadata)
    assert str(error.value) == "Validation error(s) found in the metadata. The details are - {" \
                               "'pk_test': {'keys': defaultdict(<class 'dict'>, {'pk_test_pk_id': {" \
                               "'value': {'_schema': [\"The 'references' field is only allowed when 'type' is 'FK'\"]}}})}}"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_metadata_file_with_invalid_UQ_key(rp_logger):
    rp_logger.info("Test the validation of the metadata file with the invalid UQ key")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/metadata_file_with_invalid_UQ_key.yaml")
    with pytest.raises(ValidationError) as error:
        validate_schema(metadata)
    assert str(error.value) == "Validation error(s) found in the metadata. " \
                               "The details are - {'fk_test': {'keys': defaultdict(<class 'dict'>, {" \
                               "'fk_test_uq_name': {'value': {" \
                               "'_schema': [\"The 'references' field is only allowed when 'type' is 'FK'\"]}}})}}"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_metadata_file_with_invalid_FK_key(rp_logger):
    rp_logger.info("Test the validation of the metadata file with the invalid FK key")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/metadata_file_with_invalid_FK_key.yaml")
    with pytest.raises(ValidationError) as error:
        validate_schema(metadata)
    assert str(error.value) == "Validation error(s) found in the metadata. " \
                               "The details are - {'fk_test': {'keys': defaultdict(<class 'dict'>, {" \
                               "'fk_test_fk_id': {'value': {" \
                               "'_schema': [\"The 'references' field is required when 'type' is 'FK'\"]}}})}}"
    rp_logger.info(SUCCESSFUL_MESSAGE)
