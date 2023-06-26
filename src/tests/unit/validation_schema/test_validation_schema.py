import pytest
from typing import Dict

import yaml
from yaml import Loader
from marshmallow import ValidationError

from syngen.ml.validation_schema import validate_schema

from tests.conftest import SUCCESSFUL_MESSAGE, rp_logger


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
    ({"epochs": 0}, "Validation errors found in the metadata. "
                    "The details are - {'fk_test': {'train_settings': "
                    "{'epochs': ['Must be greater than or equal to 1.']}}}"),
    ({"epochs": "not valid value"}, "Validation errors found in the metadata. "
                                    "The details are - {'fk_test': {'train_settings': "
                                    "{'epochs': ['Not a valid integer.']}}}"),
    ()
])
def test_invalid_metadata_file(rp_logger, wrong_setting, expected_error):
    rp_logger.info("Test the validation of the valid metadata file")
    metadata = load_metadata_file("./tests/unit/validation_schema/fixtures/valid_metadata_file.yaml")
    metadata["fk_test"]["train_settings"].update(wrong_setting)
    with pytest.raises(ValidationError) as error:
        validate_schema(metadata)
    assert str(error.value) == expected_error
    rp_logger.info(SUCCESSFUL_MESSAGE)
