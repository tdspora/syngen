import pytest

from syngen.ml.data_loaders import MetadataLoader

from tests.conftest import SUCCESSFUL_MESSAGE


def load_metadata():
    return MetadataLoader('./tests/unit/schema/fixtures/housing_metadata_format.yaml').load_data()


def test_schema_validation(rp_logger):
    rp_logger.info("Validating metadata schema %s", "housing_metadata_format.yaml")
    load_metadata
    rp_logger.info(SUCCESSFUL_MESSAGE)
    
    
def test_format_options(rp_logger):
    metadata = load_metadata()
    format = metadata.get("housing_properties").get("format")
    assert format.get("sep") == ","
    assert format.get("quotechar") == '"'
    assert format.get("quoting") == "All"
    assert format.get("escapechar") == '\\'
    assert format.get("encoding") == "utf-8"
    assert format.get("header") == False
    assert format.get("skiprows") == 0
    assert format.get("on_bad_lines") == 'skip'
    assert format.get("engine") == 'c'
    rp_logger.info(SUCCESSFUL_MESSAGE)
