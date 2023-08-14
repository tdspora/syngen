import pytest
from unittest.mock import Mock
from datetime import datetime

from syngen.ml.utils import (
    slugify_attribute,
    slugify_parameters,
)

from tests.conftest import SUCCESSFUL_MESSAGE


def test_slugify_attribute(rp_logger):
    mock = Mock(
        attr_1="My Test Attribute",
        attr_2="Мой другой аттрибут",
        attr_3="@#$12345*&^"
    )

    @slugify_attribute(attr_1="slug_attr1", attr_2="slug_attr2", attr_3="slug_attr3")
    def dummy_function(mock):
        pass

    dummy_function(mock)

    rp_logger.info(f"Slugifying the attributes of the mock object - {mock.attr_1, mock.attr_2, mock.attr_3}")
    assert mock.slug_attr1 == "my-test-attribute"
    assert mock.slug_attr2 == "moi-drugoi-attribut"
    assert mock.slug_attr3 == "12345"
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "parameter, expected_parameter", [
        ("My Test Attribute", "my-test-attribute"),
        ("Мой другой аттрибут", "moi-drugoi-attribut"),
        ("@#$12345*&^", "12345")
    ]
)
def test_slugify_parameters(parameter, expected_parameter, rp_logger):
    rp_logger.info(f"Slugifying the parameter - {parameter}")
    @slugify_parameters("name")
    def dummy_function(name):
        return name

    assert dummy_function(name=parameter), expected_parameter
    rp_logger.info(SUCCESSFUL_MESSAGE)

