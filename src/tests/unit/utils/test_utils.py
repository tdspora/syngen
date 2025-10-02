import pandas as pd
import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta
import numpy as np

from syngen.ml.utils import (
    slugify_attribute,
    slugify_parameters,
    datetime_to_timestamp,
    timestamp_to_datetime,
    fetch_timezone,
    convert_to_timestamp,
    convert_to_date_string,
    fetch_env_variables,
    safe_flatten
)

from tests.conftest import SUCCESSFUL_MESSAGE


def test_slugify_attribute(rp_logger):
    rp_logger.info("Test the decorator 'slugify_attribute'")
    mock = Mock(attr_1="My Test Attribute", attr_2="Мой другой аттрибут", attr_3="@#$12345*&^")

    @slugify_attribute(attr_1="slug_attr1", attr_2="slug_attr2", attr_3="slug_attr3")
    def dummy_function(mock):
        pass

    dummy_function(mock)

    rp_logger.info(
        f"Slugifying the attributes of the mock object - {mock.attr_1, mock.attr_2, mock.attr_3}"
    )
    assert mock.slug_attr1 == "my-test-attribute"
    assert mock.slug_attr2 == "moi-drugoi-attribut"
    assert mock.slug_attr3 == "12345"
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "parameter, expected_parameter",
    [
        ("My Test Attribute", "my-test-attribute"),
        ("Мой другой аттрибут", "moi-drugoi-attribut"),
        ("@#$12345*&^", "12345"),
    ],
)
def test_slugify_parameters(parameter, expected_parameter, rp_logger):
    rp_logger.info(f"Slugifying the parameter - {parameter}")

    @slugify_parameters("name")
    def dummy_function(name):
        return name

    assert dummy_function(name=parameter), expected_parameter
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_datetime_to_timestamp(rp_logger):
    rp_logger.info("Test the function 'datetime_to_timestamp'")
    test_cases = [
        ("0001-01-01", -62135596800.0, "%Y-%m-%d"),
        ("1970-01-01", 0, "%Y-%m-%d"),
        ("2000-01-01", 946684800, "%Y-%m-%d"),
        ("2023-01-01", 1672531200.0, "%Y-%m-%d"),
        ("2023-01-01 00:00:00", 1672531200.0, "%Y-%m-%d"),
        ("2023-01-01 00:00:00.000000", 1672531200.0, "%Y-%m-%d"),
        ("2023-01-01 00:00:00.000000+00:00", 1672531200.0, "%Y-%m-%d"),
        ("9999-12-31", 253402214400.0, "%Y-%m-%d"),
        ("10000-12-31", 253402214400.0, "%Y-%m-%d"),
        (np.nan, np.nan, "%Y-%m-%d"),
        ("31-11-28", 1953590400.0, "%Y-%m-%d")
    ]
    rp_logger.info("Test the method 'datetime_to_timestamp'")
    for date_time, expected_timestamp, date_format in test_cases:
        calculated_timestamp = datetime_to_timestamp(date_time, date_format)
        if np.isnan(expected_timestamp):
            assert np.isnan(calculated_timestamp)
        else:
            assert int(calculated_timestamp) == int(expected_timestamp)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_timestamp_to_datetime(rp_logger):
    test_cases = [
        (-62135596800.0, datetime(1, 1, 1, 0, 0, 0, 0)),
        (0, datetime(1970, 1, 1, 0, 0)),
        (946684800, datetime(2000, 1, 1)),
        (253402214400.0, datetime(9999, 12, 31, 23, 59, 59, 999999)),
        (253402537600.0, datetime(9999, 12, 31, 23, 59, 59, 999999)),
        (np.nan, np.nan),
        (-62135596800.0, datetime(1, 1, 1, 0, 0, 0, 0)),
    ]
    rp_logger.info("Test the function 'timestamp_to_datetime'")
    for timestamp, expected_datetime in test_cases:
        calculated_datetime = timestamp_to_datetime(timestamp)
        if isinstance(expected_datetime, float) and np.isnan(expected_datetime):
            assert np.isnan(calculated_datetime)
        else:
            assert calculated_datetime == expected_datetime
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_timestamp_to_datetime_with_delta(rp_logger):
    test_cases = [
        (0, timedelta(0)),
        (946684800, timedelta(days=10957)),
        (-946684800, timedelta(days=-10957))
    ]
    rp_logger.info(
        "Test the method 'timestamp_to_datetime' with the parameter 'delta' set to 'True'"
    )
    for timestamp, expected_datetime in test_cases:
        calculated_datetime = timestamp_to_datetime(timestamp, delta=True)
        if isinstance(expected_datetime, float) and np.isnan(expected_datetime):
            assert np.isnan(calculated_datetime)
        else:
            assert calculated_datetime == expected_datetime
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "date_column, date_format, na_values, expected_result", [
        (
            pd.Series(["01-02-2023", "03-04-2023", "05-06-2023"]),
            "%d-%m-%Y",
            [],
            [1675209600.0, 1680480000.0, 1685923200.0]
        ),
        (
            pd.Series(["01-02-2023", "03-04-2023", "05-06-2023", "label"]),
            "%d-%m-%Y",
            ["label"],
            [1675209600.0, 1680480000.0, 1685923200.0, np.NaN]
        ),
    ]
)
def test_convert_to_timestamp(date_column, date_format, na_values, expected_result, rp_logger):
    rp_logger.info("Test the function 'convert_to_timestamp'")
    assert convert_to_timestamp(date_column, date_format, na_values) == expected_result
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("value, date_format, expected_result", [
    (1675209600.0, "%d-%m-%Y", "01-02-2023"),
    (1680480000.0, "%d-%m-%Y", "03-04-2023"),
    (1685923200.0, "%d-%m-%Y", "05-06-2023"),
])
def test_convert_to_date_string(value, date_format, expected_result, rp_logger):
    rp_logger.info("Test the function 'convert_to_date_string'")
    assert convert_to_date_string(value, date_format) == expected_result
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("date_string, expected_timezone", [
    ("2023-03-15T14:30:00+00:00", "+00:00"),
    ("2023-03-15T14:30:00-01:00", "-01:00"),
    ("2023-03-15T14:30:00 +0000", "+0000"),
    ("2023-03-15T14:30:00 -0100", "-0100"),
    ("2023-03-15 14:30:00+00:00", "+00:00"),
    ("2023-03-15 14:30:00-01:00", "-01:00"),
    ("2023-03-15 14:30:00 +0000", "+0000"),
    ("2023-03-15 14:30:00 -0100", "-0100"),
    ("2023-03-15-14:30:00+00:00", "+00:00"),
    ("2023-03-15-14:30:00-01:00", "-01:00"),
    ("2023-03-15-14:30:00 +0000", "+0000"),
    ("2023-03-15-14:30:00 -0100", "-0100"),
    ("2023-03-15T14:30:00.000000+00:00", "+00:00"),
    ("2023-03-15T14:30:00.000000-01:00", "-01:00"),
    ("2023-03-15T14:30:00.000000 +0000", "+0000"),
    ("2023-03-15T14:30:00.000000 -0100", "-0100"),
    ("2023-03-15 14:30:00.000000+00:00", "+00:00"),
    ("2023-03-15 14:30:00.000000-01:00", "-01:00"),
    ("2023-03-15 14:30:00.000000 +0000", "+0000"),
    ("2023-03-15 14:30:00.000000 -0100", "-0100"),
    ("2023-03-15-14:30:00.000000+00:00", "+00:00"),
    ("2023-03-15-14:30:00.000000-01:00", "-01:00"),
    ("2023-03-15-14:30:00.000000 +0000", "+0000"),
    ("2023-03-15-14:30:00.000000 -0100", "-0100"),
    ("2023-03-15T11:29:00Z", "Z"),
    ("2023-03-15 11:29:00 AM EDT", "EDT"),
    ("2023-03-15 11:29:00 PST", "PST")
])
def test_fetch_timezone_from_date_string_with_tz(
    rp_logger, date_string, expected_timezone
):
    rp_logger.info(
        "Test the method 'fetch_timezone' that retrieves the timezone from the date string "
        "containing timezone information"
    )
    assert fetch_timezone(date_string) == expected_timezone
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("value, expected_result", [
    (
        '{"key_1": "value_1", "key_2": "value_2"}',
        {
            'flattened_data': {'key_1': 'value_1', 'key_2': 'value_2'},
            'original_data': None
        }
    ),
    (
        b'{"key_1": "value_1", "key_2": "value_2"}',
        {
            'flattened_data': {'key_1': 'value_1', 'key_2': 'value_2'},
            'original_data': None
        }
    ),
    (
        '''
        {
        "member_1": {"name": "John", "occupation": "doctor"},
        "member_2": {"name": "Jane", "occupation": "manager", "department": "financial"}
        }
        ''',
        {
            "flattened_data": {
                "member_1.name": "John",
                "member_1.occupation": "doctor",
                "member_2.department": "financial",
                "member_2.name": "Jane",
                "member_2.occupation": "manager"
            },
            "original_data": None
        }
    ),
    (
        b'''
        {
        "member_1": {"name": "John", "occupation": "doctor"},
        "member_2": {"name": "Jane", "occupation": "manager", "department": "financial"}
        }
        ''',
        {
            "flattened_data": {
                "member_1.name": "John",
                "member_1.occupation": "doctor",
                "member_2.department": "financial",
                "member_2.name": "Jane",
                "member_2.occupation": "manager"
            },
            "original_data": None
        }
    ),
    (
        bytearray(b'''
        {
        "member_1": {"name": "John", "occupation": "doctor"},
        "member_2": {"name": "Jane", "occupation": "manager", "department": "financial"}
        }
        '''),
        {
            "flattened_data": {
                "member_1.name": "John",
                "member_1.occupation": "doctor",
                "member_2.department": "financial",
                "member_2.name": "Jane",
                "member_2.occupation": "manager"
            },
            "original_data": None
        }
    ),
    ("[1, 2, 3, 4, 5]", {"flattened_data": {}, "original_data": '[1, 2, 3, 4, 5]'}),
    ("Not a JSON string", {"flattened_data": {}, "original_data": "Not a JSON string"}),
    (12345, {"flattened_data": {}, "original_data": 12345}),
    (123.45, {"flattened_data": {}, "original_data": 123.45}),
    (np.nan, {"flattened_data": {}, "original_data": np.nan}),
    (None, {"flattened_data": {}, "original_data": None}),
    (True, {"flattened_data": {}, "original_data": True}),
    (False, {"flattened_data": {}, "original_data": False}),
])
def test_safe_flatten(value, expected_result, rp_logger):
    rp_logger.info("Test the method 'safe_flatten'")
    assert safe_flatten(value) == expected_result
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_fetch_env_variables_if_all_env_variables_exist(rp_logger, monkeypatch):
    rp_logger.info(
        "Test the function 'fetch_env_variables' when all environment variables exist"
    )
    monkeypatch.setenv("TEST_ENV_VAR1", "value1")
    monkeypatch.setenv("TEST_ENV_VAR2", "value2")

    env_vars = fetch_env_variables(
        {"secret_1": "TEST_ENV_VAR1", "secret_2": "TEST_ENV_VAR2"}
    )

    assert env_vars == {
        "secret_1": "value1",
        "secret_2": "value2"
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_fetch_env_variables_if_one_of_all_env_variables_exist(rp_logger, monkeypatch):
    rp_logger.info(
        "Test the function 'fetch_env_variables' when one of all environment variables exist"
    )
    monkeypatch.setenv("TEST_ENV_VAR1", "value1")
    with pytest.raises(ValueError) as error:
        fetch_env_variables(
            {"secret_1": "TEST_ENV_VAR1", "secret_2": "TEST_ENV_VAR2"}
        )
        assert (
            "The value of the environment variable 'TEST_ENV_VAR2' wasn't fetched. "
            "Please, check whether it is set correctly."
        ) in str(error.value)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_fetch_env_variables_if_all_env_variables_dont_exist(rp_logger):
    rp_logger.info(
        "Test the function 'fetch_env_variables' if all environment variables don't exist"
    )
    with pytest.raises(ValueError) as error:
        fetch_env_variables(
            {"secret_1": "TEST_ENV_VAR1", "secret_2": "TEST_ENV_VAR2"}
        )
        assert (
            "The value of the environment variable 'TEST_ENV_VAR1' wasn't fetched. "
            "Please, check whether it is set correctly. "
            "The value of the environment variable 'TEST_ENV_VAR2' wasn't fetched. "
            "Please, check whether it is set correctly."
        ) in str(error.value)
    rp_logger.info(SUCCESSFUL_MESSAGE)
