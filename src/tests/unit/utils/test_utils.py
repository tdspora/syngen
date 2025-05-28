import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta
import numpy as np

from syngen.ml.utils import (
    slugify_attribute,
    slugify_parameters,
    datetime_to_timestamp,
    timestamp_to_datetime,
    fetch_timezone
)

from tests.conftest import SUCCESSFUL_MESSAGE


def test_slugify_attribute(rp_logger):
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
    rp_logger.info("Test the method 'timestamp_to_datetime'")
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


@pytest.mark.parametrize("date_string, expected_timezone", [
    ("March 15, 2023 02:30:00 PM America/New_York", "America/New_York"),
    ("2023-03-15 14:30:00 Europe/Berlin", "Europe/Berlin"),
    ("2023-03-15T14:30:00+00:00", "+00:00"),
    ("2023-03-15T14:30:00-01:00", "-01:00"),
    ("2023-03-15T14:30:00+0000", "+0000"),
    ("2023-03-15T14:30:00-0100", "-0100"),
    ("2023-03-15T11:29:00Z", "Z"),
    ("2023-03-15 11:29:00 AM EDT", "EDT"),
    ("2023-03-15 11:29:00 PST", "PST")
])
def test_fetch_timezone_from_date_string_with_tz(rp_logger, date_string, expected_timezone):
    rp_logger.info(
        "Test the method 'fetch_timezone' that retrieves the timezone from the date string "
        "containing timezone information"
    )
    assert fetch_timezone(date_string) == expected_timezone
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("date_string", [
    "2023-03-15 14:30:00", "Invalid date string", "2020-01-03 09:00 AM", "2020-01-03 09:00 PM",
])
def test_fetch_timezone_from_date_string_without_tz(rp_logger, date_string):
    rp_logger.info(
        "Test the method 'fetch_timezone' that retrieves the timezone from the date string "
        "not containing timezone information"
    )
    assert np.isnan(fetch_timezone(date_string))
    rp_logger.info(SUCCESSFUL_MESSAGE)