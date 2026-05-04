import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta, timezone

import numpy as np

from syngen.ml.utils import (
    slugify_attribute,
    slugify_parameters,
    datetime_to_timestamp,
    timestamp_to_datetime,
    fetch_timezone,
    convert_date_to_timestamp,
    convert_to_date,
    fetch_env_variables,
    get_source_path_extension,
    generate_unique_values_by_regex,
    is_number_regex_pattern,
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
        ("9999-12-31", 253402214400, "%Y-%m-%d"),
        ("10000-12-31", 253402300800, "%Y-%m-%d"),
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
        (-62135596800.0, datetime(1, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)),
        (0, datetime(1970, 1, 1, 0, 0)),
        (946684800, datetime(2000, 1, 1)),
        (253402214400.0, datetime(9999, 12, 31, 0, 0)),
        (253402537600.0, datetime(9999, 12, 31, 23, 59, 59, 999999)),
        (np.nan, np.nan)
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
    "value, date_format, na_values, expected_result", [
        (
            "01-02-2023", "%d-%m-%Y", [], 1675209600.0
        ),
        (
            "label", "%d-%m-%Y", ["label"], None
        ),
        (
            None, "%d-%m-%Y", [], None
        )
    ]
)
def test_convert_date_to_timestamp(value, date_format, na_values, expected_result, rp_logger):
    rp_logger.info("Test the function 'convert_date_to_timestamp'")
    assert convert_date_to_timestamp(value, date_format, na_values) == expected_result
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("value, date_format, to_datetime_conversion, expected_result", [
    (1675209600.0, "%d-%m-%Y", False, "01-02-2023"),
    (1680480000.0, "%d-%m-%Y", False, "03-04-2023"),
    (1685923200.0, "%d-%m-%Y", False, "05-06-2023"),
    (1675209600.0, "%Y-%m-%d", True, datetime(2023, 2, 1, 0, 0)),
    (1680480000.0, "%Y-%m-%d", True, datetime(2023, 4, 3, 0, 0)),
    (1685923200.0, "%Y-%m-%d", True, datetime(2023, 6, 5, 0, 0)),
])
def test_convert_to_date(value, date_format, expected_result, to_datetime_conversion, rp_logger):
    rp_logger.info("Test the function 'convert_to_date'")
    assert convert_to_date(value, date_format, to_datetime_conversion) == expected_result
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


@pytest.mark.parametrize(
    "path, expected",
    [
        ("path/to/file.csv", ".csv"),
        ("path/to/file.avro", ".avro"),
        ("path/to/file.xls", ".xls"),
        ("path/to/file.xlsx", ".xlsx"),
        ("no_extension", ""),
    ],
)
def test_get_source_path_extension_with_various_path(path, expected, rp_logger):
    """
    The helper should return the appropriate file extension(s) from the given path,
    or an empty string if no extension is present.
    """
    rp_logger.info(
        "Test the function 'get_source_path_extension' with various path formats "
        "to ensure it correctly identifies the file extension(s)."
    )
    assert get_source_path_extension(path=path) == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)


class TestGenerateUniqueValuesByRegex:
    def test_generates_correct_number_of_values(self, rp_logger):
        rp_logger.info(
            "Test that 'generate_unique_values_by_regex' returns the correct number of values"
        )
        result = generate_unique_values_by_regex(r"[A-Z]{3}", 10)
        assert len(result) == 10
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_all_values_are_unique(self, rp_logger):
        rp_logger.info(
            "Test that 'generate_unique_values_by_regex' returns only unique values"
        )
        result = generate_unique_values_by_regex(r"[A-Z]{3}", 50)
        assert len(result) == len(set(result))
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_values_match_regex_pattern(self, rp_logger):
        rp_logger.info(
            "Test that all generated values match the provided regex pattern"
        )
        pattern = r"CUST-[0-9]{4}"
        result = generate_unique_values_by_regex(pattern, 20)
        import re
        compiled = re.compile(f"^{pattern}$")
        for value in result:
            assert compiled.match(value), f"Value {value!r} does not match pattern {pattern!r}"
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_raises_when_size_exceeds_pattern_space(self, rp_logger):
        rp_logger.info(
            "Test that ValueError is raised when size exceeds the pattern space"
        )
        pattern = r"[AB]"  # pattern space = 2
        with pytest.raises(ValueError) as error:
            generate_unique_values_by_regex(pattern, 5)
        assert str(error.value) == (
            "The regex pattern '[AB]' can only generate 2 unique values, "
            "which is less than the required 5."
        )
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_exact_pattern_space_equals_size(self, rp_logger):
        rp_logger.info(
            "Test generation when the size equals the entire pattern space"
        )
        pattern = r"[0-9]{2}"  # pattern space = 100
        result = generate_unique_values_by_regex(pattern, 100)
        assert len(result) == 100
        assert len(set(result)) == 100
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_large_pattern_space_random_generation_path(self, rp_logger):
        rp_logger.info(
            "Test the random generation path when pattern space exceeds the threshold"
        )
        pattern = r"[A-Za-z0-9]{10}"  # pattern space >> PATTERN_SPACE_THRESHOLD
        result = generate_unique_values_by_regex(pattern, 500)
        assert len(result) == 500
        assert len(set(result)) == 500
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_raises_when_max_attempts_exceeded(self, rp_logger):
        rp_logger.info(
            "Test that ValueError is raised when consecutive duplicate attempts exceed the limit"
        )
        pattern = r"[AB]"
        max_attempts = 3
        with patch("syngen.ml.utils.utils.exrex") as mock_exrex:
            mock_exrex.count.return_value = 2_000_000  # Force large pattern space path
            mock_exrex.getone.return_value = "A"  # Always returns the same value
            with pytest.raises(ValueError, match="Could not generate"):
                generate_unique_values_by_regex(pattern, 5, max_attempts_per_batch=max_attempts)
        rp_logger.info(SUCCESSFUL_MESSAGE)


class TestIsNumberRegexPattern:
    def test_integer_pattern_returns_true(self, rp_logger):
        rp_logger.info(
            "Test that a pattern generating only integers is recognized as numeric"
        )
        assert is_number_regex_pattern(r"[0-9]{3}") is True
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_float_pattern_returns_true(self, rp_logger):
        rp_logger.info(
            "Test that a pattern generating float numbers is recognized as numeric"
        )
        assert is_number_regex_pattern(r"[0-9]{1,3}\.[0-9]{2}") is True
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_negative_number_pattern_returns_true(self, rp_logger):
        rp_logger.info(
            "Test that a pattern generating negative numbers is recognized as numeric"
        )
        assert is_number_regex_pattern(r"-[1-9][0-9]{2}") is True
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_alpha_pattern_returns_false(self, rp_logger):
        rp_logger.info(
            "Test that a pattern generating alphabetic strings is not numeric"
        )
        assert is_number_regex_pattern(r"[A-Z]{5}") is False
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_mixed_alpha_numeric_pattern_returns_false(self, rp_logger):
        rp_logger.info(
            "Test that a pattern generating mixed alpha-numeric strings is not numeric"
        )
        assert is_number_regex_pattern(r"CUST-[0-9]{4}") is False
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_single_digit_pattern_returns_true(self, rp_logger):
        rp_logger.info(
            "Test that a single digit pattern is recognized as numeric"
        )
        assert is_number_regex_pattern(r"[0-9]") is True
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_pattern_with_leading_whitespace_returns_false(self, rp_logger):
        rp_logger.info(
            "Test that a pattern producing values with leading whitespace is not numeric"
        )
        assert is_number_regex_pattern(r" [0-9]{3}") is False
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_pattern_with_trailing_whitespace_returns_false(self, rp_logger):
        rp_logger.info(
            "Test that a pattern producing values with trailing whitespace is not numeric"
        )
        assert is_number_regex_pattern(r"[0-9]{3} ") is False
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_custom_sample_size(self, rp_logger):
        rp_logger.info(
            "Test that the function works correctly with a custom sample size"
        )
        assert is_number_regex_pattern(r"[0-9]{4}", sample_size=50) is True
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_fixed_numeric_string_returns_true(self, rp_logger):
        rp_logger.info(
            "Test that a fixed numeric string pattern is recognized as numeric"
        )
        assert is_number_regex_pattern(r"42") is True
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_fixed_non_numeric_string_returns_false(self, rp_logger):
        rp_logger.info(
            "Test that a fixed non-numeric string pattern is not numeric"
        )
        assert is_number_regex_pattern(r"abc") is False
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_positive_number_pattern_returns_true(self, rp_logger):
        rp_logger.info(
            "Test that a pattern generating positive numbers is recognized as numeric"
        )
        assert is_number_regex_pattern(r"\+[1-9][0-9]{2}") is True
        rp_logger.info(SUCCESSFUL_MESSAGE)

    def test_scientific_notation_pattern_returns_true(self, rp_logger):
        rp_logger.info(
            "Test that a pattern generating numbers in scientific notation "
            "is recognized as numeric"
        )
        assert is_number_regex_pattern(r"[1-9]\.[0-9]{2}e[1-9]") is True
        rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("path/to/file.csv", ".csv"),
        ("path/to/file.avro", ".avro"),
        ("path/to/file.xls", ".xls"),
        ("path/to/file.xlsx", ".xlsx"),
        (None, ".csv")
    ],
)
def test_get_source_path_extension_with_metadata(path, expected, rp_logger):
    """
    The helper should return the appropriate file extension(s)
    from the 'source' field in the metadata of the certain table
    """
    rp_logger.info(
        "Test the function 'get_source_path_extension' with the provided metadata "
        "to ensure it correctly identifies the file extension(s) from the 'source' field."
    )
    test_metadata = {
        "pk_test": {
            "train_settings": {
                "source": path,
                "drop_null": False,
                "epochs": 1,
                "reports": [],
                "row_limit": 800,
            },
            "infer_settings": {
                "reports": ["accuracy"],
                "random_seed": 1,
                "run_parallel": False,
                "size": 100,
            },
            "keys": {"pk_id": {"columns": ["Id"], "type": "PK"}},
            "format": {},
            "encryption": {}
        },
    }
    assert get_source_path_extension(table_name="pk_test", metadata=test_metadata) == expected
    rp_logger.info(SUCCESSFUL_MESSAGE)
