"""
Unit tests for 'syngen.ml.format_settings':
  - 'FormatSettings' singleton and shared storage
  - 'CSVFormatSettings.format_settings'
  - 'CSVFormatSettings._resolve_quoting'
  - 'CSVFormatSettings.load_format_settings'
  - 'CSVFormatSettings.save_format_settings'
  - 'ExcelFormatSettings.format_settings'
  - 'ExcelFormatSettings.sheet_name'
  - 'ExcelFormatSettings.load_format_settings'
  - 'set_format_settings' as single write point
  - CSVLoader does not mutate format settings during load_data
  - ExcelLoader does not mutate format settings during load_data
  - 'load_saved_artifact' context manager
"""
import csv
import pytest

from syngen.ml.format_settings import (
    FormatSettings,
    CSVFormatSettings,
    ExcelFormatSettings,
    set_format_settings,
    load_saved_artifact,
)
from tests.conftest import SUCCESSFUL_MESSAGE


@pytest.fixture(autouse=True)
def reset_format_settings():
    set_format_settings({})
    yield
    set_format_settings({})


# ---------------------------------------------------------------------------
# 'FormatSettings' — singleton and shared storage
# ---------------------------------------------------------------------------

def test_format_settings_singleton_returns_same_instance(rp_logger):
    rp_logger.info("Testing 'FormatSettings' singleton behavior")
    assert FormatSettings() is FormatSettings()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_csv_format_settings_singleton_returns_same_instance(rp_logger):
    rp_logger.info("Testing 'CSVFormatSettings' singleton behavior")
    assert CSVFormatSettings() is CSVFormatSettings()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_excel_format_settings_singleton_returns_same_instance(rp_logger):
    rp_logger.info("Testing 'ExcelFormatSettings' singleton behavior")
    assert ExcelFormatSettings() is ExcelFormatSettings()


def test_set_format_settings_updates_base_getter(rp_logger):
    rp_logger.info("Testing 'set_format_settings' updates base getter")
    set_format_settings({"sep": "|"})
    assert FormatSettings().format_settings == {"sep": "|"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_set_format_settings_deep_copies_input(rp_logger):
    rp_logger.info("Testing 'set_format_settings' deep copies input")
    original = {"sep": "|"}
    set_format_settings(original)
    original["sep"] = "X"
    assert FormatSettings().format_settings["sep"] == "|"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_set_format_settings_empty_dict_clears_storage(rp_logger):
    rp_logger.info("Testing 'set_format_settings' clears storage when given empty dict")
    set_format_settings({"sep": "|"})
    set_format_settings({})
    assert FormatSettings().format_settings == {}
    rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# 'CSVFormatSettings.format_settings' — filter-on-read
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("format_settings", [
    # sep — separator character(s)
    pytest.param({"sep": ","}),
    pytest.param({"sep": "|"}),
    pytest.param({"sep": "\t"}),
    pytest.param({"sep": ";"}),
    pytest.param({"sep": ":::"}),
    pytest.param({"sep": None}),
    # quotechar — single quoting character
    pytest.param({"quotechar": '"'}),
    pytest.param({"quotechar": "'"}),
    pytest.param({"quotechar": "|"}),
    # quoting — quoting strategy
    pytest.param({"quoting": "minimal"}),
    pytest.param({"quoting": "all"}),
    pytest.param({"quoting": "non-numeric"}),
    pytest.param({"quoting": "none"}),
    pytest.param({"quoting": "Minimal"}),
    pytest.param({"quoting": "All"}),
    pytest.param({"quoting": "Non-numeric"}),
    pytest.param({"quoting": "None"}),
    # escapechar — single escape character
    pytest.param({"escapechar": "\\"}),
    pytest.param({"escapechar": "/"}),
    # encoding — file encoding
    pytest.param({"encoding": "utf-8"}),
    pytest.param({"encoding": "latin-1"}),
    pytest.param({"encoding": "utf-16"}),
    pytest.param({"encoding": None}),
    # header — row number(s) used as column names
    pytest.param({"header": 0}),
    pytest.param({"header": 1}),
    pytest.param({"header": "infer"}),
    pytest.param({"header": [0, 1]}),
    pytest.param({"header": None}),
    # skiprows — rows to skip at the start of the file
    pytest.param({"skiprows": 0}),
    pytest.param({"skiprows": 1}),
    pytest.param({"skiprows": 5}),
    pytest.param({"skiprows": [0, 1, 2]}),
    # on_bad_lines — behaviour when a bad line is encountered
    pytest.param({"on_bad_lines": "error"}),
    pytest.param({"on_bad_lines": "warn"}),
    pytest.param({"on_bad_lines": "skip"}),
    # engine — CSV parsing engine
    pytest.param({"engine": "c"}),
    pytest.param({"engine": "python"}),
    pytest.param({"engine": "pyarrow"}),
    pytest.param({"engine": None}),
    # na_values — additional strings recognised as NA/NaN
    pytest.param({"na_values": ["-", "N/A", ""]}),
    pytest.param({"na_values": []}),
    pytest.param({"na_values": None}),
    # combinations of multiple fields
    pytest.param(
        {"sep": "|", "quoting": "none", "encoding": "utf-8"},
    ),
    pytest.param(
        {"sep": "\t", "encoding": "utf-8", "on_bad_lines": "skip"},
    ),
    pytest.param(
        {"sep": ",", "quotechar": '"', "quoting": "minimal", "escapechar": "\\",
         "encoding": "utf-8", "header": 0, "skiprows": 1,
         "on_bad_lines": "warn", "engine": "python", "na_values": ["-", "N/A"]},
    ),
])
def test_csv_format_settings_valid_settings(rp_logger, format_settings):
    rp_logger.info(f"Testing 'CSVFormatSettings' with valid settings: {format_settings}")
    set_format_settings(format_settings)
    assert CSVFormatSettings().format_settings == format_settings
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_csv_format_settings_excludes_non_csv_format_settings(rp_logger):
    rp_logger.info("Testing 'CSVFormatSettings' excludes non-CSV format settings")
    set_format_settings({"sep": "|", "sheet_name": "Sheet1"})
    result = CSVFormatSettings().format_settings
    assert "sheet_name" not in result
    assert result == {"sep": "|"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_csv_format_settings_empty_when_no_csv_format_settings(rp_logger):
    rp_logger.info("Testing 'CSVFormatSettings' returns empty when no CSV format settings")
    set_format_settings({"sheet_name": None})
    assert CSVFormatSettings().format_settings == {}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_csv_format_settings_returns_deep_copy(rp_logger):
    rp_logger.info("Testing 'CSVFormatSettings' returns a deep copy")
    set_format_settings({"sep": ","})
    result = CSVFormatSettings().format_settings
    result["sep"] = "X"
    assert CSVFormatSettings().format_settings["sep"] == ","
    rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# 'ExcelFormatSettings.format_settings' — filter-on-read
# ---------------------------------------------------------------------------

def test_excel_format_settings_includes_excel_valid_settings(rp_logger):
    rp_logger.info("Testing 'ExcelFormatSettings' includes Excel valid settings")
    set_format_settings({"sheet_name": "Sheet1"})
    assert ExcelFormatSettings().format_settings == {"sheet_name": "Sheet1"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_excel_format_settings_excludes_non_excel_format_settings(rp_logger):
    rp_logger.info("Testing 'ExcelFormatSettings' excludes non-Excel format settings")
    set_format_settings({"sep": "|", "sheet_name": None})
    result = ExcelFormatSettings().format_settings
    assert "sep" not in result
    assert result == {"sheet_name": None}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_excel_format_settings_empty_when_no_excel_format_settings(rp_logger):
    rp_logger.info("Testing 'ExcelFormatSettings' returns empty when no Excel format settings")
    set_format_settings({"sep": "|", "quoting": "all"})
    assert ExcelFormatSettings().format_settings == {}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_excel_format_settings_returns_deep_copy(rp_logger):
    rp_logger.info("Testing 'ExcelFormatSettings' returns a deep copy")
    set_format_settings({"sheet_name": "Sheet1"})
    result = ExcelFormatSettings().format_settings
    result["sheet_name"] = "Other"
    assert ExcelFormatSettings().format_settings["sheet_name"] == "Sheet1"
    rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# 'ExcelFormatSettings.sheet_name'
# ---------------------------------------------------------------------------

def test_excel_sheet_name_defaults_to_zero(rp_logger):
    rp_logger.info("Testing 'ExcelFormatSettings.sheet_name' defaults to zero")
    assert ExcelFormatSettings().sheet_name == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_excel_sheet_name_returns_set_value(rp_logger):
    rp_logger.info("Testing 'ExcelFormatSettings.sheet_name' returns set value")
    set_format_settings({"sheet_name": "Sheet1"})
    assert ExcelFormatSettings().sheet_name == "Sheet1"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_excel_sheet_name_none_is_preserved(rp_logger):
    rp_logger.info("Testing 'ExcelFormatSettings.sheet_name' preserves `None` value")
    set_format_settings({"sheet_name": None})
    assert ExcelFormatSettings().sheet_name is None
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_excel_sheet_name_integer_index(rp_logger):
    rp_logger.info("Testing 'ExcelFormatSettings.sheet_name' preserves integer index")
    set_format_settings({"sheet_name": 2})
    assert ExcelFormatSettings().sheet_name == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# 'ExcelFormatSettings.load_format_settings'
# ---------------------------------------------------------------------------

def test_excel_load_format_settings_returns_excel_valid_settings(rp_logger):
    rp_logger.info(
        "Testing 'ExcelFormatSettings.load_format_settings' returns Excel valid settings"
    )
    set_format_settings({"sheet_name": "Sheet1"})
    assert ExcelFormatSettings().load_format_settings == {"sheet_name": "Sheet1"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_excel_load_format_settings_excludes_csv_format_settings(rp_logger):
    rp_logger.info(
        "Testing 'ExcelFormatSettings.load_format_settings' excludes CSV format settings"
    )
    set_format_settings({"sep": "|", "sheet_name": 0})
    result = ExcelFormatSettings().load_format_settings
    assert "sep" not in result
    assert result == {"sheet_name": 0}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_excel_load_format_settings_empty_when_no_excel_format_settings(rp_logger):
    rp_logger.info(
        "Testing 'ExcelFormatSettings.load_format_settings' "
        "returns empty when no Excel format settings"
    )
    set_format_settings({"sep": ",", "quoting": "all"})
    assert ExcelFormatSettings().load_format_settings == {}
    rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# 'set_format_settings' — single write point; both subclasses see correct views
# ---------------------------------------------------------------------------

def test_set_format_settings_csv_and_excel_see_their_own_keys(rp_logger):
    rp_logger.info(
        "Testing 'set_format_settings' ensures CSV and Excel see their own format settings"
    )
    set_format_settings({"sep": "|", "sheet_name": "Sheet1"})
    assert CSVFormatSettings().format_settings == {"sep": "|"}
    assert ExcelFormatSettings().format_settings == {"sheet_name": "Sheet1"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_set_format_settings_reset_clears_both_views(rp_logger):
    rp_logger.info(
        "Testing 'set_format_settings' reset clears both CSV and Excel views of format settings"
    )
    set_format_settings({"sep": "|", "sheet_name": "Sheet1"})
    assert CSVFormatSettings().format_settings == {"sep": "|"}
    assert ExcelFormatSettings().format_settings == {"sheet_name": "Sheet1"}
    set_format_settings({})
    assert CSVFormatSettings().format_settings == {}
    assert ExcelFormatSettings().format_settings == {}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_set_format_settings_overwrite_updates_both_views(rp_logger):
    rp_logger.info(
        "Testing 'set_format_settings' overwrites the existing format settings "
        "for both 'CSVFormatSettings' and 'ExcelFormatSettings'"
    )
    set_format_settings({"sep": "|", "sheet_name": "Sheet1"})
    set_format_settings({"sep": ",", "sheet_name": "Sheet2"})
    assert CSVFormatSettings().format_settings["sep"] == ","
    assert ExcelFormatSettings().format_settings["sheet_name"] == "Sheet2"
    assert FormatSettings().format_settings == {"sep": ",", "sheet_name": "Sheet2"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# 'CSVFormatSettings._resolve_quoting'
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("quoting, expected", [
    (0, csv.QUOTE_MINIMAL),
    (1, csv.QUOTE_ALL),
    (2, csv.QUOTE_NONNUMERIC),
    (3, csv.QUOTE_NONE),
    ("minimal", csv.QUOTE_MINIMAL),
    ("all", csv.QUOTE_ALL),
    ("non-numeric", csv.QUOTE_NONNUMERIC),
    ("none", csv.QUOTE_NONE),
    ("MINIMAL", csv.QUOTE_MINIMAL),
    ("ALL", csv.QUOTE_ALL),
    ("NON-NUMERIC", csv.QUOTE_NONNUMERIC),
    ("NONE", csv.QUOTE_NONE),
    (None, csv.QUOTE_MINIMAL),
    ("test", csv.QUOTE_MINIMAL),
])
def test_resolve_quoting(quoting, expected, rp_logger):
    rp_logger.info(
        f"Testing 'CSVFormatSettings._resolve_quoting' with quoting equals to {quoting}"
    )
    assert CSVFormatSettings._resolve_quoting(quoting) == expected


# ---------------------------------------------------------------------------
# 'CSVFormatSettings.load_format_settings'
# ---------------------------------------------------------------------------

def test_load_format_settings_returns_deep_copy(rp_logger):
    rp_logger.info("Testing 'CSVFormatSettings.load_format_settings' returns a deep copy")
    set_format_settings({"sep": ","})
    result = CSVFormatSettings().load_format_settings
    result["sep"] = "X"
    assert CSVFormatSettings().format_settings["sep"] == ","
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("quoting_input, expected_int", [
    # canonical lowercase strings → their csv module constants
    pytest.param("minimal", csv.QUOTE_MINIMAL),
    pytest.param("all", csv.QUOTE_ALL),
    pytest.param("non-numeric", csv.QUOTE_NONNUMERIC),
    pytest.param("none", csv.QUOTE_NONE),
    # mixed-case variants — _resolve_quoting is case-insensitive
    pytest.param("Minimal", csv.QUOTE_MINIMAL),
    pytest.param("All", csv.QUOTE_ALL),
    pytest.param("Non-Numeric", csv.QUOTE_NONNUMERIC),
    pytest.param("None", csv.QUOTE_NONE),
    pytest.param("MINIMAL", csv.QUOTE_MINIMAL),
    pytest.param("ALL", csv.QUOTE_ALL),
    pytest.param("NON-NUMERIC", csv.QUOTE_NONNUMERIC),
    pytest.param("NONE", csv.QUOTE_NONE),
    # integer values passed through unchanged
    pytest.param(csv.QUOTE_MINIMAL, csv.QUOTE_MINIMAL),
    pytest.param(csv.QUOTE_ALL, csv.QUOTE_ALL),
    pytest.param(csv.QUOTE_NONNUMERIC, csv.QUOTE_NONNUMERIC),
    pytest.param(csv.QUOTE_NONE, csv.QUOTE_NONE),
])
def test_load_format_settings_quoting_string_converted_to_int(
    rp_logger, quoting_input, expected_int
):
    rp_logger.info(
        f"Testing 'CSVFormatSettings.load_format_settings' converts quoting "
        f"{quoting_input!r} to int {expected_int}"
    )
    set_format_settings({"quoting": quoting_input})
    assert CSVFormatSettings().load_format_settings["quoting"] == expected_int
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_format_settings_preserves_load_only_fields(rp_logger):
    rp_logger.info("Testing 'CSVFormatSettings.load_format_settings' preserves load-only fields")
    set_format_settings(
        {"skiprows": 2, "engine": "python", "on_bad_lines": "skip", "sep": ","}
    )
    result = CSVFormatSettings().load_format_settings
    assert result["skiprows"] == 2
    assert result["engine"] == "python"
    assert result["on_bad_lines"] == "skip"
    rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# 'CSVFormatSettings.save_format_settings'
# ---------------------------------------------------------------------------

def test_save_format_settings_returns_deep_copy(rp_logger):
    rp_logger.info("Testing 'CSVFormatSettings.save_format_settings' returns a deep copy")
    set_format_settings({"sep": ","})
    result = CSVFormatSettings().save_format_settings
    result["sep"] = "X"
    assert CSVFormatSettings().format_settings["sep"] == ","
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_format_settings_removes_load_only_fields(rp_logger):
    rp_logger.info("Testing 'CSVFormatSettings.save_format_settings' removes load-only fields")
    set_format_settings(
        {"skiprows": 1, "engine": "python", "on_bad_lines": "skip", "sep": ","}
    )
    result = CSVFormatSettings().save_format_settings
    assert "skiprows" not in result
    assert "engine" not in result
    assert "on_bad_lines" not in result
    assert result["sep"] == ","
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_format_settings_expands_delimiter_alias(rp_logger):
    rp_logger.info("Testing 'CSVFormatSettings.save_format_settings' expands delimiter alias")
    set_format_settings({"sep": "\\t"})
    assert CSVFormatSettings().save_format_settings["sep"] == "\t"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_format_settings_normalizes_multi_char_sep(caplog, rp_logger):
    rp_logger.info("Testing 'CSVFormatSettings.save_format_settings' normalizes multi-char sep")
    set_format_settings({"sep": ":::"})
    with caplog.at_level("WARNING"):
        result = CSVFormatSettings().save_format_settings
    assert result["sep"] == ","
    assert "separator" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_format_settings_keeps_single_char_sep(rp_logger):
    rp_logger.info("Testing 'CSVFormatSettings.save_format_settings' keeps a single-char sep")
    set_format_settings({"sep": "|"})
    assert CSVFormatSettings().save_format_settings["sep"] == "|"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_format_settings_header_none_becomes_false(rp_logger):
    rp_logger.info(
        "Testing 'CSVFormatSettings.save_format_settings' converts header=None to False"
    )
    set_format_settings({"header": None})
    assert CSVFormatSettings().save_format_settings["header"] is False
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_format_settings_header_value_becomes_true(rp_logger):
    rp_logger.info(
        "Testing 'CSVFormatSettings.save_format_settings' converts header value to True"
    )
    set_format_settings({"header": 0})
    assert CSVFormatSettings().save_format_settings["header"] is True
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_format_settings_header_absent_not_added(rp_logger):
    rp_logger.info(
        "Testing 'CSVFormatSettings.save_format_settings' does not add header if absent"
    )
    assert "header" not in CSVFormatSettings().save_format_settings
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_format_settings_na_values_becomes_na_rep(caplog, rp_logger):
    rp_logger.info(
        "Testing 'CSVFormatSettings.save_format_settings' converts 'na_values' to 'na_rep'"
    )
    set_format_settings({"na_values": ["-", "Missing"]})
    with caplog.at_level("WARNING"):
        result = CSVFormatSettings().save_format_settings
    assert "na_values" not in result
    assert result["na_rep"] == "-"
    assert "na_values" in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_format_settings_empty_na_values_does_not_set_na_rep(rp_logger):
    rp_logger.info(
        "Testing 'CSVFormatSettings.save_format_settings' does not set 'na_rep' "
        "if 'na_values' is empty"
    )
    set_format_settings({"na_values": []})
    result = CSVFormatSettings().save_format_settings
    assert "na_values" not in result
    assert "na_rep" not in result
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("quoting_input, expected_int", [
    # canonical lowercase strings → their csv module constants
    pytest.param("minimal", csv.QUOTE_MINIMAL),
    pytest.param("all", csv.QUOTE_ALL),
    pytest.param("non-numeric", csv.QUOTE_NONNUMERIC),
    pytest.param("none", csv.QUOTE_NONE),
    # mixed-case variants — _resolve_quoting is case-insensitive
    pytest.param("Minimal", csv.QUOTE_MINIMAL),
    pytest.param("All", csv.QUOTE_ALL),
    pytest.param("Non-Numeric", csv.QUOTE_NONNUMERIC),
    pytest.param("None", csv.QUOTE_NONE),
    pytest.param("MINIMAL", csv.QUOTE_MINIMAL),
    pytest.param("ALL", csv.QUOTE_ALL),
    pytest.param("NON-NUMERIC", csv.QUOTE_NONNUMERIC),
    pytest.param("NONE", csv.QUOTE_NONE),
    # integer values passed through unchanged
    pytest.param(csv.QUOTE_MINIMAL, csv.QUOTE_MINIMAL),
    pytest.param(csv.QUOTE_ALL, csv.QUOTE_ALL),
    pytest.param(csv.QUOTE_NONNUMERIC, csv.QUOTE_NONNUMERIC),
    pytest.param(csv.QUOTE_NONE, csv.QUOTE_NONE),
])
def test_save_format_settings_quoting_string_converted_to_int(
    rp_logger, quoting_input, expected_int
):
    rp_logger.info(
        f"Testing 'CSVFormatSettings.save_format_settings' converts quoting "
        f"{quoting_input!r} to int {expected_int}"
    )
    set_format_settings({"quoting": quoting_input})
    assert CSVFormatSettings().save_format_settings["quoting"] == expected_int
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_save_format_settings_does_not_mutate_stored_settings(rp_logger):
    rp_logger.info(
        "Testing 'CSVFormatSettings.save_format_settings' does not mutate stored settings"
    )
    set_format_settings({
        "sep": ":::",
        "skiprows": 1,
        "na_values": ["-"],
        "header": None,
        "quoting": "all",
    })
    CSVFormatSettings().save_format_settings
    stored = CSVFormatSettings().format_settings
    assert stored["sep"] == ":::"
    assert stored["skiprows"] == 1
    assert stored["na_values"] == ["-"]
    assert stored["header"] is None
    assert stored["quoting"] == "all"
    rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# 'load_saved_artifact' — CSV branch (detected via CSVFormatSettings)
# ---------------------------------------------------------------------------

def test_load_saved_artifact_normalizes_multi_char_sep_inside_context(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' sets the separator to ',' inside context "
        "when the original separator is longer than one character"
    )
    set_format_settings({"sep": "|||"})
    with load_saved_artifact():
        assert FormatSettings().format_settings["sep"] == ","
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_saved_artifact_restores_original_format_after_context(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' restores original format settings after context exits"
    )
    set_format_settings({"sep": "|||"})
    with load_saved_artifact():
        pass
    assert FormatSettings().format_settings["sep"] == "|||"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_saved_artifact_does_not_change_single_char_sep(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' does not alter the separator "
        "when it is already a single character"
    )
    set_format_settings({"sep": ";"})
    with load_saved_artifact():
        assert FormatSettings().format_settings["sep"] == ";"
    assert FormatSettings().format_settings["sep"] == ";"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_saved_artifact_preserves_other_format_fields_inside_context(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' preserves the non-sep format fields inside the context"
    )
    set_format_settings({"sep": "|||", "encoding": "utf-8", "quoting": "all"})
    with load_saved_artifact():
        fmt = FormatSettings().format_settings
        assert fmt["encoding"] == "utf-8"
        assert fmt["quoting"] == "all"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_saved_artifact_restores_format_after_exception(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' restores original format settings "
        "even when an exception is raised inside the context"
    )
    set_format_settings({"sep": "|||"})
    with pytest.raises(RuntimeError):
        with load_saved_artifact():
            raise RuntimeError("simulated error")
    assert FormatSettings().format_settings["sep"] == "|||"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_saved_artifact_no_op_when_format_is_empty(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' is a no-op when format settings are empty"
    )
    with load_saved_artifact():
        assert FormatSettings().format_settings == {}
    assert FormatSettings().format_settings == {}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("sep", ["|||", ":::", "\\|\\|"])
def test_load_saved_artifact_normalizes_various_multi_char_seps(sep, rp_logger):
    rp_logger.info(
        f"Testing 'load_saved_artifact' normalizes multi-char sep {sep!r} to ',' "
        "inside context and restores it after"
    )
    set_format_settings({"sep": sep})
    with load_saved_artifact():
        assert FormatSettings().format_settings["sep"] == ","
    assert FormatSettings().format_settings["sep"] == sep
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("load_only_field, value", [
    ("engine", "python"),
    ("engine", "c"),
    ("engine", "pyarrow"),
    ("on_bad_lines", "skip"),
    ("on_bad_lines", "warn"),
    ("on_bad_lines", "error"),
    ("skiprows", 1),
    ("skiprows", 3),
])
def test_load_saved_artifact_strips_each_load_only_field_inside_context(
    rp_logger, load_only_field, value
):
    rp_logger.info(
        f"Testing 'load_saved_artifact' strips '{load_only_field}' "
        f"with value {value!r} inside context"
    )
    set_format_settings({"sep": ",", load_only_field: value})
    with load_saved_artifact():
        assert load_only_field not in FormatSettings().format_settings
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_saved_artifact_strips_all_load_only_fields_when_all_present(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' strips 'engine', 'on_bad_lines', and 'skiprows' "
        "when all three are present"
    )
    set_format_settings({
        "sep": ",",
        "engine": "python",
        "on_bad_lines": "skip",
        "skiprows": 1,
    })
    with load_saved_artifact():
        fmt = FormatSettings().format_settings
        assert "engine" not in fmt
        assert "on_bad_lines" not in fmt
        assert "skiprows" not in fmt
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_saved_artifact_restores_load_only_fields_after_context(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' restores 'engine', 'on_bad_lines', and 'skiprows' "
        "after the context exits"
    )
    set_format_settings({
        "sep": ",",
        "engine": "python",
        "on_bad_lines": "skip",
        "skiprows": 1,
    })
    with load_saved_artifact():
        pass
    fmt = FormatSettings().format_settings
    assert fmt["engine"] == "python"
    assert fmt["on_bad_lines"] == "skip"
    assert fmt["skiprows"] == 1
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_saved_artifact_preserves_non_load_only_fields_when_load_only_present(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' preserves non-load-only fields inside context "
        "even when load-only fields are also present"
    )
    set_format_settings({
        "sep": ",",
        "encoding": "utf-8",
        "quoting": "all",
        "engine": "python",
        "on_bad_lines": "skip",
        "skiprows": 2,
    })
    with load_saved_artifact():
        fmt = FormatSettings().format_settings
        assert fmt["encoding"] == "utf-8"
        assert fmt["quoting"] == "all"
        assert fmt["sep"] == ","
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_saved_artifact_combines_sep_normalization_and_load_only_stripping(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' normalizes multi-char sep to ',' and strips "
        "load-only fields simultaneously"
    )
    set_format_settings({
        "sep": "|||",
        "encoding": "utf-8",
        "engine": "python",
        "on_bad_lines": "skip",
        "skiprows": 1,
    })
    with load_saved_artifact():
        fmt = FormatSettings().format_settings
        assert fmt["sep"] == ","
        assert "engine" not in fmt
        assert "on_bad_lines" not in fmt
        assert "skiprows" not in fmt
        assert fmt["encoding"] == "utf-8"
    restored = FormatSettings().format_settings
    assert restored["sep"] == "|||"
    assert restored["engine"] == "python"
    assert restored["on_bad_lines"] == "skip"
    assert restored["skiprows"] == 1
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_saved_artifact_restores_load_only_fields_after_exception(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' restores load-only fields after an exception "
        "is raised inside the context"
    )
    set_format_settings({
        "sep": ",",
        "engine": "python",
        "on_bad_lines": "skip",
        "skiprows": 1,
    })
    with pytest.raises(RuntimeError):
        with load_saved_artifact():
            raise RuntimeError("simulated error")
    fmt = FormatSettings().format_settings
    assert fmt["engine"] == "python"
    assert fmt["on_bad_lines"] == "skip"
    assert fmt["skiprows"] == 1
    rp_logger.info(SUCCESSFUL_MESSAGE)


# ---------------------------------------------------------------------------
# 'load_saved_artifact' — Excel branch (detected via ExcelFormatSettings)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sheet_name", [
    pytest.param("Customers"),
    pytest.param("Sheet1"),
    pytest.param(1),
    pytest.param(None),
])
def test_load_saved_artifact_excel_sets_sheet_name_to_zero_inside_context(
    rp_logger, sheet_name
):
    rp_logger.info(
        f"Testing 'load_saved_artifact' detects Excel format via 'ExcelFormatSettings' "
        f"and sets 'sheet_name' to 0 inside context when original 'sheet_name' is {sheet_name!r}"
    )
    set_format_settings({"sheet_name": sheet_name})
    with load_saved_artifact():
        assert FormatSettings().format_settings["sheet_name"] == 0
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("sheet_name", [
    pytest.param("Customers"),
    pytest.param("Sheet1"),
    pytest.param(1),
    pytest.param(None),
])
def test_load_saved_artifact_excel_restores_original_sheet_name_after_context(
    rp_logger, sheet_name
):
    rp_logger.info(
        f"Testing 'load_saved_artifact' restores original 'sheet_name' {sheet_name!r} "
        "after context exits"
    )
    set_format_settings({"sheet_name": sheet_name})
    with load_saved_artifact():
        pass
    assert FormatSettings().format_settings["sheet_name"] == sheet_name
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_saved_artifact_excel_does_not_strip_sheet_name_after_context(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' does not permanently remove 'sheet_name' "
        "from format settings after the context exits"
    )
    set_format_settings({"sheet_name": "Customers"})
    with load_saved_artifact():
        pass
    assert "sheet_name" in FormatSettings().format_settings
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_saved_artifact_excel_restores_format_after_exception(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' restores original 'sheet_name' after an "
        "exception is raised inside the context"
    )
    set_format_settings({"sheet_name": "Customers"})
    with pytest.raises(RuntimeError):
        with load_saved_artifact():
            raise RuntimeError("simulated error")
    assert FormatSettings().format_settings["sheet_name"] == "Customers"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_saved_artifact_csv_branch_when_no_excel_fields_present(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' applies CSV logic (load-only stripping, "
        "sep normalisation) when 'ExcelFormatSettings' returns empty — i.e. no Excel "
        "fields are present in format settings"
    )
    set_format_settings({"sep": "|||", "engine": "python", "on_bad_lines": "skip"})
    with load_saved_artifact():
        fmt = FormatSettings().format_settings
        assert fmt["sep"] == ","
        assert "engine" not in fmt
        assert "on_bad_lines" not in fmt
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_load_saved_artifact_excel_branch_skips_csv_logic(rp_logger):
    rp_logger.info(
        "Testing 'load_saved_artifact' does not apply CSV logic (load-only stripping, "
        "sep normalisation) when 'ExcelFormatSettings' is non-empty"
    )
    set_format_settings({"sheet_name": "Customers", "sep": "|||"})
    with load_saved_artifact():
        fmt = FormatSettings().format_settings
        assert fmt["sheet_name"] == 0
        assert fmt.get("sep") == "|||"
    rp_logger.info(SUCCESSFUL_MESSAGE)
