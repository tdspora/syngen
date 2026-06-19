import csv
import copy
import inspect
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from syngen.ml.validation_schema import CSVFormatSettingsSchema, ExcelFormatSettingsSchema

_LOAD_ONLY_FIELDS = frozenset({"skiprows", "engine", "on_bad_lines"})
_DELIMITER_ALIASES: Dict[str, str] = {"\\t": "\t"}


class FormatSettings:
    """
    Base singleton that owns the raw format-settings dict.

    The setter writes to the shared class-level `_format_settings` dict once.
    Subclasses override only the getter to return a filtered view of that dict —
    they never need their own setter.
    """

    _instance: Optional["FormatSettings"] = None
    _format_settings: Dict = {}

    def __new__(cls) -> "FormatSettings":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def format_settings(self) -> Dict:
        return FormatSettings._format_settings

    @format_settings.setter
    def format_settings(self, value: Dict):
        FormatSettings._format_settings = copy.deepcopy(value)


class CSVFormatSettings(FormatSettings):
    """
    Presents a filtered view of the shared format dict for CSV operations.
    Provides load- and save-specific transformations without mutating stored settings.
    """

    _instance: Optional["CSVFormatSettings"] = None

    @property
    def format_settings(self) -> Dict:
        return {
            k: v
            for k, v in FormatSettings._format_settings.items()
            if k in CSVFormatSettingsSchema._declared_fields
        }

    @format_settings.setter
    def format_settings(self, value: Dict):
        FormatSettings._format_settings = copy.deepcopy(value)

    @staticmethod
    def _resolve_quoting(quoting) -> int:
        """
        Convert a string quoting name to the corresponding csv.QUOTE_* constant.
        If `quoting` is already an integer it is returned unchanged.
        An unrecognised string or `None` falls back to `csv.QUOTE_MINIMAL`.
        """
        if isinstance(quoting, int):
            return quoting
        quoting_map = {
            "minimal": csv.QUOTE_MINIMAL,
            "all": csv.QUOTE_ALL,
            "non-numeric": csv.QUOTE_NONNUMERIC,
            "none": csv.QUOTE_NONE,
        }
        return (
            quoting_map.get(quoting.lower(), csv.QUOTE_MINIMAL) if quoting else csv.QUOTE_MINIMAL
        )

    @property
    def load_format_settings(self) -> Dict:
        """
        Return a copy of the CSV-filtered format suitable for `pd.read_csv`.
        Quoting string values are resolved to integer constants.
        The shared format dict is never mutated.
        """
        params = copy.deepcopy(self.format_settings)
        if "quoting" in params:
            params["quoting"] = self._resolve_quoting(params["quoting"])
        return params

    @property
    def save_format_settings(self) -> Dict:
        """
        Return a copy of the CSV-filtered format suitable for `pd.DataFrame.to_csv`.

        Transformations applied (shared format dict is never mutated):
        - Load-only keys (`skiprows`, `engine`, `on_bad_lines`) are removed.
        - `sep`: `_DELIMITER_ALIASES` expansion applied first; if the result is
          longer than one character it is set to "," with a warning.
        - `header`: `None` -> `False`; any other value -> `True`.
        - `na_values`: popped; if non-empty the first element becomes `na_rep`
          with a warning.
        - `quoting`: converted to int via `_resolve_quoting`.
        - Keys not accepted by `pd.DataFrame.to_csv` are removed.
        """
        params = copy.deepcopy(self.format_settings)

        for key in _LOAD_ONLY_FIELDS:
            params.pop(key, None)

        if "sep" in params:
            sep = _DELIMITER_ALIASES.get(params["sep"], params["sep"])
            if len(sep) > 1:
                logger.warning(
                    "As the length of the value of the parameter 'separator' is more than "
                    "1 character, the 'separator' will be set to ',' in accordance with "
                    "the standard 'RFC 4180'"
                )
                sep = ","
            params["sep"] = sep

        if "header" in params:
            params["header"] = params["header"] is not None

        na_values: Optional[list] = params.pop("na_values", None)
        if na_values:
            logger.warning(
                "Since the 'na_values' parameter in the 'format' sections is not empty, "
                "the missing values will be filled with "
                "the first value from the 'na_values' parameter"
            )
            params["na_rep"] = na_values[0]

        if "quoting" in params:
            params["quoting"] = self._resolve_quoting(params["quoting"])

        valid_parameters = inspect.signature(pd.DataFrame.to_csv).parameters
        return {k: v for k, v in params.items() if k in valid_parameters}


class ExcelFormatSettings(FormatSettings):
    """
    Presents a filtered view of the shared format dict for Excel operations.
    """

    _instance: Optional["ExcelFormatSettings"] = None

    @property
    def format_settings(self) -> Dict:
        return {
            k: v
            for k, v in FormatSettings._format_settings.items()
            if k in ExcelFormatSettingsSchema._declared_fields
        }

    @format_settings.setter
    def format_settings(self, value: Dict):
        FormatSettings._format_settings = copy.deepcopy(value)

    @property
    def sheet_name(self):
        return FormatSettings._format_settings.get("sheet_name", 0)

    @property
    def load_format_settings(self) -> Dict:
        """
        Return format params filtered to fields valid for pd.read_excel.
        The shared format dict is never mutated.
        """
        return {
            k: v
            for k, v in FormatSettings._format_settings.items()
            if k in ExcelFormatSettingsSchema._declared_fields
        }


def set_format_settings(format_dict: Dict):
    FormatSettings().format_settings = format_dict
