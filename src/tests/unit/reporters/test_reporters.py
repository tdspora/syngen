import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from syngen.ml.reporters import AccuracyReporter, SampleAccuracyReporter
from syngen.ml.utils import convert_date_to_timestamp
from syngen.ml.vae.models.dataset import Dataset
from tests.conftest import SUCCESSFUL_MESSAGE

TABLE_NAME = "salesterritoryhistory"
KEY_COLUMNS = ["businessentityid", "territoryid", "startdate"]
DATE_FORMAT = "%Y-%m-%d"


def _get_dataframe() -> pd.DataFrame:
    """
    A table shaped like 'salesterritoryhistory' from AdventureWorks:
    the key includes the date column 'startdate'; 'enddate' is a date column outside the key
    """
    size = 200
    start = datetime.date(2020, 1, 1)
    return pd.DataFrame({
        "businessentityid": [i % 20 + 1 for i in range(size)],
        "territoryid": [i % 10 + 1 for i in range(size)],
        "startdate": [
            (start + datetime.timedelta(days=i)).strftime(DATE_FORMAT) for i in range(size)
        ],
        "enddate": [
            (start + datetime.timedelta(days=i + 30)).strftime(DATE_FORMAT)
            for i in range(size)
        ],
        "salesquota": [i * 1.5 for i in range(size)],
    })


def _get_metadata(key_type: str) -> dict:
    key = {"type": key_type, "columns": KEY_COLUMNS}
    if key_type == "FK":
        key["references"] = {"table": "parent_table", "columns": ["a", "b", "c"]}
    return {TABLE_NAME: {"keys": {f"{TABLE_NAME}_{key_type.lower()}": key}}}


def _get_dataset(df: pd.DataFrame, metadata: dict) -> Dataset:
    dataset = Dataset(
        df=df.copy(),
        schema={"fields": {}, "format": "CSV"},
        metadata=metadata,
        table_name=TABLE_NAME,
        paths={"initial_order_of_columns_path": "mock_path.pkl"},
        main_process="train",
    )
    dataset.launch_detection()
    return dataset


def _get_paths(tmp_path) -> dict:
    return {
        "path_to_flatten_metadata": str(tmp_path / "flatten_metadata.json"),
        "dataset_pickle_path": str(tmp_path / "dataset.pkl"),
    }


@pytest.mark.parametrize("key_type", ["PK", "UQ", "FK"])
def test_preprocess_data_skips_date_column_in_key(key_type, tmp_path, rp_logger):
    rp_logger.info(
        "Test that 'Reporter.preprocess_data' doesn't fail when a date column "
        "is a part of the %s key and converts only the date columns kept in the report",
        key_type,
    )
    df = _get_dataframe()
    metadata = _get_metadata(key_type)
    dataset = _get_dataset(df, metadata)
    assert "startdate" in dataset.date_mapping

    reporter = AccuracyReporter(
        table_name=TABLE_NAME,
        paths=_get_paths(tmp_path),
        config={},
        metadata=metadata,
    )
    with patch("syngen.ml.reporters.reporters.fetch_config", return_value=dataset):
        (
            original,
            synthetic,
            float_columns,
            int_columns,
            categorical_columns,
            date_columns,
        ) = reporter.preprocess_data(df.copy(), df.copy())

    assert date_columns == {"enddate"}
    assert "enddate" in int_columns
    assert float_columns == {"salesquota"}
    for data in (original, synthetic):
        assert set(data.columns) == {"enddate", "salesquota"}
        assert list(data["enddate"]) == [
            convert_date_to_timestamp(value, DATE_FORMAT, None) for value in df["enddate"]
        ]
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "reporter_class, test_class_name",
    [
        (AccuracyReporter, "AccuracyTest"),
        (SampleAccuracyReporter, "SampleAccuracyTest"),
    ],
)
def test_report_with_date_column_in_composite_pk(
    reporter_class, test_class_name, tmp_path, rp_logger
):
    rp_logger.info(
        "Test that '%s.report' runs for a table whose composite primary key "
        "includes a date column and leaves this column out of the report",
        reporter_class.__name__,
    )
    df = _get_dataframe()
    metadata = _get_metadata("PK")
    dataset = _get_dataset(df, metadata)

    reporter = reporter_class(
        table_name=TABLE_NAME,
        paths=_get_paths(tmp_path),
        config={"reports": ["accuracy", "sample"]},
        metadata=metadata,
    )
    with (
        patch("syngen.ml.reporters.reporters.fetch_config", return_value=dataset),
        patch.object(
            reporter_class, "_extract_report_data", return_value=(df.copy(), df.copy())
        ),
        patch(f"syngen.ml.reporters.reporters.{test_class_name}") as mock_test_class,
    ):
        reporter.report()

    mock_test_class.return_value.report.assert_called_once()
    report_kwargs = mock_test_class.return_value.report.call_args.kwargs
    assert report_kwargs["date_columns"] == ["enddate"]
    assert sorted(report_kwargs["cont_columns"]) == ["enddate", "salesquota"]
    call_args = mock_test_class.call_args
    frames = (
        list(call_args.args[:2])
        or [call_args.kwargs["original"], call_args.kwargs["synthetic"]]
    )
    for data in frames:
        assert "startdate" not in data.columns
        assert pd.api.types.is_numeric_dtype(data["enddate"])
    rp_logger.info(SUCCESSFUL_MESSAGE)
