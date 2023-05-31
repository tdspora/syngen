import pytest

import pandas as pd
import pandavro as pdx

from syngen.ml.vae.models.dataset import Dataset


CSV_SCHEMA = {"fields": {}, "format": "CSV"}

AVRO_SCHEMA = {
    "fields": {
        "UUIDv1": "string",
        "UUIDv2": "string",
        "UUIDv3": "string",
        "UUIDv4": "string",
        "UUIDv5": "string",
        "ULID": "string"
    },
    "format": "Avro"
}

@pytest.mark.parametrize(
    "path_to_test_table", [
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns.csv",
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns_with_missing_values.csv"
])
def test_is_valid_uuid_defined_in_csv_table_without_missing_values(path_to_test_table):
    df = pd.read_csv(path_to_test_table)
    dataset = Dataset(
        df=df,
        schema=CSV_SCHEMA,
        metadata={"table_name": "test_table"},
        table_name="test_table",
        fk_kde_path="path/to/kde",
    )

    dataset._set_uuid_columns(df)
    assert dataset.uuid_columns == {"UUIDv1", "UUIDv2", "UUIDv3", "UUIDv4", "UUIDv5", "ULID"}
    assert dataset.uuid_columns_types == {
        "UUIDv1": 1, "UUIDv2": 2, "UUIDv3": 3, "UUIDv4": 4, "UUIDv5": 5, "ULID": "ulid"
    }

@pytest.mark.parametrize(
    "path_to_test_table", [
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns.avro",
        "./tests/unit/dataset/fixtures/table_with_diff_uuid_columns_with_missing_values.avro"
])
def test_is_valid_uuid_defined_in_csv_table_without_missing_values(path_to_test_table):
    df = pdx.from_avro(path_to_test_table)
    dataset = Dataset(
        df=df,
        schema=AVRO_SCHEMA,
        metadata={"table_name": "test_table"},
        table_name="test_table",
        fk_kde_path="path/to/kde",
    )

    dataset._set_uuid_columns(df)
    assert dataset.uuid_columns == {"UUIDv1", "UUIDv2", "UUIDv3", "UUIDv4", "UUIDv5", "ULID"}
    assert dataset.uuid_columns_types == {
        "UUIDv1": 1, "UUIDv2": 2, "UUIDv3": 3, "UUIDv4": 4, "UUIDv5": 5, "ULID": "ulid"
    }

