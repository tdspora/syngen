import pytest
from datetime import date, time

import numpy as np
from numpy import dtype
import pandas as pd
import pandavro as pdx
import fastavro

from syngen.ml.convertor import CSVConvertor, AvroConvertor
from syngen.ml.convertor.convertor import Convertor

from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME
from unittest.mock import patch


def _make_binary_convertor(df: pd.DataFrame, column: str) -> Convertor:
    """
    Build a minimal Convertor instance that exercises the binary cast path.
    AvroConvertor maps bytes → string (no binary path), so we use the base
    class directly and configure it to treat `column` as binary.
    'Convertor._get_custom_schema' is abstract (raises 'NotImplementedError'),
    so it's patched here to return the unified schema shape directly.
    """
    with patch.object(
        Convertor, "_get_custom_schema", return_value={"fields": {column: "binary"}}
    ):
        convertor = Convertor(
            original_schema={column: "binary"}, df=df
        )
    return convertor


def _flatten_avro_schema(schema: dict) -> dict:
    """
    Flatten an Avro schema to a simple dict mapping column names to types.
    This is a minimal implementation for testing purposes.
    """
    return {
        field["name"]: field["type"]
        for field in schema.get("fields", [])
    }


def test_initiate_csv_convertor(rp_logger):
    rp_logger.info("Initiating the instance of the class CSVConvertor")
    df = pd.read_csv(
        f"{DIR_NAME}/unit/convertors/fixtures/csv_tables/table_with_diff_data_types.csv"
    )
    convertor = CSVConvertor(df)
    assert convertor.custom_schema == {"fields": {}, "format": "CSV"}
    assert convertor.preprocessed_df.shape == df.shape
    assert convertor.preprocessed_df.dtypes.to_dict() == {
        "employeekey": dtype("int64"),
        "parentemployeekey": dtype("float64"),
        "employeenationalidalternatekey": pd.StringDtype(),
        "parentemployeenationalidalternatekey": dtype("float64"),
        "salesterritorykey": dtype("int64"),
        "firstname": pd.StringDtype(),
        "lastname": pd.StringDtype(),
        "middlename": pd.StringDtype(),
        "namestyle": dtype("bool"),
        "title": pd.StringDtype(),
        "hiredate": pd.StringDtype(),
        "birthdate": pd.StringDtype(),
        "loginid": pd.StringDtype(),
        "emailaddress": pd.StringDtype(),
        "phone": pd.StringDtype(),
        "maritalstatus": pd.StringDtype(),
        "emergencycontactname": pd.StringDtype(),
        "emergencycontactphone": pd.StringDtype(),
        "salariedflag": dtype("bool"),
        "gender": pd.StringDtype(),
        "payfrequency": dtype("int64"),
        "baserate": dtype("float64"),
        "vacationhours": dtype("int64"),
        "sickleavehours": dtype("int64"),
        "currentflag": dtype("bool"),
        "salespersonflag": dtype("bool"),
        "departmentname": pd.StringDtype(),
        "startdate": pd.StringDtype(),
        "enddate": pd.StringDtype(),
        "status": pd.StringDtype(),
        "employeephoto": pd.StringDtype()
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initiate_avro_convertor(rp_logger):
    rp_logger.info("Initiating the instance of the class AvroConvertor")
    path = f"{DIR_NAME}/unit/convertors/fixtures/avro_tables/table_with_diff_data_types.avro"
    df = pdx.from_avro(path)

    with open(path, "rb") as f:
        original_schema = fastavro.reader(f).writer_schema

    convertor = AvroConvertor(_flatten_avro_schema(original_schema), df)

    assert df.dtypes.to_dict() == {
        "employeekey": dtype("int64"),
        "parentemployeekey": dtype("float64"),
        "employeenationalidalternatekey": dtype("O"),
        "salesterritorykey": dtype("int64"),
        "firstname": dtype("O"),
        "lastname": dtype("O"),
        "middlename": dtype("O"),
        "namestyle": dtype("O"),
        "title": dtype("O"),
        "hiredate": dtype("O"),
        "birthdate": dtype("O"),
        "loginid": dtype("O"),
        "emailaddress": dtype("O"),
        "phone": dtype("O"),
        "maritalstatus": dtype("O"),
        "emergencycontactname": dtype("O"),
        "emergencycontactphone": dtype("O"),
        "salariedflag": dtype("bool"),
        "gender": dtype("O"),
        "payfrequency": dtype("int64"),
        "baserate": dtype("float64"),
        "vacationhours": dtype("int64"),
        "sickleavehours": dtype("int64"),
        "currentflag": dtype("bool"),
        "salespersonflag": dtype("bool"),
        "departmentname": dtype("O"),
        "startdate": dtype("O"),
        "enddate": dtype("O"),
        "status": dtype("O"),
        "employeephoto": dtype("O"),
    }

    assert convertor.custom_schema == {
        "fields": {
            "employeekey": "int",
            "parentemployeekey": "float",
            "employeenationalidalternatekey": "string",
            "salesterritorykey": "int",
            "firstname": "string",
            "lastname": "string",
            "middlename": "string",
            "namestyle": "boolean",
            "title": "string",
            "hiredate": "string",
            "birthdate": "string",
            "loginid": "string",
            "emailaddress": "string",
            "phone": "string",
            "maritalstatus": "string",
            "emergencycontactname": "string",
            "emergencycontactphone": "string",
            "salariedflag": "boolean",
            "gender": "string",
            "payfrequency": "int",
            "baserate": "float",
            "vacationhours": "int",
            "sickleavehours": "int",
            "currentflag": "boolean",
            "salespersonflag": "boolean",
            "departmentname": "string",
            "startdate": "string",
            "enddate": "string",
            "status": "string",
            "employeephoto": "string"
        },
        "format": "Avro",
        "date_types_to_restore": {}
    }
    assert convertor.preprocessed_df.dtypes.to_dict() == {
        "employeekey": dtype("int64"),
        "parentemployeekey": dtype("float64"),
        "employeenationalidalternatekey": pd.StringDtype(),
        "salesterritorykey": dtype("int64"),
        "firstname": pd.StringDtype(),
        "lastname": pd.StringDtype(),
        "middlename": pd.StringDtype(),
        "namestyle": pd.BooleanDtype(),
        "title": pd.StringDtype(),
        "hiredate": pd.StringDtype(),
        "birthdate": pd.StringDtype(),
        "loginid": pd.StringDtype(),
        "emailaddress": pd.StringDtype(),
        "phone": pd.StringDtype(),
        "maritalstatus": pd.StringDtype(),
        "emergencycontactname": pd.StringDtype(),
        "emergencycontactphone": pd.StringDtype(),
        "salariedflag": dtype("bool"),
        "gender": pd.StringDtype(),
        "payfrequency": dtype("int64"),
        "baserate": dtype("float64"),
        "vacationhours": dtype("int64"),
        "sickleavehours": dtype("int64"),
        "currentflag": dtype("bool"),
        "salespersonflag": dtype("bool"),
        "departmentname": pd.StringDtype(),
        "startdate": pd.StringDtype(),
        "enddate": pd.StringDtype(),
        "status": pd.StringDtype(),
        "employeephoto": pd.StringDtype(),
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initiate_avro_convertor_without_provided_schema(rp_logger):
    rp_logger.info("Initiating the instance of the class AvroConvertor without a provided schema")
    df = pdx.from_avro(
        f"{DIR_NAME}/unit/convertors/fixtures/avro_tables/table_with_diff_data_types.avro"
    )

    convertor = AvroConvertor(original_schema=None, df=df)

    assert df.dtypes.to_dict() == {
        "employeekey": dtype("int64"),
        "parentemployeekey": dtype("float64"),
        "employeenationalidalternatekey": dtype("O"),
        "salesterritorykey": dtype("int64"),
        "firstname": dtype("O"),
        "lastname": dtype("O"),
        "middlename": dtype("O"),
        "namestyle": dtype("O"),
        "title": dtype("O"),
        "hiredate": dtype("O"),
        "birthdate": dtype("O"),
        "loginid": dtype("O"),
        "emailaddress": dtype("O"),
        "phone": dtype("O"),
        "maritalstatus": dtype("O"),
        "emergencycontactname": dtype("O"),
        "emergencycontactphone": dtype("O"),
        "salariedflag": dtype("bool"),
        "gender": dtype("O"),
        "payfrequency": dtype("int64"),
        "baserate": dtype("float64"),
        "vacationhours": dtype("int64"),
        "sickleavehours": dtype("int64"),
        "currentflag": dtype("bool"),
        "salespersonflag": dtype("bool"),
        "departmentname": dtype("O"),
        "startdate": dtype("O"),
        "enddate": dtype("O"),
        "status": dtype("O"),
        "employeephoto": dtype("O"),
    }

    assert convertor.custom_schema == {
        "fields": {},
        "format": "Avro",
        "date_types_to_restore": {}
    }
    assert convertor.preprocessed_df.dtypes.to_dict() == {
        "employeekey": dtype("int64"),
        "parentemployeekey": dtype("float64"),
        "employeenationalidalternatekey": dtype("O"),
        "salesterritorykey": dtype("int64"),
        "firstname": dtype("O"),
        "lastname": dtype("O"),
        "middlename": dtype("O"),
        "namestyle": dtype("O"),
        "title": dtype("O"),
        "hiredate": dtype("O"),
        "birthdate": dtype("O"),
        "loginid": dtype("O"),
        "emailaddress": dtype("O"),
        "phone": dtype("O"),
        "maritalstatus": dtype("O"),
        "emergencycontactname": dtype("O"),
        "emergencycontactphone": dtype("O"),
        "salariedflag": dtype("bool"),
        "gender": dtype("O"),
        "payfrequency": dtype("int64"),
        "baserate": dtype("float64"),
        "vacationhours": dtype("int64"),
        "sickleavehours": dtype("int64"),
        "currentflag": dtype("bool"),
        "salespersonflag": dtype("bool"),
        "departmentname": dtype("O"),
        "startdate": dtype("O"),
        "enddate": dtype("O"),
        "status": dtype("O"),
        "employeephoto": dtype("O"),
    }
    pd.testing.assert_series_equal(convertor.preprocessed_df.dtypes, df.dtypes)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initiate_avro_convertor_with_schema_containing_logical_date_types(rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor"
        "with the schema containing logical date/time types"
    )
    path = (
        f"{DIR_NAME}/unit/convertors/fixtures/avro_tables/table_with_date_logical_types.avro"
    )
    df = pdx.from_avro(path)

    with open(path, "rb") as f:
        original_schema = fastavro.reader(f).writer_schema

    convertor = AvroConvertor(_flatten_avro_schema(original_schema), df)

    assert df.dtypes.to_dict() == {
        "date_col": dtype("O"),
        "date_col_nullable": dtype("O"),
        "date_string_col": dtype("O"),
        "local_timestamp_micros_col": dtype("<M8[ns]"),
        "local_timestamp_micros_col_nullable": dtype("<M8[ns]"),
        "local_timestamp_millis_col": dtype("<M8[ns]"),
        "local_timestamp_millis_col_nullable": dtype("<M8[ns]"),
        "time_micros_col": dtype("O"),
        "time_micros_col_nullable": dtype("O"),
        "time_millis_col": dtype("O"),
        "time_millis_col_nullable": dtype("O"),
        "timestamp_micros_col": pd.DatetimeTZDtype(tz="UTC"),
        "timestamp_micros_col_nullable": pd.DatetimeTZDtype(tz="UTC"),
        "timestamp_millis_col": pd.DatetimeTZDtype(tz="UTC"),
        "timestamp_millis_col_nullable": pd.DatetimeTZDtype(tz="UTC"),
    }

    assert convertor.custom_schema == {
        "date_types_to_restore": {
            "date_col": "date",
            "date_col_nullable": "date",
            "local_timestamp_micros_col": "datetime",
            "local_timestamp_micros_col_nullable": "datetime",
            "local_timestamp_millis_col": "datetime",
            "local_timestamp_millis_col_nullable": "datetime",
            "time_micros_col": "time",
            "time_micros_col_nullable": "time",
            "time_millis_col": "time",
            "time_millis_col_nullable": "time",
            "timestamp_micros_col": "datetime",
            "timestamp_micros_col_nullable": "datetime",
            "timestamp_millis_col": "datetime",
            "timestamp_millis_col_nullable": "datetime",
        },
        "fields": {
            "date_col": "date",
            "date_col_nullable": "date",
            "date_string_col": "string",
            "local_timestamp_micros_col": "date",
            "local_timestamp_micros_col_nullable": "date",
            "local_timestamp_millis_col": "date",
            "local_timestamp_millis_col_nullable": "date",
            "time_micros_col": "date",
            "time_micros_col_nullable": "date",
            "time_millis_col": "date",
            "time_millis_col_nullable": "date",
            "timestamp_micros_col": "date",
            "timestamp_micros_col_nullable": "date",
            "timestamp_millis_col": "date",
            "timestamp_millis_col_nullable": "date",
        },
        "format": "Avro",
    }
    assert convertor.preprocessed_df.dtypes.to_dict() == {
        "date_col": dtype("O"),
        "date_col_nullable": dtype("O"),
        "date_string_col": pd.StringDtype(),
        "local_timestamp_micros_col": dtype("<M8[ns]"),
        "local_timestamp_micros_col_nullable": dtype("<M8[ns]"),
        "local_timestamp_millis_col": dtype("<M8[ns]"),
        "local_timestamp_millis_col_nullable": dtype("<M8[ns]"),
        "time_micros_col": dtype("O"),
        "time_micros_col_nullable": dtype("O"),
        "time_millis_col": dtype("O"),
        "time_millis_col_nullable": dtype("O"),
        "timestamp_micros_col": pd.DatetimeTZDtype(tz="UTC"),
        "timestamp_micros_col_nullable": pd.DatetimeTZDtype(tz="UTC"),
        "timestamp_millis_col": pd.DatetimeTZDtype(tz="UTC"),
        "timestamp_millis_col_nullable": pd.DatetimeTZDtype(tz="UTC"),
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_preprocess_df_if_column_is_int(rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor with the schema containing "
        "only one column with data type - 'int'"
    )
    df = pd.DataFrame({"Test": [1, 2, 3]})

    convertor = AvroConvertor({"Test": ["int"]}, df)
    assert convertor.preprocessed_df.dtypes.to_dict() == {"Test": "int64"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_preprocess_df_if_column_is_int_in_column_with_null_values(rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor with the schema "
        "containing one nullable column with data type - 'int'"
    )
    df = pd.DataFrame({"Test": [1, 2, np.NaN]})

    convertor = AvroConvertor({"Test": ["int"]}, df)
    assert convertor.preprocessed_df.dtypes.to_dict() == {"Test": "float64"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_preprocess_df_if_column_is_float(rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor "
        "with the schema containing one column with data type - 'float'"
    )
    df = pd.DataFrame({"Test": [1.0, 2.0, 3.0]})

    convertor = AvroConvertor({"Test": ["float"]}, df)
    assert convertor.preprocessed_df.dtypes.to_dict() == {"Test": "float64"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_preprocess_df_if_column_is_double(rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor "
        "with the schema containing one column with data type - 'double'"
    )
    df = pd.DataFrame({"Test": [1.055, 2.034, 3.099]})

    convertor = AvroConvertor({"Test": ["double"]}, df)
    assert convertor.preprocessed_df.dtypes.to_dict() == {"Test": "float64"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_preprocess_df_if_column_is_string(rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor "
        "with the schema containing  one column with data type - 'string'"
    )
    df = pd.DataFrame({"Test": ["1", "2", "3"]})

    convertor = AvroConvertor({"Test": ["string"]}, df)
    assert convertor.preprocessed_df.dtypes.to_dict() == {"Test": "string[python]"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("dataframe, expected_dtype", [
    (pd.DataFrame({"Test": [True, False, True]}), "bool"),
    (
        pd.DataFrame({"Test": [True, False, True, np.NaN, None]}),
        "boolean"
    )
])
def test_preprocess_df_if_column_is_bool(dataframe, expected_dtype, rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor "
        "with the schema containing one column with data type - 'boolean'"
    )
    convertor = AvroConvertor({"Test": ["boolean"]}, dataframe)
    assert convertor.preprocessed_df.dtypes.to_dict() == {"Test": expected_dtype}
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_preprocess_df_if_column_is_datetime(rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor "
        "with the schema containing one column with data type - 'datetime'"
    )
    df = pd.DataFrame({"Test": pd.date_range("20130101", periods=3)})

    convertor = AvroConvertor({"Test": ["string"]}, df)
    assert convertor.preprocessed_df.dtypes.to_dict() == {"Test": "string[python]"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("avro_type, column, expected_dtype_to_restore", [
    # Avro 'date' logical type -> pandavro yields object dtype of datetime.date
    (
        {"type": "int", "logicalType": "date"},
        pd.Series([date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)], dtype="object"),
        "date"
    ),
    # nullable 'date' (union with null)
    (
        ["null", {"type": "int", "logicalType": "date"}],
        pd.Series([date(2023, 1, 1), None, date(2023, 1, 3)], dtype="object"),
        "date"
    ),
    # Avro 'timestamp-micros' -> pandavro yields tz-aware datetime64[ns, UTC]
    (
        {"type": "long", "logicalType": "timestamp-micros"},
        pd.to_datetime(
            pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
        ).dt.tz_localize("UTC"),
        "datetime"
    ),
    # nullable 'timestamp-micros' (union with null) -> tz-aware datetime64[ns, UTC]
    (
        ["null", {"type": "long", "logicalType": "timestamp-micros"}],
        pd.to_datetime(
            pd.Series(["2023-01-01", None, "2023-01-03"])
        ).dt.tz_localize("UTC"),
        "datetime"
    ),
    # Avro 'timestamp-millis' (tz-naive datetime64)
    (
        {"type": "long", "logicalType": "timestamp-millis"},
        pd.to_datetime(pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])),
        "datetime"
    ),
    # nullable 'timestamp-millis' (union with null)
    (
        ["null", {"type": "long", "logicalType": "timestamp-millis"}],
        pd.to_datetime(pd.Series(["2023-01-01", None, "2023-01-03"])),
        "datetime"
    ),
    # Avro 'time-millis' logical type -> pandavro yields object dtype of datetime.time
    (
        {"type": "int", "logicalType": "time-millis"},
        pd.Series([time(10, 0, 0), time(14, 30, 0), time(23, 59, 59)], dtype="object"),
        "time"
    ),
    # nullable 'time-millis' (union with null)
    (
        ["null", {"type": "int", "logicalType": "time-millis"}],
        pd.Series([time(10, 0, 0), None, time(23, 59, 59)], dtype="object"),
        "time"
    ),
    # Avro 'time-micros' logical type -> pandavro yields object dtype of datetime.time
    (
        {"type": "long", "logicalType": "time-micros"},
        pd.Series([time(1, 0, 0), time(2, 30, 0), time(23, 59, 59)], dtype="object"),
        "time"
    ),
    # nullable 'time-micros' (union with null)
    (
        ["null", {"type": "long", "logicalType": "time-micros"}],
        pd.Series([time(1, 0, 0), None, time(23, 59, 59)], dtype="object"),
        "time"
    ),
    # Avro 'local-timestamp-millis' -> tz-naive datetime64 (no UTC localization)
    (
        {"type": "long", "logicalType": "local-timestamp-millis"},
        pd.to_datetime(pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])),
        "datetime"
    ),
    # nullable 'local-timestamp-millis' (union with null)
    (
        ["null", {"type": "long", "logicalType": "local-timestamp-millis"}],
        pd.to_datetime(pd.Series(["2023-01-01", None, "2023-01-03"])),
        "datetime"
    ),
    # Avro 'local-timestamp-micros' -> tz-naive datetime64 (no UTC localization)
    (
        {"type": "long", "logicalType": "local-timestamp-micros"},
        pd.to_datetime(pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])),
        "datetime"
    ),
    # nullable 'local-timestamp-micros' (union with null)
    (
        ["null", {"type": "long", "logicalType": "local-timestamp-micros"}],
        pd.to_datetime(pd.Series(["2023-01-01", None, "2023-01-03"])),
        "datetime"
    )
])
def test_preprocess_df_maps_avro_logical_date_types_to_date(
    avro_type, column, expected_dtype_to_restore, rp_logger
):
    """EPMCTDM-7581 (Avro counterpart): Avro date/time/timestamp logical types must
    be mapped to the unified 'date' type (like Parquet/Delta via
    PyArrowSchemaConvertor) instead of 'int'. Previously they were mapped to 'int',
    and 'Convertor._update_data_types' crashed with a TypeError when calling
    '.astype' on the loaded datetime.date / datetime64 column."""
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor with the schema "
        "containing one column with an Avro date/time/timestamp logical type"
    )
    df = pd.DataFrame({"Test": column})
    expected_dtype = df["Test"].dtype

    convertor = AvroConvertor({"Test": avro_type}, df)

    assert convertor.custom_schema["fields"] == {"Test": "date"}
    assert convertor.custom_schema["date_types_to_restore"] == {
        "Test": expected_dtype_to_restore
    }
    assert convertor.preprocessed_df["Test"].dtype == expected_dtype
    assert not pd.api.types.is_numeric_dtype(convertor.preprocessed_df["Test"])
    assert convertor.preprocessed_df["Test"].dtype == expected_dtype
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_preprocess_df_if_column_is_binary(caplog, rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor "
        "with the schema containing one column with data type - 'bytes'"
    )
    df = pd.DataFrame({"Test": [b"test", b"test2", b"test3"]})
    error_message = (
        "It seems that the column - 'Test' has unsupported data type - '['bytes']'"
    )

    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            AvroConvertor({"Test": ["bytes"]}, df)

    assert str(error.value) == error_message
    assert error_message in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "avro_type",
    [
        {"type": "array", "items": "string"},
        {"type": "map", "values": "string"},
        {
            "type": "record",
            "name": "SubRecord",
            "fields": [{"name": "field_1", "type": "string"}],
        },
        {"type": "enum", "name": "Suit", "symbols": ["SPADES", "HEARTS"]},
        {"type": "fixed", "name": "MD5", "size": 16},
    ],
    ids=["array", "map", "record", "enum", "fixed"],
)
def test_initiate_avro_convertor_if_schema_contains_complex_data_type(
    avro_type, caplog, rp_logger
):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor with the schema "
        f"containing the unsupported complex Avro data type - '{avro_type['type']}'"
    )
    df = pd.DataFrame({"Test": [None, None, None]})
    error_message = (
        f"It seems that the column - 'Test' has unsupported data type - '{avro_type}'"
    )

    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            AvroConvertor(original_schema={"Test": avro_type}, df=df)

    assert str(error.value) == error_message
    assert error_message in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_preprocess_df_if_column_is_null(rp_logger):
    rp_logger.info(
        "Preprocessing the dataframe contained the column "
        "with data type - 'null' by the class AvroConvertor"
    )
    df = pd.DataFrame({"Test": [np.NaN, np.NaN, np.NaN]})

    convertor = AvroConvertor({"Test": ["null"]}, df)
    assert convertor.preprocessed_df.dtypes.to_dict() == {
        "Test": dtype("float64")
    }
    assert convertor.custom_schema == {
        "fields": {"Test": "null"},
        "format": "Avro",
        "date_types_to_restore": {}
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_preprocess_df_if_column_is_invalid_null(rp_logger, caplog):
    rp_logger.info(
        "Preprocessing the dataframe contained the column "
        "with data type - 'null' by the class AvroConvertor "
        "in case this column isn't empty"
    )
    df = pd.DataFrame({"Test": [np.NaN, "value", np.NaN]})
    error_message = ("It seems that the data type - 'null' isn't correct "
                     "for the column - 'Test' as it's not empty")
    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            AvroConvertor(original_schema={"Test": ["null"]}, df=df)
            assert str(error.value) == error_message
            assert caplog.text == error_message

    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initiate_avro_convertor_if_schema_contains_unsupported_data_type(caplog, rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor "
        "with the schema containing unsupported data type"
    )
    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            df = pd.DataFrame({"Test": [1, 2, 3]})

            schema = {"Test": "test"}

            AvroConvertor(original_schema=schema, df=df)

            assert (
                str(error.value)
                == "It seems that the column - 'Test' has unsupported data type - 'test'"
            )
            assert (
                "It seems that the column - 'Test' has unsupported data type - 'test'"
                in caplog.text
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_ascii_text_decodes_to_string(rp_logger):
    rp_logger.info(
        "Casting a binary column with ASCII plain-text values in the base Convertor "
        "should decode bytes to strings and store the encoding"
    )
    df = pd.DataFrame({"Blob": [b"hello", b"world"]})
    convertor = _make_binary_convertor(df, "Blob")
    assert convertor.preprocessed_df["Blob"].tolist() == ["hello", "world"]
    assert convertor.custom_schema["encoding"]["Blob"] == "ascii"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_utf8_text_decodes_to_string(rp_logger):
    rp_logger.info(
        "Casting a binary column with UTF-8 plain-text values "
        "in the base Convertor should decode bytes to strings and store "
        "the encoding"
    )
    value = "Héllo Wörld für dich testing more text here for detection purposes"
    encoded_value = value.encode("utf-8")
    df = pd.DataFrame({"Blob": [encoded_value, encoded_value]})
    convertor = _make_binary_convertor(df, "Blob")
    assert convertor.preprocessed_df["Blob"].tolist() == [value, value]
    assert convertor.custom_schema["encoding"]["Blob"] == "utf-8"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_utf8_sig_text_decodes_to_string(rp_logger):
    rp_logger.info(
        "Casting a binary column with UTF-8-SIG (BOM-prefixed) plain-text values "
        "in the base Convertor should decode bytes to strings and store the encoding"
    )
    value = "Héllo Wörld für dich testing detection purposes here"
    encoded_value = value.encode("utf-8-sig")
    df = pd.DataFrame({"Blob": [encoded_value, encoded_value]})
    convertor = _make_binary_convertor(df, "Blob")
    assert convertor.preprocessed_df["Blob"].tolist() == [value, value]
    assert convertor.custom_schema["encoding"]["Blob"] == "UTF-8-SIG"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_utf16_text_decodes_to_string(rp_logger):
    rp_logger.info(
        "Casting a binary column with UTF-16 plain-text values in the base Convertor "
        "should decode bytes to strings and store the encoding"
    )
    value = "Hello World testing detection purposes here now"
    encoded_value = value.encode("utf-16")
    df = pd.DataFrame({"Blob": [encoded_value, encoded_value]})
    convertor = _make_binary_convertor(df, "Blob")
    assert convertor.preprocessed_df["Blob"].tolist() == [value, value]
    assert convertor.custom_schema["encoding"]["Blob"] == "UTF-16"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_windows1252_text_decodes_to_string(rp_logger):
    rp_logger.info(
        "Casting a binary column with Windows-1252 plain-text values in the base "
        "Convertor should decode bytes to strings and store the encoding"
    )
    value = "café and naïve more text here for detection purposes indeed"
    encoded_value = value.encode("windows-1252")
    df = pd.DataFrame({"Blob": [encoded_value, encoded_value]})
    convertor = _make_binary_convertor(df, "Blob")
    assert convertor.preprocessed_df["Blob"].tolist() == [value, value]
    assert convertor.custom_schema["encoding"]["Blob"] == "Windows-1252"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_mixed_encodings_skips_decoding(caplog, rp_logger):
    rp_logger.info(
        "Casting a binary column with mixed-encoding plain-text bytes in the base Convertor "
        "should log a warning and replace all values with null"
    )
    utf8_value = "Héllo Wörld für dich testing".encode("utf-8")
    windows1252_value = b"caf\xe9 and na\xefve more text here for detection"
    df = pd.DataFrame({"Text": [utf8_value, windows1252_value]})
    with caplog.at_level("WARNING"):
        convertor = _make_binary_convertor(df, "Text")
    assert (
        "The binary column 'Text' contains plain-text values with "
        "multiple different character encodings: 'Windows-1252', 'utf-8'. "
        "Decoding will be skipped; binary values will be replaced by null values."
    ) in caplog.text
    assert convertor.preprocessed_df["Text"].isna().all()
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_application_bytes_skips_decoding(caplog, rp_logger):
    rp_logger.info(
        "Casting a binary column with application/octet-stream bytes in the base Convertor "
        "should log a warning and replace all values with null"
    )
    binary_payload = b"\x00\x00\x00\x00\x00\x00\x00\x00"
    df = pd.DataFrame({"Data": [binary_payload, binary_payload]})
    with caplog.at_level("WARNING"):
        convertor = _make_binary_convertor(df, "Data")
    assert (
        "The binary column 'Data' contains values with non-decodable "
        "MIME type(s): 'application/octet-stream'. "
        "Decoding will be skipped; binary values will be replaced by null values."
    ) in caplog.text
    assert convertor.preprocessed_df["Data"].isna().all()
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_html_bytes_skips_decoding(caplog, rp_logger):
    rp_logger.info(
        "Casting a binary column with HTML bytes in the base Convertor "
        "should log a warning and replace all values with null"
    )
    html_bytes = (
        b'<!DOCTYPE html><html lang="en"><head>'
        b'<meta charset="UTF-8"><title>Test</title></head>'
        b"<body><p>Hello World</p></body></html>"
    )
    df = pd.DataFrame({"Content": [html_bytes, html_bytes]})
    with caplog.at_level("WARNING"):
        convertor = _make_binary_convertor(df, "Content")
    assert (
        "The binary column 'Content' contains values with non-decodable "
        "MIME type(s): 'text/html'. "
        "Decoding will be skipped; binary values will be replaced by null values."
    ) in caplog.text
    assert convertor.preprocessed_df["Content"].isna().all()
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_xml_bytes_skips_decoding(caplog, rp_logger):
    rp_logger.info(
        "Casting a binary column with XML bytes in the base Convertor "
        "should log a warning and replace all values with null"
    )
    xml_bytes = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b"<root><item>value</item></root>"
    )
    df = pd.DataFrame({"Content": [xml_bytes, xml_bytes]})
    with caplog.at_level("WARNING"):
        convertor = _make_binary_convertor(df, "Content")
    assert (
        "The binary column 'Content' contains values with non-decodable "
        "MIME type(s): 'text/xml'. "
        "Decoding will be skipped; binary values will be replaced by null values."
    ) in caplog.text
    assert convertor.preprocessed_df["Content"].isna().all()
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_python_source_bytes_skips_decoding(caplog, rp_logger):
    rp_logger.info(
        "Casting a binary column with Python source bytes in the base Convertor "
        "should log a warning and replace all values with null"
    )
    python_bytes = (
        b"# -*- coding: utf-8 -*-\n"
        b"import os\n\n"
        b"def foo():\n"
        b'    return os.path.join("a", "b")\n'
    )
    df = pd.DataFrame({"Content": [python_bytes, python_bytes]})
    with caplog.at_level("WARNING"):
        convertor = _make_binary_convertor(df, "Content")
    assert (
        "The binary column 'Content' contains values with non-decodable "
        "MIME type(s): 'text/x-python'. "
        "Decoding will be skipped; binary values will be replaced by null values."
    ) in caplog.text
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_image_bytes_skips_decoding(caplog, rp_logger):
    rp_logger.info(
        "Casting a binary column with JPEG image bytes in the base Convertor "
        "should log a warning and replace all values with null"
    )
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x00\x00\x00\x01\x00\x01\x00\x00"
    df = pd.DataFrame({"Photo": [jpeg_bytes, jpeg_bytes]})
    with caplog.at_level("WARNING"):
        convertor = _make_binary_convertor(df, "Photo")
    assert (
        "The binary column 'Photo' contains values with non-decodable "
        "MIME type(s): 'image/jpeg'. "
        "Decoding will be skipped; binary values will be replaced by null values."
    ) in caplog.text
    assert convertor.preprocessed_df["Photo"].isna().all()
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_audio_bytes_skips_decoding(caplog, rp_logger):
    rp_logger.info(
        "Casting a binary column with MP3 audio bytes in the base Convertor "
        "should log a warning and replace all values with null"
    )
    mp3_bytes = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 20
    df = pd.DataFrame({"Audio": [mp3_bytes, mp3_bytes]})
    with caplog.at_level("WARNING"):
        convertor = _make_binary_convertor(df, "Audio")
    assert (
        "The binary column 'Audio' contains values with non-decodable "
        "MIME type(s): 'audio/mpeg'. "
        "Decoding will be skipped; binary values will be replaced by null values."
    ) in caplog.text
    assert convertor.preprocessed_df["Audio"].isna().all()
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_video_bytes_skips_decoding(caplog, rp_logger):
    rp_logger.info(
        "Casting a binary column with MP4 video bytes in the base Convertor "
        "should log a warning and replace all values with null"
    )
    mp4_bytes = (
        b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 20
    )
    df = pd.DataFrame({"Video": [mp4_bytes, mp4_bytes]})
    with caplog.at_level("WARNING"):
        convertor = _make_binary_convertor(df, "Video")
    assert (
        "The binary column 'Video' contains values with non-decodable "
        "MIME type(s): 'video/mp4'. "
        "Decoding will be skipped; binary values will be replaced by null values."
    ) in caplog.text
    assert convertor.preprocessed_df["Video"].isna().all()
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_font_bytes_skips_decoding(caplog, rp_logger):
    rp_logger.info(
        "Casting a binary column with WOFF font bytes in the base Convertor "
        "should log a warning and replace all values with null"
    )
    woff_bytes = b"wOFF\x00\x01\x00\x00" + b"\x00" * 20
    df = pd.DataFrame({"Font": [woff_bytes, woff_bytes]})
    with caplog.at_level("WARNING"):
        convertor = _make_binary_convertor(df, "Font")
    assert (
        "The binary column 'Font' contains values with non-decodable "
        "MIME type(s): 'font/woff'. "
        "Decoding will be skipped; binary values will be replaced by null values."
    ) in caplog.text
    assert convertor.preprocessed_df["Font"].isna().all()
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_multiple_non_decodable_mime_types_skips_decoding(
    caplog, rp_logger
):
    rp_logger.info(
        "Casting a binary column containing values with several different "
        "non-decodable MIME types in the base Convertor should log a warning "
        "listing all of them and replace all values with null"
    )
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x00\x00\x00\x01\x00\x01\x00\x00"
    mp3_bytes = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 20
    df = pd.DataFrame({"Media": [jpeg_bytes, mp3_bytes]})
    with caplog.at_level("WARNING"):
        convertor = _make_binary_convertor(df, "Media")
    assert (
        "The binary column 'Media' contains values with non-decodable "
        "MIME type(s): 'audio/mpeg', 'image/jpeg'. "
        "Decoding will be skipped; binary values will be replaced by null values."
    ) in caplog.text
    assert convertor.preprocessed_df["Media"].isna().all()
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_non_decodable_and_plain_text_mix_skips_decoding(
    caplog, rp_logger
):
    rp_logger.info(
        "Casting a binary column mixing a non-decodable MIME type value with a "
        "plain-text value in the base Convertor should log a warning naming only "
        "the non-decodable MIME type and replace all values with null"
    )
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x00\x00\x00\x01\x00\x01\x00\x00"
    text_bytes = b"hello"
    df = pd.DataFrame({"Mixed": [jpeg_bytes, text_bytes]})
    with caplog.at_level("WARNING"):
        convertor = _make_binary_convertor(df, "Mixed")
    assert (
        "The binary column 'Mixed' contains values with non-decodable "
        "MIME type(s): 'image/jpeg'. "
        "Decoding will be skipped; binary values will be replaced by null values."
    ) in caplog.text
    assert convertor.preprocessed_df["Mixed"].isna().all()
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_all_null_values_skips_decoding(rp_logger):
    rp_logger.info(
        "Casting a binary column containing only null values in the base Convertor "
        "should leave the column as-is without raising an error"
    )
    df = pd.DataFrame({"Data": [None, np.nan]})
    convertor = _make_binary_convertor(df, "Data")
    assert convertor.preprocessed_df["Data"].isna().all()
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_some_null_values_decodes_non_null_values(rp_logger):
    rp_logger.info(
        "Casting a binary column with a decodable encoding that also contains "
        "null values in the base Convertor should decode the non-null values "
        "and leave the null values untouched, without raising an error"
    )
    df = pd.DataFrame({"Blob": [b"hello", None, np.nan]})
    convertor = _make_binary_convertor(df, "Blob")
    result = convertor.preprocessed_df["Blob"].tolist()
    assert result[0] == "hello"
    assert pd.isna(result[1])
    assert pd.isna(result[2])
    assert convertor.custom_schema["encoding"]["Blob"] == "ascii"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_decode_error_replaces_column_with_null(rp_logger):
    rp_logger.info(
        "Casting a binary column whose validated encoding still fails to decode "
        "in the base Convertor should replace all values in the column with null "
        "instead of raising an error"
    )
    df = pd.DataFrame({"Blob": [b"hello", b"world"]})
    with patch.object(
        Convertor,
        "_decode_binary_column",
        side_effect=UnicodeDecodeError("ascii", b"\xff", 0, 1, "invalid byte"),
    ):
        convertor = _make_binary_convertor(df, "Blob")
    assert convertor.preprocessed_df["Blob"].isna().all()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_encoding_detection_stops_after_sample_size(rp_logger):
    rp_logger.info(
        "Casting a binary column with more non-null values than the encoding "
        "detection sample size in the base Convertor should run 'chardet.detect' "
        "only on the sampled values instead of scanning the whole column"
    )
    row_count = Convertor._ENCODING_DETECTION_SAMPLE_SIZE + 50
    df = pd.DataFrame({"Blob": [b"hello"] * row_count})
    with patch(
        "syngen.ml.convertor.convertor.chardet.detect", wraps=__import__("chardet").detect
    ) as mocked_detect:
        convertor = _make_binary_convertor(df, "Blob")
    assert mocked_detect.call_count == Convertor._ENCODING_DETECTION_SAMPLE_SIZE
    assert convertor.custom_schema["encoding"]["Blob"] == "ascii"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_postprocess_encoded_columns_with_unknown_codec_raises_lookup_error(caplog, rp_logger):
    rp_logger.info(
        "Postprocessing an encoded column with a codec name unknown to Python's "
        "codec registry in the base Convertor should raise the 'LookupError'"
    )
    convertor = Convertor.__new__(Convertor)
    convertor.preprocessed_df = pd.DataFrame({"Blob": ["hello"]})
    convertor.custom_schema = {"encoding": {"Blob": "not-a-real-codec"}}
    with caplog.at_level("ERROR"):
        with pytest.raises(LookupError) as error:
            convertor._postprocess_encoded_columns()
    assert "unknown encoding: not-a-real-codec" in str(error.value)
    assert (
        "Failed to encode a value in the column 'Blob' "
        "using the encoding 'not-a-real-codec'."
    ) in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_postprocess_encoded_columns_with_non_encodable_char_raises_unicode_encode_error(
    caplog, rp_logger
):
    rp_logger.info(
        "Postprocessing an encoded column with a value containing a character "
        "that can't be represented in the recorded encoding in the base Convertor "
        "should raise the 'UnicodeEncodeError'"
    )
    convertor = Convertor.__new__(Convertor)
    convertor.preprocessed_df = pd.DataFrame({"Blob": ["héllo"]})
    convertor.custom_schema = {"encoding": {"Blob": "ascii"}}
    with caplog.at_level("ERROR"):
        with pytest.raises(UnicodeEncodeError) as error:
            convertor._postprocess_encoded_columns()
    assert "'ascii' codec can't encode character" in str(error.value)
    assert (
        "Failed to encode a value in the column 'Blob' "
        "using the encoding 'ascii'."
    ) in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_postprocess_encoded_columns_with_non_string_encoding_raises_type_error(
    caplog, rp_logger
):
    rp_logger.info(
        "Postprocessing an encoded column with a recorded encoding that isn't a "
        "string in the base Convertor should raise the 'TypeError'"
    )
    convertor = Convertor.__new__(Convertor)
    convertor.preprocessed_df = pd.DataFrame({"Blob": ["hello"]})
    convertor.custom_schema = {"encoding": {"Blob": 123}}
    with caplog.at_level("ERROR"):
        with pytest.raises(TypeError) as error:
            convertor._postprocess_encoded_columns()
    assert "encode() argument 'encoding' must be str, not int" in str(error.value)
    assert (
        "Failed to encode a value in the column 'Blob' "
        "using the encoding '123'."
    ) in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)
