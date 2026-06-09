import pytest
from datetime import UTC, date, time

import numpy as np
from numpy import dtype
import pandas as pd

from syngen.ml.convertor import CSVConvertor, AvroConvertor
from syngen.ml.data_loaders import DataLoader

from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


def test_initiate_csv_convertor(rp_logger):
    rp_logger.info("Initiating the instance of the class CSVConvertor")
    df, _ = DataLoader(
        f"{DIR_NAME}/unit/convertors/fixtures/csv_tables/table_with_diff_data_types.csv"
    ).load_data()
    convertor = CSVConvertor(df)
    assert convertor.schema == {"fields": {}, "format": "CSV"}
    pd.testing.assert_frame_equal(convertor.preprocessed_df, df)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initiate_avro_convertor(rp_logger):
    rp_logger.info("Initiating the instance of the class AvroConvertor")
    data_loader = DataLoader(
        f"{DIR_NAME}/unit/convertors/fixtures/avro_tables/table_with_diff_data_types.avro"
    )
    df, _ = data_loader.load_data()
    
    original_schema = data_loader.original_schema
    schema = data_loader.file_loader._get_preprocessed_schema(original_schema)

    convertor = AvroConvertor(schema, df)

    assert df.dtypes.to_dict() == {
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

    assert convertor.converted_schema == {
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
    pd.testing.assert_series_equal(convertor.preprocessed_df.dtypes, df.dtypes)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initiate_avro_convertor_without_provided_schema(rp_logger):
    rp_logger.info("Initiating the instance of the class AvroConvertor without a provided schema")
    df, _ = DataLoader(
        f"{DIR_NAME}/unit/convertors/fixtures/avro_tables/table_with_diff_data_types.avro"
    ).load_data()

    convertor = AvroConvertor(schema=None, df=df)

    assert df.dtypes.to_dict() == {
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

    assert convertor.converted_schema == {
        "fields": {},
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
    pd.testing.assert_series_equal(convertor.preprocessed_df.dtypes, df.dtypes)
    rp_logger.info(SUCCESSFUL_MESSAGE)
    

def test_initiate_avro_convertor_with_schema_containing_logical_date_types(rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor"
        "with the schema containing logical date/time types"
    )
    data_loader = DataLoader(
        f"{DIR_NAME}/unit/convertors/fixtures/avro_tables/table_with_date_logical_types.avro"
    )
    df, _ = data_loader.load_data()
    
    original_schema = data_loader.original_schema
    schema = data_loader.file_loader._get_preprocessed_schema(original_schema)

    convertor = AvroConvertor(schema, df)

    assert df.dtypes.to_dict() == {
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

    assert convertor.converted_schema == {
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
    pd.testing.assert_series_equal(convertor.preprocessed_df.dtypes, df.dtypes)
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


@pytest.mark.parametrize("avro_type, column", [
    # Avro 'date' logical type -> pandavro yields object dtype of datetime.date
    (
        {"type": "int", "logicalType": "date"},
        pd.Series([date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)], dtype="object"),
    ),
    # nullable 'date' (union with null)
    (
        ["null", {"type": "int", "logicalType": "date"}],
        pd.Series([date(2023, 1, 1), None, date(2023, 1, 3)], dtype="object"),
    ),
    # Avro 'timestamp-micros' -> pandavro yields tz-aware datetime64[ns, UTC]
    (
        {"type": "long", "logicalType": "timestamp-micros"},
        pd.to_datetime(
            pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
        ).dt.tz_localize("UTC"),
    ),
    # nullable 'timestamp-micros' (union with null) -> tz-aware datetime64[ns, UTC]
    (
        ["null", {"type": "long", "logicalType": "timestamp-micros"}],
        pd.to_datetime(
            pd.Series(["2023-01-01", None, "2023-01-03"])
        ).dt.tz_localize("UTC"),
    ),
    # Avro 'timestamp-millis' (tz-naive datetime64)
    (
        {"type": "long", "logicalType": "timestamp-millis"},
        pd.to_datetime(pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])),
    ),
    # nullable 'timestamp-millis' (union with null)
    (
        ["null", {"type": "long", "logicalType": "timestamp-millis"}],
        pd.to_datetime(pd.Series(["2023-01-01", None, "2023-01-03"])),
    ),
    # Avro 'time-millis' logical type -> pandavro yields object dtype of datetime.time
    (
        {"type": "int", "logicalType": "time-millis"},
        pd.Series([time(10, 0, 0), time(14, 30, 0), time(23, 59, 59)], dtype="object"),
    ),
    # nullable 'time-millis' (union with null)
    (
        ["null", {"type": "int", "logicalType": "time-millis"}],
        pd.Series([time(10, 0, 0), None, time(23, 59, 59)], dtype="object"),
    ),
    # Avro 'time-micros' logical type -> pandavro yields object dtype of datetime.time
    (
        {"type": "long", "logicalType": "time-micros"},
        pd.Series([time(1, 0, 0), time(2, 30, 0), time(23, 59, 59)], dtype="object"),
    ),
    # nullable 'time-micros' (union with null)
    (
        ["null", {"type": "long", "logicalType": "time-micros"}],
        pd.Series([time(1, 0, 0), None, time(23, 59, 59)], dtype="object"),
    ),
    # Avro 'local-timestamp-millis' -> tz-naive datetime64 (no UTC localization)
    (
        {"type": "long", "logicalType": "local-timestamp-millis"},
        pd.to_datetime(pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])),
    ),
    # nullable 'local-timestamp-millis' (union with null)
    (
        ["null", {"type": "long", "logicalType": "local-timestamp-millis"}],
        pd.to_datetime(pd.Series(["2023-01-01", None, "2023-01-03"])),
    ),
    # Avro 'local-timestamp-micros' -> tz-naive datetime64 (no UTC localization)
    (
        {"type": "long", "logicalType": "local-timestamp-micros"},
        pd.to_datetime(pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])),
    ),
    # nullable 'local-timestamp-micros' (union with null)
    (
        ["null", {"type": "long", "logicalType": "local-timestamp-micros"}],
        pd.to_datetime(pd.Series(["2023-01-01", None, "2023-01-03"])),
    )
])
def test_preprocess_df_maps_avro_logical_date_types_to_date(avro_type, column, rp_logger):
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

    assert convertor.converted_schema["fields"] == {"Test": "date"}
    # The date column must be preserved as-is - no numeric coercion, no crash.
    assert convertor.preprocessed_df["Test"].dtype == expected_dtype
    assert not pd.api.types.is_numeric_dtype(convertor.preprocessed_df["Test"])
    assert convertor.preprocessed_df["Test"].dtype == expected_dtype
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_preprocess_df_if_column_is_binary(rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor "
        "with the schema containing one column with data type - 'bytes'"
    )
    df = pd.DataFrame({"Test": [b"test", b"test2", b"test3"]})

    convertor = AvroConvertor({"Test": ["bytes"]}, df)
    assert convertor.preprocessed_df.dtypes.to_dict() == {"Test": "string[python]"}
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
    assert convertor.converted_schema == {
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
            AvroConvertor({"Test": ["null"]}, df)
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

            AvroConvertor(schema, df)

            assert (
                str(error.value)
                == "It seems that the column - 'Test' has unsupported data type - 'test'"
            )
            assert (
                "It seems that the column - 'Test' has unsupported data type - 'test'"
                in caplog.text
            )
    rp_logger.info(SUCCESSFUL_MESSAGE)
