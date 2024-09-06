import pytest

import numpy as np
from numpy import dtype
import pandas as pd
import pandavro as pdx

from syngen.ml.convertor import CSVConvertor, AvroConvertor
from syngen.ml.data_loaders import DataLoader

from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME

SCHEMA = {
    "employeekey": ["int", "null"],
    "parentemployeekey": ["int", "null"],
    "employeenationalidalternatekey": ["string", "null"],
    "salesterritorykey": ["int", "null"],
    "firstname": ["string", "null"],
    "lastname": ["string", "null"],
    "middlename": ["string", "null"],
    "namestyle": ["boolean", "null"],
    "title": ["string", "null"],
    "hiredate": ["string", "null"],
    "birthdate": ["string", "null"],
    "loginid": ["string", "null"],
    "emailaddress": ["string", "null"],
    "phone": ["string", "null"],
    "maritalstatus": ["string", "null"],
    "emergencycontactname": ["string", "null"],
    "emergencycontactphone": ["string", "null"],
    "salariedflag": ['boolean', "null"],
    "gender": ["string", "null"],
    "payfrequency": ["int", "null"],
    "baserate": ["double", "null"],
    "vacationhours": ["int", "null"],
    "sickleavehours": ["int", "null"],
    "currentflag": ["boolean", "null"],
    "salespersonflag": ["boolean", "null"],
    "departmentname": ["string", "null"],
    "startdate": ["string", "null"],
    "enddate": ["string", "null"],
    "status": ["string", "null"],
    "employeephoto": ["bytes", "null"]
}


def test_initiate_csv_convertor(rp_logger):
    rp_logger.info("Initiating the instance of the class CSVConvertor")
    df, schema = DataLoader(
        f"{DIR_NAME}/unit/convertors/fixtures/csv_tables/table_with_diff_data_types.csv"
    ).load_data()
    convertor = CSVConvertor({"fields": {}, "format": "CSV"}, df)
    assert convertor.converted_schema == {"fields": {}, "format": "CSV"}
    pd.testing.assert_frame_equal(convertor.preprocessed_df, df)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initiate_avro_convertor(rp_logger):
    rp_logger.info("Initiating the instance of the class AvroConvertor")
    df, schema = DataLoader(
        f"{DIR_NAME}/unit/convertors/fixtures/avro_tables/table_with_diff_data_types.avro"
    ).load_data()

    convertor = AvroConvertor(SCHEMA, df)

    assert df.dtypes.to_dict() == {
        "employeekey": dtype("int64"),
        "parentemployeekey": dtype("int64"),
        "employeenationalidalternatekey": pd.StringDtype(),
        "salesterritorykey": dtype("int64"),
        "firstname": pd.StringDtype(),
        "lastname": pd.StringDtype(),
        "middlename": pd.StringDtype(),
        "namestyle": dtype("int64"),
        "title": pd.StringDtype(),
        "hiredate": pd.StringDtype(),
        "birthdate": pd.StringDtype(),
        "loginid": pd.StringDtype(),
        "emailaddress": pd.StringDtype(),
        "phone": pd.StringDtype(),
        "maritalstatus": pd.StringDtype(),
        "emergencycontactname": pd.StringDtype(),
        "emergencycontactphone": pd.StringDtype(),
        "salariedflag": dtype("int64"),
        "gender": pd.StringDtype(),
        "payfrequency": dtype("int64"),
        "baserate": dtype("float64"),
        "vacationhours": dtype("int64"),
        "sickleavehours": dtype("int64"),
        "currentflag": dtype("int64"),
        "salespersonflag": dtype("int64"),
        "departmentname": pd.StringDtype(),
        "startdate": pd.StringDtype(),
        "enddate": pd.StringDtype(),
        "status": pd.StringDtype(),
        "employeephoto": pd.StringDtype(),
    }

    assert convertor.converted_schema == {
        "fields": {
            "employeekey": "int",
            "parentemployeekey": "int",
            "employeenationalidalternatekey": "string",
            "salesterritorykey": "int",
            "firstname": "string",
            "lastname": "string",
            "middlename": "string",
            "namestyle": "int",
            "title": "string",
            "hiredate": "string",
            "birthdate": "string",
            "loginid": "string",
            "emailaddress": "string",
            "phone": "string",
            "maritalstatus": "string",
            "emergencycontactname": "string",
            "emergencycontactphone": "string",
            "salariedflag": "int",
            "gender": "string",
            "payfrequency": "int",
            "baserate": "float",
            "vacationhours": "int",
            "sickleavehours": "int",
            "currentflag": "int",
            "salespersonflag": "int",
            "departmentname": "string",
            "startdate": "string",
            "enddate": "string",
            "status": "string",
            "employeephoto": "string"
        },
        "format": "Avro"
    }
    assert convertor.preprocessed_df.dtypes.to_dict() == {
        "employeekey": dtype("int64"),
        "parentemployeekey": dtype("int64"),
        "employeenationalidalternatekey": pd.StringDtype(),
        "salesterritorykey": dtype("int64"),
        "firstname": pd.StringDtype(),
        "lastname": pd.StringDtype(),
        "middlename": pd.StringDtype(),
        "namestyle": dtype("int64"),
        "title": pd.StringDtype(),
        "hiredate": pd.StringDtype(),
        "birthdate": pd.StringDtype(),
        "loginid": pd.StringDtype(),
        "emailaddress": pd.StringDtype(),
        "phone": pd.StringDtype(),
        "maritalstatus": pd.StringDtype(),
        "emergencycontactname": pd.StringDtype(),
        "emergencycontactphone": pd.StringDtype(),
        "salariedflag": dtype("int64"),
        "gender": pd.StringDtype(),
        "payfrequency": dtype("int64"),
        "baserate": dtype("float64"),
        "vacationhours": dtype("int64"),
        "sickleavehours": dtype("int64"),
        "currentflag": dtype("int64"),
        "salespersonflag": dtype("int64"),
        "departmentname": pd.StringDtype(),
        "startdate": pd.StringDtype(),
        "enddate": pd.StringDtype(),
        "status": pd.StringDtype(),
        "employeephoto": pd.StringDtype(),
    }
    pd.testing.assert_series_equal(convertor.preprocessed_df.dtypes, df.dtypes)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initiate_avro_convertor_if_schema_contains_unsupported_data_type(caplog, rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor "
        "with the schema containing unsupported data type"
    )
    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            df = pdx.from_avro(
                f"{DIR_NAME}/unit/convertors/fixtures/avro_tables/table_with_diff_data_types.avro"
            )

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
    df = pd.DataFrame({"Test": [1, 2, np.NAN]})

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


def test_preprocess_df_if_column_is_boolean(rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor "
        "with the schema containing one column with data type - 'boolean'"
    )
    df = pd.DataFrame({"Test": [True, False, True]})

    convertor = AvroConvertor({"Test": ["boolean"]}, df)
    assert convertor.preprocessed_df.dtypes.to_dict() == {"Test": "int64"}
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
        "format": "Avro"
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


def test_preprocess_df_if_column_has_unsupported_data_type(rp_logger):
    rp_logger.info(
        "Initiating the instance of the class AvroConvertor "
        "with the schema containing unsupported data type"
    )
    df = pd.DataFrame({"Test": [1, 2, 3]})

    with pytest.raises(ValueError) as error:
        AvroConvertor({"Test": ["test"]}, df)
        assert (
            str(error.value)
            == "It seems that the column - 'Test' has unsupported data type - 'test'"
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)
