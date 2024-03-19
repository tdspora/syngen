import pytest

import numpy as np
from numpy import dtype
import pandas as pd
import pandavro as pdx

from syngen.ml.convertor import CSVConvertor, AvroConvertor
from syngen.ml.data_loaders import DataLoader

from tests.conftest import SUCCESSFUL_MESSAGE

SCHEMA = {
    "EmployeeKey": ["int", "null"],
    "ParentEmployeeKey": ["long", "null"],
    "EmployeeNationalIDAlternateKey": ["string", "null"],
    "ParentEmployeeNationalIDAlternateKey": ["string", "null"],
    "SalesTerritoryKey": ["int", "null"],
    "FirstName": ["string", "null"],
    "LastName": ["string", "null"],
    "MiddleName": ["string", "null"],
    "NameStyle": ["boolean", "null"],
    "Title": ["string", "null"],
    "HireDate": ["string", "null"],
    "BirthDate": ["string", "null"],
    "LoginID": ["string", "null"],
    "EmailAddress": ["string", "null"],
    "Phone": ["string", "null"],
    "MaritalStatus": ["string", "null"],
    "EmergencyContactName": ["string", "null"],
    "EmergencyContactPhone": ["string", "null"],
    "SalariedFlag": ["boolean", "null"],
    "Gender": ["string", "null"],
    "PayFrequency": ["float", "null"],
    "BaseRate": ["double", "null"],
    "VacationHours": ["int", "null"],
    "SickLeaveHours": ["int", "null"],
    "CurrentFlag": ["boolean", "null"],
    "SalesPersonFlag": ["boolean", "null"],
    "DepartmentName": ["string", "null"],
    "StartDate": ["string", "null"],
    "EndDate": ["string", "null"],
    "Status": ["string", "null"],
    "EmployeePhoto": ["bytes", "null"],
}


def test_initiate_csv_convertor(rp_logger):
    rp_logger.info("Initiating the instance of the class CSVConvertor")
    df, schema = DataLoader(
        "tests/unit/convertors/fixtures/csv_tables/table_with_diff_data_types.csv"
    ).load_data()
    convertor = CSVConvertor(df)
    assert convertor.converted_schema == {"fields": {}, "format": "CSV"}
    pd.testing.assert_frame_equal(convertor.preprocessed_df, df)
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_initiate_avro_convertor(rp_logger):
    rp_logger.info("Initiating the instance of the class AvroConvertor")
    df = pdx.from_avro(
        "tests/unit/convertors/fixtures/avro_tables/table_with_diff_data_types.avro"
    )

    convertor = AvroConvertor(SCHEMA, df)

    assert df.dtypes.to_dict() == {
        "EmployeeKey": dtype("int64"),
        "ParentEmployeeKey": dtype("float64"),
        "EmployeeNationalIDAlternateKey": "string[python]",
        "ParentEmployeeNationalIDAlternateKey": "string[python]",
        "SalesTerritoryKey": dtype("int64"),
        "FirstName": "string[python]",
        "LastName": "string[python]",
        "MiddleName": "string[python]",
        "NameStyle": dtype("int64"),
        "Title": "string[python]",
        "HireDate": "string[python]",
        "BirthDate": "string[python]",
        "LoginID": "string[python]",
        "EmailAddress": "string[python]",
        "Phone": "string[python]",
        "MaritalStatus": "string[python]",
        "EmergencyContactName": "string[python]",
        "EmergencyContactPhone": "string[python]",
        "SalariedFlag": dtype("int64"),
        "Gender": "string[python]",
        "PayFrequency": dtype("int64"),
        "BaseRate": dtype("float64"),
        "VacationHours": dtype("int64"),
        "SickLeaveHours": dtype("int64"),
        "CurrentFlag": dtype("int64"),
        "SalesPersonFlag": dtype("int64"),
        "DepartmentName": "string[python]",
        "StartDate": "string[python]",
        "EndDate": "string[python]",
        "Status": "string[python]",
        "EmployeePhoto": "string[python]",
    }

    assert convertor.converted_schema == {
        "fields": {
            "EmployeeKey": "int",
            "ParentEmployeeKey": "int",
            "EmployeeNationalIDAlternateKey": "string",
            "ParentEmployeeNationalIDAlternateKey": "string",
            "SalesTerritoryKey": "int",
            "FirstName": "string",
            "LastName": "string",
            "MiddleName": "string",
            "NameStyle": "int",
            "Title": "string",
            "HireDate": "string",
            "BirthDate": "string",
            "LoginID": "string",
            "EmailAddress": "string",
            "Phone": "string",
            "MaritalStatus": "string",
            "EmergencyContactName": "string",
            "EmergencyContactPhone": "string",
            "SalariedFlag": "int",
            "Gender": "string",
            "PayFrequency": "float",
            "BaseRate": "float",
            "VacationHours": "int",
            "SickLeaveHours": "int",
            "CurrentFlag": "int",
            "SalesPersonFlag": "int",
            "DepartmentName": "string",
            "StartDate": "string",
            "EndDate": "string",
            "Status": "string",
            "EmployeePhoto": "string",
        },
        "format": "Avro",
    }
    assert convertor.preprocessed_df.dtypes.to_dict() == {
        "EmployeeKey": dtype("int64"),
        "ParentEmployeeKey": dtype("float64"),
        "EmployeeNationalIDAlternateKey": "string[python]",
        "ParentEmployeeNationalIDAlternateKey": "string[python]",
        "SalesTerritoryKey": dtype("int64"),
        "FirstName": "string[python]",
        "LastName": "string[python]",
        "MiddleName": "string[python]",
        "NameStyle": dtype("int64"),
        "Title": "string[python]",
        "HireDate": "string[python]",
        "BirthDate": "string[python]",
        "LoginID": "string[python]",
        "EmailAddress": "string[python]",
        "Phone": "string[python]",
        "MaritalStatus": "string[python]",
        "EmergencyContactName": "string[python]",
        "EmergencyContactPhone": "string[python]",
        "SalariedFlag": dtype("int64"),
        "Gender": "string[python]",
        "PayFrequency": dtype("int64"),
        "BaseRate": dtype("float64"),
        "VacationHours": dtype("int64"),
        "SickLeaveHours": dtype("int64"),
        "CurrentFlag": dtype("int64"),
        "SalesPersonFlag": dtype("int64"),
        "DepartmentName": "string[python]",
        "StartDate": "string[python]",
        "EndDate": "string[python]",
        "Status": "string[python]",
        "EmployeePhoto": "string[python]",
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
                "tests/unit/convertors/fixtures/avro_tables/table_with_diff_data_types.avro"
            )

            schema = {"Test": ["test", "null"]}

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
