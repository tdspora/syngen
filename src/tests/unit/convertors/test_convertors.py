import pytest

import numpy as np
from numpy import dtype
import pandas as pd

from syngen.ml.convertor import CSVConvertor, AvroConvertor
from syngen.ml.convertor.convertor import Convertor
from syngen.ml.data_loaders import DataLoader

from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


def _make_binary_convertor(df: pd.DataFrame, column: str) -> Convertor:
    """
    Build a minimal Convertor instance that exercises the binary cast path.
    AvroConvertor maps bytes → string (no binary path), so we use the base
    class directly and configure it to treat `column` as binary.
    """
    convertor = Convertor(original_schema={column: "binary"}, df=df)
    return convertor


def test_initiate_csv_convertor(rp_logger):
    rp_logger.info("Initiating the instance of the class CSVConvertor")
    df, _ = DataLoader(
        f"{DIR_NAME}/unit/convertors/fixtures/csv_tables/table_with_diff_data_types.csv"
    ).load_data()
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
    df, schema = DataLoader(
        f"{DIR_NAME}/unit/convertors/fixtures/avro_tables/table_with_diff_data_types.avro"
    ).load_data()

    convertor = AvroConvertor(schema["fields"], df)

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
        "format": "Avro"
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

    convertor = AvroConvertor(original_schema=None, df=df)

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

    assert convertor.custom_schema == {
        "fields": {},
        "format": "Avro"
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
    assert convertor.custom_schema == {
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


# ---------------------------------------------------------------------------
# Base Convertor binary-path tests
# ---------------------------------------------------------------------------

def test_cast_binary_column_with_ascii_text_decodes_to_string(rp_logger):
    rp_logger.info(
        "Casting a binary column with ASCII plain-text values in the base Convertor "
        "should decode bytes to strings and store the encoding"
    )
    df = pd.DataFrame({"Blob": [b"hello", b"world"]})
    convertor = _make_binary_convertor(df, "Blob")
    convertor._cast_binary_column("Blob")
    assert convertor.preprocessed_df["Blob"].tolist() == ["hello", "world"]
    assert convertor.custom_schema["encoding"]["Blob"] == "ascii"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_non_decodable_mime_skips_decoding(caplog, rp_logger):
    rp_logger.info(
        "Casting a binary column with JPEG image bytes in the base Convertor "
        "should log a warning and leave bytes as-is"
    )
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x00\x00\x00\x01\x00\x01\x00\x00"
    df = pd.DataFrame({"Photo": [jpeg_bytes, jpeg_bytes]})
    convertor = _make_binary_convertor(df, "Photo")
    with caplog.at_level("WARNING"):
        convertor._cast_binary_column("Photo")
    assert "non-decodable MIME type(s): 'image/jpeg'" in caplog.text
    assert convertor.preprocessed_df["Photo"].tolist() == [jpeg_bytes, jpeg_bytes]
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_mixed_encodings_skips_decoding(caplog, rp_logger):
    rp_logger.info(
        "Casting a binary column with mixed-encoding plain-text bytes in the base Convertor "
        "should log a warning and leave bytes as-is"
    )
    utf8_value = "Héllo Wörld für dich testing".encode("utf-8")
    windows1252_value = b"caf\xe9 and na\xefve more text here for detection"
    df = pd.DataFrame({"Text": [utf8_value, windows1252_value]})
    convertor = _make_binary_convertor(df, "Text")
    with caplog.at_level("WARNING"):
        convertor._cast_binary_column("Text")
    assert "multiple different character encodings" in caplog.text
    assert convertor.preprocessed_df["Text"].tolist() == [utf8_value, windows1252_value]
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_html_bytes_skips_decoding(caplog, rp_logger):
    rp_logger.info(
        "Casting a binary column with HTML bytes in the base Convertor "
        "should log a warning and leave bytes as-is"
    )
    html_bytes = (
        b'<!DOCTYPE html><html lang="en"><head>'
        b'<meta charset="UTF-8"><title>Test</title></head>'
        b"<body><p>Hello World</p></body></html>"
    )
    df = pd.DataFrame({"Content": [html_bytes, html_bytes]})
    convertor = _make_binary_convertor(df, "Content")
    with caplog.at_level("WARNING"):
        convertor._cast_binary_column("Content")
    assert "non-decodable MIME type(s): 'text/html'" in caplog.text
    assert convertor.preprocessed_df["Content"].tolist() == [html_bytes, html_bytes]
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_application_bytes_skips_decoding(caplog, rp_logger):
    rp_logger.info(
        "Casting a binary column with application/octet-stream bytes in the base Convertor "
        "should log a warning and leave bytes as-is"
    )
    binary_payload = b"\x00\x00\x00\x00\x00\x00\x00\x00"
    df = pd.DataFrame({"Data": [binary_payload, binary_payload]})
    convertor = _make_binary_convertor(df, "Data")
    with caplog.at_level("WARNING"):
        convertor._cast_binary_column("Data")
    assert "non-decodable MIME type(s): 'application/octet-stream'" in caplog.text
    assert convertor.preprocessed_df["Data"].tolist() == [binary_payload, binary_payload]
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_none_mime_type_and_none_encoding_skips_decoding(
    caplog, rp_logger
):
    rp_logger.info(
        "Casting a binary column where chardet returns mime_type=None and encoding=None "
        "in the base Convertor should log a warning and leave bytes as-is"
    )
    from unittest.mock import patch
    raw_value = b"\x80\x81\x82"
    df = pd.DataFrame({"Data": [raw_value, raw_value]})
    convertor = _make_binary_convertor(df, "Data")
    mock_result = {"encoding": None, "confidence": 0.0, "mime_type": None}
    warning_fragment = "character encoding could not be determined"
    with caplog.at_level("WARNING"):
        with patch("chardet.detect", return_value=mock_result):
            convertor._cast_binary_column("Data")
    assert warning_fragment in caplog.text
    assert convertor.preprocessed_df["Data"].tolist() == [raw_value, raw_value]
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_none_mime_type_and_ascii_encoding_decodes(rp_logger):
    rp_logger.info(
        "Casting a binary column where chardet returns mime_type=None but encoding='ascii' "
        "in the base Convertor should decode the bytes to strings"
    )
    from unittest.mock import patch
    raw_value = b"hello"
    df = pd.DataFrame({"Data": [raw_value, raw_value]})
    convertor = _make_binary_convertor(df, "Data")
    mock_result = {"encoding": "ascii", "confidence": 0.9, "mime_type": None}
    with patch("chardet.detect", return_value=mock_result):
        convertor._cast_binary_column("Data")
    assert convertor.preprocessed_df["Data"].tolist() == ["hello", "hello"]
    assert convertor.custom_schema["encoding"]["Data"] == "ascii"
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_cast_binary_column_with_all_null_values_does_not_raise(rp_logger):
    rp_logger.info(
        "Casting a binary column containing only null values in the base Convertor "
        "should not raise an error"
    )
    df = pd.DataFrame({"Data": [None, None, None]})
    convertor = _make_binary_convertor(df, "Data")
    convertor._cast_binary_column("Data")
    assert convertor.preprocessed_df["Data"].isnull().all()
    assert "encoding" not in convertor.custom_schema
    rp_logger.info(SUCCESSFUL_MESSAGE)
