import numpy as np
import pytest
import datetime
import random
import string
from unittest.mock import patch, MagicMock

import pandas as pd

from syngen.ml.vae.models.dataset import Dataset
from syngen.ml.data_loaders import DataLoader
from syngen.ml.utils import generate_uuids
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME

CSV_SCHEMA = {"fields": {}, "format": "CSV"}

AVRO_SCHEMA = {
    "fields": {
        "UUIDv1": "string",
        "UUIDv2": "string",
        "UUIDv3": "string",
        "UUIDv4": "string",
        "UUIDv5": "string",
        "ULID": "string",
    },
    "format": "Avro",
}


@pytest.mark.parametrize(
    "path_to_test_table, expected_schema",
    [
        (
            f"{DIR_NAME}/unit/dataset/fixtures/table_with_diff_uuid_columns.csv",
            {}
        ),
        (
            f"{DIR_NAME}/unit/dataset/fixtures/"
            "table_with_diff_uuid_columns_with_missing_values.csv",
            {}
        ),
        (
            f"{DIR_NAME}/unit/dataset/fixtures/table_with_diff_uuid_columns.avro",
            {
                "UUIDv1": "string",
                "UUIDv2": "string",
                "UUIDv3": "string",
                "UUIDv4": "string",
                "UUIDv5": "string",
                "ULID": "string",
            },
        ),
        (
            f"{DIR_NAME}/unit/dataset/fixtures/"
            "table_with_diff_uuid_columns_with_missing_values.avro",
            {
                "UUIDv1": "string",
                "UUIDv2": "string",
                "UUIDv3": "string",
                "UUIDv4": "string",
                "UUIDv5": "string",
                "ULID": "string",
            },
        )
    ],
)
@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
@patch.object(Dataset, "_set_categorical_columns")
def test_is_valid_uuid_defined_in_csv_table_without_missing_values(
    mock_set_categorical_columns,
    mock_fetch_config,
    path_to_test_table,
    expected_schema,
    rp_logger
):
    rp_logger.info(
        "Test the process of the detection of UUID columns"
    )
    df, schema = DataLoader(path_to_test_table).load_data()
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata={"mock_table": {}},
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path",
        },
        main_process="train"
    )
    assert mock_dataset.fields == expected_schema
    mock_dataset.launch_detection()

    assert mock_dataset.uuid_columns == {
        "UUIDv1",
        "UUIDv2",
        "UUIDv3",
        "UUIDv4",
        "UUIDv5",
        "ULID",
    }
    assert mock_dataset.uuid_columns_types == {
        "UUIDv1": 1,
        "UUIDv2": 2,
        "UUIDv3": 3,
        "UUIDv4": 4,
        "UUIDv5": 5,
        "ULID": "ulid",
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_save_dataset(rp_logger):
    rp_logger.info("Test the process of saving the dataset")
    df, schema = DataLoader(f"{DIR_NAME}/unit/dataset/fixtures/data.csv").load_data()
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata={"mock_table": {}},
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    fetched_dataset = mock_dataset.__getstate__()
    assert "df" not in fetched_dataset
    assert set(fetched_dataset.keys()) == {
        "fields",
        "schema_format",
        "metadata",
        "table_name",
        "paths",
        "main_process",
        "features",
        "columns",
        "is_fitted",
        "all_columns",
        "null_num_column_names",
        "zero_num_column_names",
        "nan_labels_dict",
        "uuid_columns",
        "uuid_columns_types",
        "nan_labels_in_uuid",
        "dropped_columns",
        "order_of_columns",
        "custom_categorical_columns",
        "categorical_columns",
        "str_columns",
        "float_columns",
        "int_columns",
        "date_columns",
        "tech_columns",
        "date_mapping",
        "binary_columns",
        "email_columns",
        "long_text_columns",
        "primary_keys_mapping",
        "primary_keys_list",
        "primary_key_name",
        "pk_columns",
        "unique_keys_mapping",
        "unique_keys_mapping_list",
        "unique_keys_list",
        "uq_columns_lists",
        "uq_columns",
        "foreign_keys_mapping",
        "foreign_keys_list",
        "fk_columns",
        "keys_mapping",
        "format",
        "cast_to_float",
        "cast_to_integer",
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_is_valid_categorical_defined_in_csv_table(rp_logger):
    rp_logger.info(
        "Test the process of the detection of "
        "the categorical columns in the table in '.csv' format"
    )
    df, schema = DataLoader(
        f"{DIR_NAME}/unit/dataset/fixtures/table_with_categ_columns.csv"
    ).load_data()
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata={"mock_table": {}},
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path",
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.categorical_columns == {
        "time",
        "ptd_dt",
        "email",
        "id",
        "timestamp",
        "ensure",
        "upd_dt"
    }

    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_set_custom_categorical_columns(rp_logger):
    rp_logger.info(
        "Test the process of the detection of "
        "the categorical columns that has been set by a user"
    )
    df, schema = DataLoader(
        f"{DIR_NAME}/unit/dataset/fixtures/data_with_emails.csv"
    ).load_data()
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata={
            "mock_table": {
                "train_settings": {
                    "source": "path/to/source.csv",
                    "column_types": {
                        "categorical": [
                            "DocNumber",
                            "MetadataSubject",
                            "ExtractedFrom"
                        ]
                    }
                },
                "infer_settings": {},
                "keys": {
                    "pk_key": {
                        "type": "PK",
                        "columns": ["DocNumber"]
                    }
                }
            }
        },
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path",
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.categorical_columns == {
        "SenderPersonId",
        "MetadataSubject",
        "ExtractedFrom"
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_is_valid_binary_defined_in_csv_table(rp_logger):
    rp_logger.info(
        "Test the process of the detection of the binary columns in the table in '.csv' format"
    )
    df, schema = DataLoader(
        f"{DIR_NAME}/unit/dataset/fixtures/table_with_binary_columns.csv"
    ).load_data()
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata={"mock_table": {}},
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path",
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.binary_columns == {
        "time",
        "upd_dt",
        "ptd_dt",
        "email",
        "id",
        "timestamp"
    }
    assert mock_dataset.categorical_columns == {"ensure"}


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_check_non_existent_columns(rp_logger):
    rp_logger.info("Test the process of checking non-existent columns")
    df, schema = DataLoader(f"{DIR_NAME}/unit/dataset/fixtures/data.csv").load_data()
    metadata = {
        "mock_table": {
            "keys": {
                "PK": {
                    "type": "PK",
                    "columns": [
                        "id",
                        "non_existent_pk_column",
                        "non_existent_pk_column_2",
                    ],
                },
                "UQ": {
                    "type": "UQ",
                    "columns": ["first_name", "non_existent_uq_column"],
                },
            }
        }
    }
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.dropped_columns = set()
    mock_dataset.launch_detection()
    assert mock_dataset.non_existent_columns == {
        "non_existent_pk_column",
        "non_existent_uq_column",
        "non_existent_pk_column_2",
    }
    assert mock_dataset.metadata == {
        "mock_table": {
            "keys": {
                "PK": {"type": "PK", "columns": ["id"]},
                "UQ": {"type": "UQ", "columns": ["first_name"]},
            }
        }
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "initial_date_format, expected_date_format",
    [
        ("%Y/%m/%d", "%Y/%m/%d"),
        ("%Y-%m-%d", "%Y-%m-%d"),
        ("%B %d, %Y", "%B %d, %Y"),
        ("%b %d, %Y", "%b %d, %Y"),
        ("%d %B %Y", "%d %B %Y"),
        ("%m-%b-%y", "%d-%m-%Y"),
        ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S"),
        ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"),
        ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M%S"),
        ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z"),
        ("%Y-%m-%dT%H:%M%z", "%Y-%m-%dT%H:%M%z"),
        ("%Y-%m-%dT%H:%M %z", "%Y-%m-%dT%H:%M %z"),
        ("%Y%m%dT%H%M%z", "%Y%m%dT%H%M%z"),
        ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S.%f%z"),
        ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %z"),
        ("%A, %d %b %Y %H:%M:%S %z", "%A, %d %b %Y %H:%M:%S %z"),
        ("%Y-%m-%dT%H:%M:%S%Z", "%Y-%m-%dT%H:%M:%S%Z"),
        ("%Y-%m-%dT%H:%M%Z", "%Y-%m-%dT%H:%M%Z"),
        ("%Y-%m-%dT%H:%M %Z", "%Y-%m-%dT%H:%M %Z"),
        ("%Y%m%dT%H%M%Z", "%Y%m%dT%H%M%Z"),
        ("%Y-%m-%dT%H:%M:%S.%f%Z", "%Y-%m-%dT%H:%M:%S.%f%Z"),
        ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %Z"),
        ("%A, %d %b %Y %H:%M:%S %Z", "%A, %d %b %Y %H:%M:%S %Z"),
        ("%Y-%m-%d %H:%M:%S GMT", "%Y-%m-%d %H:%M:%S %z")
    ]
)
@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_define_date_format_with_diff_format(
    mock_fetch_config,
    initial_date_format,
    expected_date_format,
    rp_logger
):
    rp_logger.info(
        "Test the process of identifying the date format in the date column "
        "where the initial date format - %s and expected date format: %s",
        initial_date_format,
        expected_date_format,
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }

    start_date = datetime.datetime(
        2020,
        1,
        1,
        tzinfo=datetime.timezone.utc if "%z" in initial_date_format.lower() else None
    )
    data = {
        "Date": [(start_date + datetime.timedelta(days=x)).
                 strftime(initial_date_format) for x in range(10000)]
    }

    df = pd.DataFrame(data, columns=["Date"])
    mock_dataset = Dataset(
        df=df,
        schema=CSV_SCHEMA,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.date_mapping == {"Date": expected_date_format}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "initial_date_format, expected_date_format, data",
    [
        ("%m-%d-%Y", "%m-%d-%Y", [
            "01-10-2081", "06-09-2070", "03-23-2077", "05-17-2079",
            "11-17-2084", "01-16-2077", "04-15-2079", "10-28-2082",
            "01-12-2079", "12-11-2084", "04-23-2078", "10-20-2078",
            "02-11-2072", "07-11-2081", "02-12-2071", "05-03-2082",
            "02-21-2085", "09-07-2083", "03-20-2081", "04-19-2070",
            "04-02-2079", "07-15-2081", "06-10-2079", "05-31-2082",
            "12-12-2070", "02-11-2070", "04-13-2087", "08-28-2084",
            "07-08-2080", "11-01-2072", "03-26-2081", "07-05-2075",
            "08-30-2082", "12-14-2079", "05-31-2083", "04-06-2075",
            "01-27-2072", "10-12-2076", "02-24-2075", "04-01-2080",
            "04-09-2082", "06-03-2087", "02-13-2071", "01-30-2082",
            "10-07-2086", "06-30-2083", "01-19-2084", "07-14-2077",
            "11-21-2089", "01-12-2077", "03-12-2084", "02-23-2071",
            "05-25-2079", "03-28-2076", "03-02-2085", "08-30-2089",
            "08-13-2082", "06-02-2075", "06-05-2071", "09-02-2086"
        ]),
        ("%d-%b-%y", "%d-%m-%Y", [
            "02-JUN-79", "30-MAR-85", "08-FEB-76", "22-OCT-81",
            "10-SEP-89", "29-APR-87", "04-AUG-77", "24-JAN-71",
            "08-NOV-80", "01-OCT-79", "23-JUN-81", "08-NOV-71",
            "17-FEB-76", "26-MAY-74", "05-NOV-77", "16-JUN-82",
            "17-JUN-72", "12-JUN-89", "22-DEC-82", "23-OCT-89",
            "25-JAN-79", "11-MAY-85", "12-AUG-83", "21-NOV-80",
            "23-AUG-79", "24-FEB-83", "07-SEP-71", "02-DEC-81",
            "23-FEB-74", "10-NOV-89", "18-MAR-82", "25-APR-73",
            "26-AUG-89", "03-OCT-74", "24-JUN-71", "13-JAN-84",
            "28-MAY-74", "04-MAY-78", "14-JAN-87", "29-JUL-84",
            "26-MAR-84", "02-MAR-71", "26-APR-72", "28-AUG-89",
            "05-AUG-80", "23-AUG-75", "21-SEP-77", "01-FEB-73",
            "23-JUL-83", "01-SEP-83", "21-SEP-88", "23-MAR-85",
            "22-MAY-76", "25-MAR-79", "21-APR-85", "18-OCT-86",
            "12-JUL-84", "04-FEB-70", "29-NOV-77", "25-JUN-89"
        ]),
        ("%d-%m-%y", "%d-%m-%Y", [
            "14-06-81", "18-06-85", "24-04-81", "19-01-80",
            "13-04-87", "27-04-78", "30-11-89", "21-05-89",
            "03-04-73", "20-06-82", "11-01-83", "23-02-83",
            "08-02-74", "23-05-89", "26-09-79", "11-01-88",
            "17-06-89", "17-04-78", "12-07-84", "23-02-86",
            "12-05-82", "23-04-84", "11-06-74", "09-03-74",
            "30-11-87", "17-11-70", "05-04-70", "18-07-71",
            "11-06-85", "19-11-85", "01-05-75", "31-01-76",
            "10-01-81", "03-11-85", "23-05-74", "17-06-88",
            "22-12-77", "24-01-84", "10-10-81", "31-07-74",
            "28-08-74", "01-03-75", "07-08-71", "03-04-70",
            "08-08-88", "01-11-80", "15-01-84", "06-03-86",
            "16-11-81", "22-12-79", "09-02-71", "03-12-70",
            "05-07-81", "23-11-73", "06-04-82", "18-07-71",
            "21-08-75", "19-04-77", "26-01-73", "05-06-81"
        ]),
        ("%d-%m-%Y", "%d-%m-%Y", [
            "01-01-2074", "27-03-2081", "17-03-2087", "09-09-2089",
            "09-08-2086", "11-04-2082", "12-04-2072", "06-09-2081",
            "09-09-2080", "23-03-2070", "17-09-2070", "06-03-2082",
            "23-03-2078", "24-03-2076", "31-03-2073", "28-10-2084",
            "28-09-2074", "27-04-2073", "15-03-2086", "21-10-2080",
            "08-04-2087", "08-07-2070", "13-03-2073", "21-10-2071",
            "17-04-2076", "27-12-2081", "08-04-2070", "01-03-2076",
            "13-02-2071", "05-04-2082", "30-09-2085", "02-04-2071",
            "10-10-2084", "31-10-2085", "16-04-2080", "02-10-2070",
            "26-07-2073", "12-03-2073", "02-03-2074", "16-09-2077",
            "24-03-2070", "12-02-2088", "20-06-2070", "08-08-2075",
            "13-11-2087", "30-11-2086", "26-07-2084", "26-06-2081",
            "22-12-2075", "15-07-2079", "23-09-2070", "23-09-2077",
            "14-11-2074", "16-06-2089", "02-11-2082", "23-08-2082",
            "28-12-2072", "02-04-2076", "03-10-2086", "18-10-2071"
        ]),
        ("%d-%m-%Y", "%d-%m-%Y", [
            "01-01-2074", "27-03-2081", "17-03-2087", "09-09-2089",
            "09-08-2086", "11-04-2082", "12-04-2072", "06-09-2081",
            "09-09-2080", "23-03-2070", "17-09-2070", "06-03-2082",
            "23-03-2078", "24-03-2076", "31-03-2073", "28-10-2084",
            "28-09-2074", "27-04-2073", "15-03-2086", "21-10-2080",
            "08-04-2087", "08-07-2070", "13-03-2073", "21-10-2071",
            "17-04-2076", "27-12-2081", "08-04-2070", "01-03-2076",
            "13-02-2071", "05-04-2082", "30-09-2085", "02-04-2071",
            "10-10-2084", "31-10-2085", "16-04-2080", "02-10-2070",
            "26-07-2073", "12-03-2073", "02-03-2074", "16-09-2077",
            "24-03-2070", "12-02-2088", "20-06-2070", "08-08-2075",
            "13-11-2087", "30-11-2086", "26-07-2084", "26-06-2081",
            "22-12-2075", "15-07-2079", "23-09-2070", "23-09-2077",
            "14-11-2074", "16-06-2089", "02-11-2082", "23-08-2082",
            "28-12-2072", "02-04-2076", "03-10-2086", "18-10-2071",
            np.nan, np.nan, np.nan
        ]),
        ("%m/%d/%Y", "%m/%d/%Y", [
            "02/27/2089", "11/24/2088", "02/16/2087", "08/30/2071",
            "12/14/2071", "09/05/2074", "05/28/2081", "04/24/2075",
            "03/19/2070", "01/02/2086", "02/16/2072", "08/05/2070",
            "09/27/2070", "08/28/2072", "06/03/2087", "11/12/2082",
            "10/25/2080", "01/17/2080", "02/12/2077", "12/23/2073",
            "07/29/2080", "08/19/2079", "10/22/2083", "08/01/2087",
            "02/27/2071", "10/28/2087", "01/13/2076", "03/01/2089",
            "10/01/2071", "09/20/2087", "05/22/2084", "05/26/2076",
            "09/09/2077", "12/04/2083", "08/20/2070", "01/20/2075",
            "10/10/2082", "03/29/2077", "08/13/2076", "07/12/2082",
            "10/19/2082", "01/20/2087", "11/09/2081", "12/04/2078",
            "12/26/2084", "03/25/2072", "01/13/2088", "01/25/2081",
            "04/17/2079", "03/17/2087", "04/10/2071", "08/10/2087",
            "06/12/2078", "07/18/2087", "06/23/2085", "05/30/2084",
            "02/20/2070", "05/13/2072", "05/20/2073", "02/13/2079"
        ]),
        ("%d/%m/%Y", "%d/%m/%Y", [
            "14/09/2087", "07/02/2078", "08/06/2079", "23/02/2087",
            "20/08/2075", "08/05/2071", "05/02/2077", "27/01/2089",
            "22/01/2089", "28/04/2080", "24/06/2081", "05/02/2082",
            "04/09/2087", "21/08/2075", "28/07/2085", "10/04/2084",
            "02/03/2082", "07/09/2074", "24/01/2073", "14/04/2079",
            "29/05/2089", "07/01/2082", "19/08/2076", "11/04/2081",
            "20/04/2074", "06/05/2084", "26/09/2074", "28/09/2085",
            "02/12/2087", "10/06/2085", "01/07/2085", "19/06/2074",
            "16/04/2089", "28/07/2080", "11/03/2083", "25/09/2073",
            "19/02/2073", "29/01/2083", "30/06/2076", "17/01/2085",
            "31/05/2071", "27/07/2080", "17/06/2086", "10/08/2078",
            "03/05/2076", "09/05/2071", "30/05/2084", "25/01/2072",
            "14/08/2088", "09/11/2080", "01/05/2087", "25/06/2087",
            "11/10/2086", "23/01/2075", "19/08/2088", "24/09/2077",
            "09/05/2087", "05/08/2079", "04/11/2080", "12/01/2081"
        ]),
        ("%d.%m.%Y", "%d.%m.%Y", [
            "10.11.2077", "25.03.2080", "07.02.2085", "04.07.2073",
            "31.08.2088", "05.01.2072", "12.12.2085", "16.07.2081",
            "04.10.2071", "15.01.2079", "23.09.2080", "01.12.2081",
            "18.06.2079", "07.04.2081", "10.08.2081", "05.07.2073",
            "01.02.2078", "25.12.2084", "16.09.2070", "24.07.2074",
            "23.03.2081", "09.12.2080", "05.02.2072", "14.10.2081",
            "14.08.2074", "03.06.2074", "28.10.2086", "03.09.2076",
            "05.02.2086", "27.09.2080", "09.04.2077", "04.10.2074",
            "19.03.2077", "27.08.2085", "22.06.2074", "15.09.2073",
            "06.03.2089", "16.06.2078", "04.07.2084", "16.06.2082",
            "31.07.2082", "12.06.2087", "20.08.2073", "26.03.2074",
            "11.11.2072", "18.04.2072", "12.05.2071", "12.10.2080",
            "18.01.2072", "24.12.2073", "30.08.2075", "17.02.2072",
            "02.01.2075", "18.01.2088", "26.09.2080", "18.10.2084",
            "25.01.2071", "25.10.2086", "10.12.2076", "10.12.2079"
        ]),
        ("%Y-%m-%d %I:%M:%S %p", "%Y-%m-%d %I:%M:%S %p", [
            "2020-01-01 12:00:00 AM", "2020-01-01 01:00:00 AM", "2020-01-01 02:00:00 AM",
            "2020-01-01 03:00:00 AM", "2020-01-01 04:00:00 AM", "2020-01-01 05:00:00 AM",
            "2020-01-01 06:00:00 AM", "2020-01-01 07:00:00 AM", "2020-01-01 08:00:00 AM",
            "2020-01-01 09:00:00 AM", "2020-01-01 10:00:00 AM", "2020-01-01 11:00:00 AM",
            "2020-01-01 12:00:00 PM", "2020-01-01 01:00:00 PM", "2020-01-01 02:00:00 PM",
            "2020-01-01 03:00:00 PM", "2020-01-01 04:00:00 PM", "2020-01-01 05:00:00 PM",
            "2020-01-01 06:00:00 PM", "2020-01-01 07:00:00 PM", "2020-01-01 08:00:00 PM",
            "2020-01-01 09:00:00 PM", "2020-01-01 10:00:00 PM", "2020-01-01 11:00:00 PM",
            "2020-01-02 12:00:00 AM", "2020-01-02 01:00:00 AM", "2020-01-02 02:00:00 AM",
            "2020-01-02 03:00:00 AM", "2020-01-02 04:00:00 AM", "2020-01-02 05:00:00 AM",
            "2020-01-02 06:00:00 AM", "2020-01-02 07:00:00 AM", "2020-01-02 08:00:00 AM",
            "2020-01-02 09:00:00 AM", "2020-01-02 10:00:00 AM", "2020-01-02 11:00:00 AM",
            "2020-01-02 12:00:00 PM", "2020-01-02 01:00:00 PM", "2020-01-02 02:00:00 PM",
            "2020-01-02 03:00:00 PM", "2020-01-02 04:00:00 PM", "2020-01-02 05:00:00 PM",
            "2020-01-02 06:00:00 PM", "2020-01-02 07:00:00 PM", "2020-01-02 08:00:00 PM",
            "2020-01-02 09:00:00 PM", "2020-01-02 10:00:00 PM", "2020-01-02 11:00:00 PM",
            "2020-01-03 12:00:00 AM", "2020-01-03 01:00:00 AM", "2020-01-03 02:00:00 AM",
            "2020-01-03 03:00:00 AM", "2020-01-03 04:00:00 AM", "2020-01-03 05:00:00 AM",
            "2020-01-03 06:00:00 AM", "2020-01-03 07:00:00 AM", "2020-01-03 08:00:00 AM",
            "2020-01-03 09:00:00 AM", "2020-01-03 10:00:00 AM", "2020-01-03 11:00:00 AM"
        ]),
        ("%Y-%m-%d %I:%M %p", "%Y-%m-%d %I:%M %p", [
            "2020-01-01 12:00 AM", "2020-01-01 01:00 AM", "2020-01-01 02:00 AM",
            "2020-01-01 03:00 AM", "2020-01-01 04:00 AM", "2020-01-01 05:00 AM",
            "2020-01-01 06:00 AM", "2020-01-01 07:00 AM", "2020-01-01 08:00 AM",
            "2020-01-01 09:00 AM", "2020-01-01 10:00 AM", "2020-01-01 11:00 AM",
            "2020-01-01 12:00 PM", "2020-01-01 01:00 PM", "2020-01-01 02:00 PM",
            "2020-01-01 03:00 PM", "2020-01-01 04:00 PM", "2020-01-01 05:00 PM",
            "2020-01-01 06:00 PM", "2020-01-01 07:00 PM", "2020-01-01 08:00 PM",
            "2020-01-01 09:00 PM", "2020-01-01 10:00 PM", "2020-01-01 11:00 PM",
            "2020-01-02 12:00 AM", "2020-01-02 01:00 AM", "2020-01-02 02:00 AM",
            "2020-01-02 03:00 AM", "2020-01-02 04:00 AM", "2020-01-02 05:00 AM",
            "2020-01-02 06:00 AM", "2020-01-02 07:00 AM", "2020-01-02 08:00 AM",
            "2020-01-02 09:00 AM", "2020-01-02 10:00 AM", "2020-01-02 11:00 AM",
            "2020-01-02 12:00 PM", "2020-01-02 01:00 PM", "2020-01-02 02:00 PM",
            "2020-01-02 03:00 PM", "2020-01-02 04:00 PM", "2020-01-02 05:00 PM",
            "2020-01-02 06:00 PM", "2020-01-02 07:00 PM", "2020-01-02 08:00 PM",
            "2020-01-02 09:00 PM", "2020-01-02 10:00 PM", "2020-01-02 11:00 PM",
            "2020-01-03 12:00 AM", "2020-01-03 01:00 AM", "2020-01-03 02:00 AM",
            "2020-01-03 03:00 AM", "2020-01-03 04:00 AM", "2020-01-03 05:00 AM",
            "2020-01-03 06:00 AM", "2020-01-03 07:00 AM", "2020-01-03 08:00 AM",
            "2020-01-03 09:00 AM", "2020-01-03 10:00 AM", "2020-01-03 11:00 AM"
        ]),
    ]
)
def test_define_date_format_with_diff_format_and_provided_data(
    initial_date_format,
    expected_date_format,
    data,
    rp_logger
):
    rp_logger.info(
        "Test the process of identifying the date format in the date column "
        f"where the initial date format - '{initial_date_format}' "
        f"and expected date format: '{expected_date_format}'"
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }

    df = pd.DataFrame({"Date": data}, columns=["Date"])
    with patch("syngen.ml.vae.models.dataset.fetch_config", lambda x: MagicMock()):
        mock_dataset = Dataset(
            df=df,
            schema=CSV_SCHEMA,
            metadata=metadata,
            table_name="mock_table",
            paths={
                "train_config_pickle_path": "mock_path"
            },
            main_process="train"
        )
    mock_dataset.launch_detection()
    assert mock_dataset.date_mapping == {"Date": expected_date_format}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize(
    "initial_date_format, expected_date_format, extreme_values",
    [
        ("%Y-%m-%d", "%Y-%m-%d", ["4723-10-17T07:45:35Z", "9999-12-31T05:22:15Z"]),
        ("%Y/%m/%d", "%Y/%m/%d", ["3/10/17T07:45:35Z", "9/12/31T05:22:15Z"]),
        ("%m/%d/%Y", "%m/%d/%Y", ["11/30/2017T07:45:35Z", "02/27/1999T05:22:15Z"])
    ]
)
@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_define_date_format_with_extreme_values(
    mock_fetch_config,
    initial_date_format,
    expected_date_format,
    extreme_values,
    rp_logger
):
    rp_logger.info(
        "Test the process of identifying the date format in the date column "
        "where the initial date format - %s and expected date format: %s",
        initial_date_format,
        expected_date_format,
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }
    data = {
        "Date": [(datetime.datetime(2020, 1, 1) + datetime.timedelta(days=x)).
                 strftime(initial_date_format) for x in range(10000)]
    }
    data.update({"Date": data['Date'] + extreme_values})
    df = pd.DataFrame(data, columns=["Date"])
    mock_dataset = Dataset(
        df=df,
        schema=CSV_SCHEMA,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.date_mapping == {"Date": expected_date_format}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_is_valid_uuid(rp_logger):
    rp_logger.info(
        "Test the method 'is_valid_uuid' that checks if the given value is a valid UUID",
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }
    data = {
        "UUID_1": [str(i) for i in generate_uuids(1, 100)],
        "UUID_2": [str(i) for i in generate_uuids(2, 100)],
        "UUID_3": [str(i) for i in generate_uuids(3, 100)],
        "UUID_4": [str(i) for i in generate_uuids(4, 100)],
        "UUID_5": [str(i) for i in generate_uuids(5, 100)],
        "ULID": [str(i) for i in generate_uuids("ulid", 100)],
        "binary": [random.choice([b'0', b'1']) for _ in range(100)],
        "boolean": [random.choice([True, False]) for _ in range(100)],
        "integer": [random.randint(1, 20) for _ in range(100)],
        "float": [random.uniform(1.0, 20.0) for _ in range(100)],
    }
    df = pd.DataFrame(
        data,
        columns=["UUID_1", "UUID_2", "UUID_3", "UUID_4", "UUID_5"]
    )
    mock_dataset = Dataset(
        df=df,
        schema=CSV_SCHEMA,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.uuid_columns == {
        "UUID_1", "UUID_2", "UUID_3", "UUID_4", "UUID_5"
    }
    assert mock_dataset.uuid_columns_types == {
        "UUID_1": 1,
        "UUID_2": 2,
        "UUID_3": 3,
        "UUID_4": 4,
        "UUID_5": 5
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_handling_uuid_columns_with_missing_values(rp_logger):
    rp_logger.info(
        "Test the process of handling uuid columns containing missing values"
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }
    df, schema = DataLoader(
        f"{DIR_NAME}/unit/dataset/fixtures/table_with_diff_uuid_columns_with_nan_labels.csv"
    ).load_data()
    mock_dataset = Dataset(
        df=df,
        schema=CSV_SCHEMA,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.uuid_columns == {"column_with_nan_label"}
    assert mock_dataset.str_columns == {
        "column_with_nan_label_and_nulls",
        "column_with_several_nan_labels",
        "name"
    }
    assert mock_dataset.nan_labels_dict == {
        "column_with_nan_label": "no_uuids"
    }
    assert mock_dataset.uuid_columns_types == {
        "column_with_nan_label": 4.0
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_set_email_columns(rp_logger):
    rp_logger.info(
        "Test the method '_set_email_columns' of the class Dataset",
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }

    df, schema = DataLoader(f"{DIR_NAME}/unit/dataset/fixtures/data_with_emails.csv").load_data()
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.email_columns == {"ExtractedFrom"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_set_long_text_columns(rp_logger):
    rp_logger.info(
        "Test the method '_set_long_text_columns' of the class Dataset",
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }

    alphabet = string.ascii_letters + string.digits + string.punctuation + " "
    df = pd.DataFrame({
        "column1": range(1, 101),
        "column2": range(101, 201),
        "column3": range(201, 301),
        "column4": range(301, 401),
        "long_text_column":
            ["".join(random.choice(alphabet)
                     for _ in range(250))
             for _ in range(1, 96)] +
            [np.NaN] * 5
    })
    mock_dataset = Dataset(
        df=df,
        schema=CSV_SCHEMA,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.long_text_columns == {"long_text_column"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_handle_missing_values_in_numeric_columns_in_csv_file(rp_logger):
    rp_logger.info(
        "Test the process of handling missing values "
        "in numeric columns in a '.csv' file",
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }

    data = {
        "column1": range(1, 101),
        "column2": range(101, 201),
        "column3": range(201, 301),
        "column4": range(301, 401),
        "column5": [str(i) for i in range(401, 491)] + ["Not available" for i in range(10)]
    }
    df = pd.DataFrame(data)

    mock_dataset = Dataset(
        df=df,
        schema=CSV_SCHEMA,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.int_columns == {"column1", "column2", "column3", "column4", "column5"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_cast_to_numeric_in_csv_file(rp_logger):
    rp_logger.info(
        "Test the process of casting the string values to numeric provided in a '.csv' file"
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }

    data = {
        "column1": range(1, 101),
        "column2": [str(i) for i in range(101, 201)],
        "column3": range(201, 301),
        "column4": range(301, 401),
        "column5": [str(i) for i in np.linspace(0, 1, 100)]
    }
    df = pd.DataFrame(data)

    mock_dataset = Dataset(
        df=df,
        schema=CSV_SCHEMA,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.int_columns == {"column1", "column2", "column3", "column4"}
    assert mock_dataset.float_columns == {"column5"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_handle_missing_values_in_numeric_columns_in_avro_file(rp_logger):
    rp_logger.info(
        "Test the process of handling missing values "
        "in numeric columns in a '.avro' file",
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }

    data = {
        "column1": range(1, 101),
        "column2": range(101, 201),
        "column3": range(201, 301),
        "column4": range(301, 401),
        "column5": [str(i) for i in range(401, 491)] + ["Not available" for i in range(10)]
    }
    df = pd.DataFrame(data)

    schema = {
        "format": "Avro",
        "fields": {
            "column1": "int",
            "column2": "int",
            "column3": "int",
            "column4": "int",
            "column5": "string"
        }
    }
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.int_columns == {"column1", "column2", "column3", "column4", "column5"}
    assert mock_dataset.nan_labels_dict == {"column5": "Not available"}
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch("syngen.ml.vae.models.dataset.fetch_config", return_value=MagicMock())
def test_cast_to_numeric_in_avro_file(rp_logger):
    rp_logger.info(
        "Test the process of casting the string values to numeric provided in a '.avro' file"
    )
    metadata = {
        "mock_table": {
            "keys": {}
        }
    }

    data = {
        "column1": range(1, 101),
        "column2": range(101, 201),
        "column3": range(201, 301),
        "column4": range(301, 401),
        "column5": [str(i) for i in range(401, 491)] + [np.NaN for i in range(10)],
        "column6": [str(i) for i in np.linspace(0, 1, 100)]
    }
    df = pd.DataFrame(data)

    schema = {
        "format": "Avro",
        "fields": {
            "column1": "int",
            "column2": "int",
            "column3": "int",
            "column4": "int",
            "column5": "string",
            "column6": "string"
        }
    }
    mock_dataset = Dataset(
        df=df,
        schema=schema,
        metadata=metadata,
        table_name="mock_table",
        paths={
            "train_config_pickle_path": "mock_path"
        },
        main_process="train"
    )
    mock_dataset.launch_detection()
    assert mock_dataset.int_columns == {"column1", "column2", "column3", "column4", "column5"}
    assert mock_dataset.float_columns == {"column6"}
    rp_logger.info(SUCCESSFUL_MESSAGE)
