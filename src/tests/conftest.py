import pytest
import os
import logging
import shutil

from loguru import logger
import yaml
from _pytest.logging import LogCaptureFixture
import pandas as pd
from reportportal_client import RPLogger
from cryptography.fernet import Fernet

from syngen.ml.mlflow_tracker import MlflowTracker
from syngen.ml.data_loaders import DataEncryptor


SUCCESSFUL_MESSAGE = "The test passed successfully"
os.environ["FERNET_KEY"] = "VrToTpXdm35CNT3Tur3EGIa2OZ8bfjo-asHo_b-0DTY="
os.environ["FERNET_KEY_2"] = "k64ntCKv3k7ihkNmbjN5cIlkRGxkPoHskJNcKB6bVuI="


def get_dir_name():
    """
    Get the directory name for the current file
    """
    return os.path.abspath(os.path.dirname(__file__))


DIR_NAME = get_dir_name()


@pytest.fixture
def test_csv_path():
    test_path = "test.csv"
    yield test_path
    if os.path.exists(test_path):
        os.remove(test_path)


@pytest.fixture
def test_avro_path():
    test_path = "test.avro"
    yield test_path
    if os.path.exists(test_path):
        os.remove(test_path)


@pytest.fixture
def test_pickle_path():
    test_path = "test.pkl"
    yield test_path
    if os.path.exists(test_path):
        os.remove(test_path)


@pytest.fixture
def test_yaml_path():
    test_path = "test.yaml"
    yield test_path
    if os.path.exists(test_path):
        os.remove(test_path)


@pytest.fixture
def test_yml_path():
    test_path = "test.yml"
    yield test_path
    if os.path.exists(test_path):
        os.remove(test_path)


@pytest.fixture
def test_xls_path():
    test_path = "test.xls"
    yield test_path
    if os.path.exists(test_path):
        os.remove(test_path)


@pytest.fixture
def test_xlsx_path():
    test_path = "test.xlsx"
    yield test_path
    if os.path.exists(test_path):
        os.remove(test_path)


@pytest.fixture
def test_df():
    return pd.DataFrame(
        {
            "gender": [0, 1, 0, 1],
            "height": [
                157.18518021548246,
                166.7731072622863,
                162.91821942384928,
                173.51448996432848,
            ],
            "id": [925, 84, 821, 383],
        }
    )


@pytest.fixture
def test_avro_schema():
    return {
        "type": "record",
        "name": "Root",
        "fields": [
            {"name": "gender", "type": ["null", "long"]},
            {"name": "height", "type": ["null", "double"]},
            {"name": "id", "type": ["null", "long"]}
        ]
    }


@pytest.fixture
def test_metadata_storage():
    os.makedirs("model_artifacts", exist_ok=True)
    path_to_metadata_storage = "model_artifacts/metadata"
    os.makedirs(path_to_metadata_storage, exist_ok=True)
    with open(f"{path_to_metadata_storage}/metadata_of_table_a.yaml", "w") as f:
        yaml.dump(
            {
                "table_a": {
                    "train_settings": {
                        "source": "path/to/table_a.csv",
                        "reports": ["accuracy", "sample"],
                    },
                    "infer_settings": {"destination": "path/to/generated_table_a.csv"},
                    "keys": {
                        "pk_id": {"type": "PK", "columns": ["id"]},
                        "uq_id": {"type": "UQ", "columns": ["name"]},
                    },
                },
            },
            f,
        )

    with open(f"{path_to_metadata_storage}/metadata_of_table_d.yaml", "w") as f:
        yaml.dump(
            {
                "table_d": {
                    "train_settings": {
                        "source": "path/to/table_d.csv",
                        "reports": ["accuracy", "sample"],
                    },
                    "infer_settings": {"destination": "path/to/generated_table_d.csv"},
                    "keys": {
                        "pk_id": {"type": "PK", "columns": ["another_id"]},
                    },
                },
            },
            f,
        )
    yield path_to_metadata_storage
    if os.path.exists(path_to_metadata_storage):
        shutil.rmtree("model_artifacts")


@pytest.fixture
def test_success_file():
    path_to_test_dir = "model_artifacts/resources/table"
    os.makedirs(path_to_test_dir, exist_ok=True)
    success_file_path = f"{path_to_test_dir}/train_message.success"
    with open(success_file_path, "w") as f:
        f.write("PROGRESS")

    yield success_file_path
    if os.path.exists(success_file_path):
        shutil.rmtree("model_artifacts")


@pytest.fixture
def test_metadata_file():
    return {
        "pk_test": {
            "train_settings": {
                "source": "..\\data\\pk_test.csv",
                "drop_null": False,
                "epochs": 1,
                "reports": [],
                "row_limit": 800,
            },
            "infer_settings": {
                "reports": ["accuracy"],
                "random_seed": 1,
                "run_parallel": False,
                "size": 100,
            },
            "keys": {"pk_id": {"columns": ["Id"], "type": "PK"}},
        }
    }


@pytest.fixture(scope="session")
def rp_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.setLoggerClass(RPLogger)
    return logger


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(scope="function")
def mlflow_tracker():
    yield MlflowTracker(experiment_name="test_experiment", is_active=True)


@pytest.fixture
def valid_fernet_key():
    return Fernet.generate_key().decode()


@pytest.fixture
def valid_simple_dataframe():
    return pd.DataFrame({"column1": [1, 2, 3], "column2": ["a", "b", "c"]})


@pytest.fixture
def data_encryptor(tmp_path, valid_fernet_key):
    return DataEncryptor(path=str(tmp_path / "test.dat"), fernet_key=valid_fernet_key)


@pytest.fixture
def invalid_fernet_key():
    return "invalid_key"
