from unittest.mock import patch, MagicMock
import pytest
import os

import pandas as pd

from syngen.ml.worker import Worker
from syngen.ml.config import Validator
from syngen.ml.utils import ValidationError, fetch_env_variables

from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


FERNET_KEY = os.getenv("FERNET_KEY")
FERNET_KEY_2 = os.getenv("FERNET_KEY_2")


@patch.object(Validator, "run")
def test_init_worker_for_training_process_with_absent_metadata_path(mock_validator_run, rp_logger):
    """
    Test the initialization of 'Worker' class with the absent metadata path
    during the training process
    """
    rp_logger.info(
        "Test the initialization of the instance of 'Worker' class "
        "with the absent metadata during the training process"
    )
    worker = Worker(
        table_name="table",
        metadata_path=None,
        settings={
            "source": "path/to/source.csv",
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "reports": ["accuracy", "sample"],
        },
        log_level="INFO",
        type_of_process="train",
        encryption_settings=fetch_env_variables({"fernet_key": "FERNET_KEY"})
        )
    assert worker.metadata == {
        "table": {
            "train_settings": {
                "source": "path/to/source.csv",
                "batch_size": 1000,
                "drop_null": True,
                "epochs": 20,
                "reports": ["accuracy", "sample"],
                "row_limit": 1000,
            },
            "infer_settings": {},
            "encryption": {
                "fernet_key": FERNET_KEY
            },
            "keys": {},
            "format": {}
        }
    }
    mock_validator_run.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "run")
def test_init_worker_for_infer_process_with_absent_metadata_path(mock_validator_run, rp_logger):
    """
    Test the initialization of 'Worker' class with the absent metadata path
    during the inference process
    """
    rp_logger.info(
        "Test the initialization of the instance of 'Worker' class "
        "with the absent metadata during the inference process"
    )
    worker = Worker(
        table_name="table",
        metadata_path=None,
        settings={
            "size": 100,
            "run_parallel": False,
            "batch_size": 100,
            "reports": [],
            "random_seed": 1,
        },
        log_level="INFO",
        type_of_process="infer",
        encryption_settings=fetch_env_variables({"fernet_key": "FERNET_KEY"})
    )
    assert worker.metadata == {
        "table": {
            "train_settings": {"source": None},
            "infer_settings": {
                "size": 100,
                "run_parallel": False,
                "batch_size": 100,
                "reports": [],
                "random_seed": 1,
            },
            "encryption": {
                "fernet_key": FERNET_KEY
            },
            "keys": {},
            "format": {}
        }
    }
    mock_validator_run.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "run")
def test_init_worker_with_metadata_path(mock_validator_run, rp_logger):
    """
    Test the initialization of 'Worker' class with the metadata path
    """
    rp_logger.info(
        "Test the initialization of the instance of 'Worker' class with the metadata path"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata.yaml",
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "reports": ["accuracy", "sample"],
        },
        log_level="INFO",
        type_of_process="train",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    assert worker.metadata == {
        "table": {
            "train_settings": {
                "source": "./path/to/table.csv",
                "epochs": 100,
                "drop_null": False,
                "reports": [],
                "row_limit": 800,
                "batch_size": 2000,
            },
            "infer_settings": {
                "size": 200,
                "run_parallel": True,
                "random_seed": 2,
                "reports": ["accuracy"],
                "batch_size": 200,
            },
            "encryption": {
                "fernet_key": FERNET_KEY
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["Id"]
                }
            },
            "format": {}
        },
    }
    mock_validator_run.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "run")
def test_init_worker_with_empty_settings_in_metadata_in_train_process(
    mock_validator_run, rp_logger
):
    """
    Test the initialization during the training process
    of 'Worker' class with metadata contained the information of one table
    in which the training, inference, keys settings are empty
    """
    rp_logger.info(
        "Test the initialization of the instance of 'Worker' class with provided metadata "
        "contained the information of one table in which 'train_settings', 'infer_settings', and "
        "'keys' are empty during the training process"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_with_empty_settings.yaml",
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "reports": ["accuracy", "sample"],
        },
        log_level="INFO",
        type_of_process="train",
        encryption_settings=fetch_env_variables({"fernet_key": "FERNET_KEY"})
    )
    assert worker.metadata == {
        "table": {
            "train_settings": {
                "source": "./path/to/table.csv",
                "epochs": 20,
                "drop_null": True,
                "reports": ["accuracy", "sample"],
                "row_limit": 1000,
                "batch_size": 1000,
            },
            "infer_settings": {},
            "encryption": {
                "fernet_key": FERNET_KEY
            },
            "keys": {},
            "format": {}
        },
    }
    mock_validator_run.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "run")
def test_init_worker_with_empty_settings_in_metadata_in_infer_process(
    mock_validator_run, rp_logger
):
    """
    Test the initialization during the inference process
    of 'Worker' class with metadata contained the information of one table
    in which the training, inference, keys settings are empty
    """
    rp_logger.info(
        "Test the initialization of the instance of 'Worker' class with provided metadata "
        "contained the information of one table in which 'train_settings', 'infer_settings', and "
        "'keys' are empty during the inference process"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_with_empty_settings.yaml",
        settings={
            "size": 200,
            "run_parallel": False,
            "batch_size": 200,
            "reports": ["accuracy"],
            "random_seed": 5,
        },
        log_level="INFO",
        type_of_process="infer",
        encryption_settings=fetch_env_variables({"fernet_key": "FERNET_KEY"})
    )
    assert worker.metadata == {
        "table": {
            "train_settings": {
                "source": "./path/to/table.csv"
            },
            "infer_settings": {
                "size": 200,
                "run_parallel": False,
                "batch_size": 200,
                "random_seed": 5,
                "reports": ["accuracy"]
            },
            "encryption": {
                "fernet_key": FERNET_KEY
            },
            "keys": {},
            "format": {}
        },
    }
    mock_validator_run.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "run")
def test_init_worker_for_training_with_metadata_with_global_settings(
    mock_validator_run, rp_logger
):
    """
    Test the initialization of 'Worker' class during the training process
    with the metadata contained related tables and global settings
    """
    rp_logger.info(
        "Test the initialization of the instance of 'Worker' class during the training process "
        "with provided metadata contained related tables and global settings"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_with_global_settings.yaml",
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "reports": ["accuracy", "sample"],
        },
        log_level="INFO",
        type_of_process="train",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    assert worker.metadata == {
        "pk_test": {
            "train_settings": {
                "source": "./path/to/pk_test.csv",
                "row_limit": 800,
                "epochs": 5,
                "drop_null": True,
                "batch_size": 1000,
                "reports": ["accuracy", "sample"],
            },
            "infer_settings": {"reports": []},
            "encryption": {"fernet_key": FERNET_KEY},
            "keys": {"pk_id": {"type": "PK", "columns": ["Id"]}},
            "format": {}
        },
        "fk_test": {
            "keys": {
                "fk_id": {
                    "type": "FK",
                    "columns": ["Id"],
                    "references": {"table": "pk_test", "columns": ["Id"]},
                }
            },
            "train_settings": {
                "source": "./path/to/fk_test.csv",
                "epochs": 5,
                "drop_null": True,
                "row_limit": 500,
                "batch_size": 1000,
                "reports": ["accuracy", "sample"],
            },
            "infer_settings": {},
            "encryption": {"fernet_key": FERNET_KEY},
            "format": {}
        },
    }
    mock_validator_run.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "run")
def test_init_worker_for_inference_with_metadata_with_global_settings(
    mock_validator_run, rp_logger
):
    """
    Test the initialization of 'Worker' class during an inference process
    with metadata contained the information of related tables with the global settings
    """
    rp_logger.info(
        "Test the initialization of the instance of 'Worker' class during the inference process "
        "with provided metadata contained the information of related tables with global settings"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_with_global_settings.yaml",
        settings={
            "size": 200,
            "run_parallel": False,
            "batch_size": 200,
            "reports": [],
            "random_seed": 5,
        },
        log_level="INFO",
        type_of_process="infer",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    assert worker.metadata == {
        "pk_test": {
            "train_settings": {
                "source": "./path/to/pk_test.csv",
                "row_limit": 800
            },
            "infer_settings": {
                "reports": [],
                "size": 1000,
                "run_parallel": True,
                "batch_size": 200,
                "random_seed": 5,
            },
            "encryption": {
                "fernet_key": FERNET_KEY
            },
            "keys": {"pk_id": {"type": "PK", "columns": ["Id"]}},
            "format": {}
        },
        "fk_test": {
            "keys": {
                "fk_id": {
                    "type": "FK",
                    "columns": ["Id"],
                    "references": {"table": "pk_test", "columns": ["Id"]},
                }
            },
            "train_settings": {"source": "./path/to/fk_test.csv"},
            "infer_settings": {
                "size": 1000,
                "run_parallel": True,
                "reports": ["accuracy"],
                "batch_size": 200,
                "random_seed": 5,
            },
            "encryption": {
                "fernet_key": FERNET_KEY
            },
            "format": {}
        },
    }
    mock_validator_run.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables")
def test_launch_train_with_metadata(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_generate_reports,
    mock_collect_metrics_in_train,
    rp_logger,
):
    """
    Test that 'launch_train' method calls all necessary methods
    in case the metadata file contained the information of one table
    with only the primary key was provided
    """
    rp_logger.info(
        "Test that 'launch_train' method calls all necessary methods "
        "in case the metadata file contained the information of one table "
        "with only the primary key was provided"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata.yaml",
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "reports": ["accuracy", "sample"],
        },
        log_level="INFO",
        type_of_process="train",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        ["table"],
        ["table"],
        {
            "table": {
                "train_settings": {
                    "source": "./path/to/table.csv",
                    "epochs": 100,
                    "drop_null": False,
                    "reports": [],
                    "row_limit": 800,
                    "batch_size": 2000,
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": ["accuracy"],
                    "batch_size": 200,
                },
                "encryption": {"fernet_key": FERNET_KEY},
                "keys": {"pk_id": {"type": "PK", "columns": ["Id"]}},
                "format": {}
            }
        },
        {
            "table": {
                "train_settings": {
                    "source": "./path/to/table.csv",
                    "epochs": 100,
                    "drop_null": False,
                    "reports": [],
                    "row_limit": 800,
                    "batch_size": 2000,
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": ["accuracy"],
                    "batch_size": 200,
                },
                "encryption": {"fernet_key": FERNET_KEY},
                "keys": {"pk_id": {"type": "PK", "columns": ["Id"]}},
                "format": {}
            }
        },
        False
    )
    mock_gather_existed_columns.assert_called_once_with("table")
    mock_check_existence_of_source.assert_called_once_with("table")
    mock_check_existence_of_key_columns.assert_called_once_with("table")
    mock_check_existence_of_referenced_columns.assert_called_once_with("table")
    mock_validate_fernet_key.assert_called_once_with("table", FERNET_KEY)
    mock_validate_metadata.assert_called_once_with("table")
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(
        ["table"],
        ["table"],
        False
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables")
def test_launch_train_with_metadata_of_related_tables(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_generate_reports,
    mock_collect_metrics_in_train,
    rp_logger,
):
    """
    Test that 'launch_train' method calls all necessary methods
    in case the metadata file of related tables was provided
    """
    rp_logger.info(
        "Test that 'launch_train' method calls all necessary methods in case "
        "the metadata file of related tables was provided"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_of_related_tables.yaml",
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "reports": ["accuracy", "sample"],
        },
        log_level="INFO",
        type_of_process="train",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        ["pk_test", "fk_test"],
        ["pk_test", "fk_test"],
        {
            "pk_test": {
                "train_settings": {
                    "source": "./path/to/pk_test.csv",
                    "epochs": 1,
                    "drop_null": False,
                    "row_limit": 800,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "reports": ["accuracy"],
                },
                "encryption": {"fernet_key": None},
                "keys": {"pk_id": {"type": "PK", "columns": ["Id"]}},
                "format": {}
            },
            "fk_test": {
                "train_settings": {
                    "source": "./path/to/fk_test.csv",
                    "epochs": 5,
                    "drop_null": True,
                    "reports": ["accuracy", "sample"],
                    "row_limit": 600,
                    "batch_size": 1000,
                },
                "infer_settings": {
                    "size": 90,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": [],
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "fk_id": {
                        "type": "FK",
                        "columns": ["Id"],
                        "references": {"table": "pk_test", "columns": ["Id"]},
                    }
                },
                "format": {}
            },
        },
        {
            "pk_test": {
                "train_settings": {
                    "source": "./path/to/pk_test.csv",
                    "epochs": 1,
                    "drop_null": False,
                    "row_limit": 800,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "reports": ["accuracy"],
                },
                "encryption": {"fernet_key": None},
                "keys": {"pk_id": {"type": "PK", "columns": ["Id"]}},
                "format": {}
            },
            "fk_test": {
                "train_settings": {
                    "source": "./path/to/fk_test.csv",
                    "epochs": 5,
                    "drop_null": True,
                    "reports": ["accuracy", "sample"],
                    "row_limit": 600,
                    "batch_size": 1000,
                },
                "infer_settings": {
                    "size": 90,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": [],
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "fk_id": {
                        "type": "FK",
                        "columns": ["Id"],
                        "references": {"table": "pk_test", "columns": ["Id"]},
                    }
                },
                "format": {}
            },
        },
        True
    )
    assert mock_gather_existed_columns.call_count == 2
    assert mock_check_existence_of_source.call_count == 2
    assert mock_check_existence_of_key_columns.call_count == 2
    assert mock_check_existence_of_referenced_columns.call_count == 2
    mock_validate_fernet_key.assert_not_called()
    assert mock_validate_metadata.call_count == 2
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(
        ["pk_test", "fk_test"],
        ["pk_test", "fk_test"],
        True
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables")
def test_launch_train_with_metadata_of_related_tables_with_diff_keys(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_generate_reports,
    mock_collect_metrics_in_train,
    rp_logger,
):
    """
    Test that 'launch_train' method calls all necessary methods
    by providing the metadata file of related tables
    in which the table contains several different keys
    """
    rp_logger.info(
        "Test that 'launch_train' method calls all necessary methods by providing "
        "the metadata file of related tables in which the table contains several different keys"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                      "metadata_of_related_tables_with_diff_keys.yaml",
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "reports": ["accuracy", "sample"],
        },
        log_level="INFO",
        type_of_process="train",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        ["tdm_models", "tdm_clusters"],
        ["tdm_clusters", "tdm_models_pk", "tdm_models_fk"],
        {
            "tdm_models": {
                "train_settings": {
                    "source": "./path/to/tdm_models.csv",
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "keys": {
                    "tdm_models_pkey": {"type": "PK", "columns": ["id"]},
                    "tdm_models_fkey": {
                        "type": "FK",
                        "columns": ["cluster_id"],
                        "references": {"table": "tdm_clusters", "columns": ["id"]},
                    },
                },
                "infer_settings": {},
                "encryption": {"fernet_key": None},
                "format": {}
            },
            "tdm_clusters": {
                "train_settings": {
                    "source": "./path/to/tdm_clusters.csv",
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "keys": {"tdm_clusters_pkey": {"type": "PK", "columns": ["id"]}},
                "infer_settings": {},
                "encryption": {"fernet_key": None},
                "format": {}
            },
        },
        {
            "tdm_models": {
                "train_settings": {
                    "source": "./path/to/tdm_models.csv",
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "keys": {
                    "tdm_models_pkey": {"type": "PK", "columns": ["id"]},
                    "tdm_models_fkey": {
                        "type": "FK",
                        "columns": ["cluster_id"],
                        "references": {"table": "tdm_clusters", "columns": ["id"]},
                    },
                },
                "infer_settings": {},
                "encryption": {"fernet_key": None},
                "format": {}
            },
            "tdm_clusters": {
                "train_settings": {
                    "source": "./path/to/tdm_clusters.csv",
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "keys": {"tdm_clusters_pkey": {"type": "PK", "columns": ["id"]}},
                "infer_settings": {},
                "encryption": {"fernet_key": None},
                "format": {}
            },
            "tdm_models_pk": {
                "train_settings": {
                    "source": "./path/to/tdm_models.csv",
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "keys": {"tdm_models_pkey": {"type": "PK", "columns": ["id"]}},
                "infer_settings": {},
                "encryption": {"fernet_key": None},
                "format": {}
            },
            "tdm_models_fk": {
                "train_settings": {
                    "source": "./path/to/tdm_models.csv",
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "keys": {
                    "tdm_models_fkey": {
                        "type": "FK",
                        "columns": ["cluster_id"],
                        "references": {"table": "tdm_clusters", "columns": ["id"]},
                    }
                },
                "infer_settings": {},
                "encryption": {"fernet_key": None},
                "format": {}
            },
        },
        True
    )
    assert mock_gather_existed_columns.call_count == 2
    assert mock_check_existence_of_source.call_count == 2
    assert mock_check_existence_of_key_columns.call_count == 2
    assert mock_check_existence_of_referenced_columns.call_count == 2
    mock_validate_fernet_key.assert_not_called()
    assert mock_validate_metadata.call_count == 2
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(
        ["tdm_models", "tdm_clusters"],
        ["tdm_clusters", "tdm_models_pk", "tdm_models_fk"],
        True
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables")
def test_launch_train_without_metadata(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_generate_reports,
    mock_collect_metrics_in_train,
    rp_logger,
):
    """
    Test that 'launch_train' method calls all necessary methods
    in case the metadata file wasn't provided and training process was launched through CLI
    """
    rp_logger.info(
        "Test that 'launch_train' method calls all necessary methods "
        "in case the metadata file wasn't provided and a training process was launched through CLI"
    )
    worker = Worker(
        table_name="table",
        metadata_path=None,
        settings={
            "source": "./path/to/source.csv",
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "reports": ["accuracy", "sample"],
        },
        log_level="INFO",
        type_of_process="train",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        ["table"],
        ["table"],
        {
            "table": {
                "train_settings": {
                    "source": "./path/to/source.csv",
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "infer_settings": {},
                "encryption": {"fernet_key": None},
                "keys": {},
                "format": {}
            }
        },
        {
            "table": {
                "train_settings": {
                    "source": "./path/to/source.csv",
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "infer_settings": {},
                "encryption": {"fernet_key": None},
                "keys": {},
                "format": {}
            }
        },
        True
    )
    mock_gather_existed_columns.assert_called_once_with("table")
    mock_check_existence_of_source.assert_called_once_with("table")
    mock_check_existence_of_key_columns.assert_called_once_with("table")
    mock_check_existence_of_referenced_columns.assert_called_once_with("table")
    mock_validate_fernet_key.assert_not_called()
    mock_validate_metadata.assert_called_once_with("table")
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(
        ["table"],
        ["table"],
        True
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables")
def test_launch_train_with_metadata_contained_global_settings(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_generate_reports,
    mock_collect_metrics_in_train,
    rp_logger,
):
    """
    Test that 'launch_train' method calls all necessary methods
    in case the metadata file of related tables was provided, and it contains global settings
    """
    rp_logger.info(
        "Test that 'launch_train' method calls all necessary methods "
        "in case the metadata file of related tables was provided, "
        "and it contains global settings"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_with_global_settings.yaml",
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "reports": ["accuracy", "sample"],
        },
        log_level="INFO",
        type_of_process="train",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        ["pk_test", "fk_test"],
        ["pk_test", "fk_test"],
        {
            "pk_test": {
                "train_settings": {
                    "source": "./path/to/pk_test.csv",
                    "row_limit": 800,
                    "epochs": 5,
                    "drop_null": True,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "infer_settings": {"reports": []},
                "encryption": {"fernet_key": FERNET_KEY},
                "keys": {"pk_id": {"type": "PK", "columns": ["Id"]}},
                "format": {}
            },
            "fk_test": {
                "keys": {
                    "fk_id": {
                        "type": "FK",
                        "columns": ["Id"],
                        "references": {"table": "pk_test", "columns": ["Id"]},
                    }
                },
                "train_settings": {
                    "source": "./path/to/fk_test.csv",
                    "epochs": 5,
                    "drop_null": True,
                    "row_limit": 500,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "infer_settings": {},
                "encryption": {"fernet_key": FERNET_KEY},
                "format": {}
            },
        },
        {
            "pk_test": {
                "train_settings": {
                    "source": "./path/to/pk_test.csv",
                    "row_limit": 800,
                    "epochs": 5,
                    "drop_null": True,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "infer_settings": {"reports": []},
                "encryption": {"fernet_key": FERNET_KEY},
                "keys": {"pk_id": {"type": "PK", "columns": ["Id"]}},
                "format": {}
            },
            "fk_test": {
                "keys": {
                    "fk_id": {
                        "type": "FK",
                        "columns": ["Id"],
                        "references": {"table": "pk_test", "columns": ["Id"]},
                    }
                },
                "train_settings": {
                    "source": "./path/to/fk_test.csv",
                    "epochs": 5,
                    "drop_null": True,
                    "row_limit": 500,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "infer_settings": {},
                "encryption": {"fernet_key": FERNET_KEY},
                "format": {}
            },
        },
        True
    )
    assert mock_gather_existed_columns.call_count == 2
    assert mock_check_existence_of_source.call_count == 2
    assert mock_check_existence_of_key_columns.call_count == 2
    assert mock_check_existence_of_referenced_columns.call_count == 2
    assert mock_validate_metadata.call_count == 2
    assert mock_validate_fernet_key.call_count == 2
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(
        ["pk_test", "fk_test"],
        ["pk_test", "fk_test"],
        True
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Worker, "_Worker__infer_tables")
def test_launch_infer_with_metadata(
    mock_infer_tables,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_check_access_to_input_data,
    mock_generate_reports,
    mock_collect_metrics_in_infer,
    rp_logger,
):
    """
    Test that 'launch_infer' method calls all necessary methods
    in case the metadata file of one table with only the primary key was provided
    """
    rp_logger.info(
        "Test that 'launch_infer' method calls all necessary methods "
        "in case the metadata file of one table with only the primary key was provided"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata.yaml",
        settings={
            "size": 200,
            "run_parallel": True,
            "random_seed": 2,
            "reports": ["accuracy", "sample"],
            "batch_size": 200,
        },
        log_level="INFO",
        type_of_process="infer",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_infer()
    mock_infer_tables.assert_called_once_with(
        ["table"],
        {
            "table": {
                "train_settings": {
                    "source": "./path/to/table.csv",
                    "epochs": 100,
                    "drop_null": False,
                    "reports": [],
                    "row_limit": 800,
                    "batch_size": 2000,
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": ["accuracy"],
                    "batch_size": 200,
                },
                "encryption": {"fernet_key": FERNET_KEY},
                "keys": {"pk_id": {"type": "PK", "columns": ["Id"]}},
                "format": {}
            }
        },
        0.25,
        type_of_process="infer"
    )
    mock_check_completion_of_training.assert_called_once_with("table")
    mock_check_existence_of_destination.assert_called_once_with("table")
    mock_validate_metadata.assert_called_once_with("table")
    mock_validate_fernet_key.assert_called_once_with("table", FERNET_KEY)
    mock_check_access_to_input_data.assert_called_once_with("table")
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_infer.assert_called_once_with(["table"])
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Worker, "_Worker__infer_tables")
def test_launch_infer_with_metadata_of_related_tables(
    mock_infer_tables,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_check_access_to_input_data,
    mock_generate_reports,
    mock_collect_metrics_in_infer,
    rp_logger,
):
    """
    Test that 'launch_infer' method calls all necessary methods
    in case the metadata file of related tables was provided
    """
    rp_logger.info(
        "Test that 'launch_infer' method calls all necessary methods in case "
        "the metadata file of related tables was provided"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_of_related_tables.yaml",
        settings={
            "size": 300,
            "run_parallel": True,
            "reports": ["accuracy"],
            "batch_size": 200,
            "random_seed": 1,
        },
        log_level="INFO",
        type_of_process="infer",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_infer()
    mock_infer_tables.assert_called_once_with(
        ["pk_test", "fk_test"],
        {
            "pk_test": {
                "train_settings": {
                    "source": "./path/to/pk_test.csv",
                    "epochs": 1,
                    "drop_null": False,
                    "row_limit": 800,
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "reports": ["accuracy"],
                    "batch_size": 200,
                    "random_seed": 1,
                },
                "encryption": {"fernet_key": None},
                "keys": {"pk_id": {"type": "PK", "columns": ["Id"]}},
                "format": {}
            },
            "fk_test": {
                "train_settings": {
                    "source": "./path/to/fk_test.csv",
                    "epochs": 5,
                    "drop_null": True,
                    "reports": ["accuracy", "sample"],
                    "row_limit": 600,
                },
                "infer_settings": {
                    "size": 90,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": [],
                    "batch_size": 200,
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "fk_id": {
                        "type": "FK",
                        "columns": ["Id"],
                        "references": {"table": "pk_test", "columns": ["Id"]},
                    }
                },
                "format": {}
            },
        },
        0.125,
        type_of_process="infer"
    )
    assert mock_check_completion_of_training.call_count == 2
    assert mock_check_existence_of_destination.call_count == 2
    assert mock_validate_metadata.call_count == 2
    mock_validate_fernet_key.assert_not_called()
    mock_check_access_to_input_data.assert_called_once_with("pk_test")
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_infer.assert_called_once_with(["pk_test", "fk_test"])
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Worker, "_Worker__infer_tables")
def test_launch_infer_with_metadata_of_related_tables_with_diff_keys(
    mock_infer_tables,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_check_access_to_input_data,
    mock_generate_reports,
    mock_collect_metrics_in_infer,
    rp_logger,
):
    """
    Test that 'launch_infer' method calls all necessary methods
    by providing the metadata file of related tables in which
    the table contains several different keys was provided
    """
    rp_logger.info(
        "Test that 'launch_infer' method calls all necessary methods "
        "by providing the metadata file of related tables "
        "in which the table contains several different keys was provided"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                      "metadata_of_related_tables_with_diff_keys.yaml",
        settings={
            "size": 300,
            "run_parallel": True,
            "reports": ["accuracy"],
            "batch_size": 200,
            "random_seed": 1,
        },
        log_level="INFO",
        type_of_process="infer",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_infer()
    mock_infer_tables.assert_called_once_with(
        ["tdm_clusters", "tdm_models_pk", "tdm_models_fk"],
        {
            "tdm_models": {
                "train_settings": {"source": "./path/to/tdm_models.csv"},
                "keys": {
                    "tdm_models_pkey": {"type": "PK", "columns": ["id"]},
                    "tdm_models_fkey": {
                        "type": "FK",
                        "columns": ["cluster_id"],
                        "references": {"table": "tdm_clusters", "columns": ["id"]},
                    },
                },
                "infer_settings": {
                    "size": 300,
                    "run_parallel": True,
                    "reports": ["accuracy"],
                    "batch_size": 200,
                    "random_seed": 1,
                },
                "encryption": {"fernet_key": None},
                "format": {}
            },
            "tdm_clusters": {
                "train_settings": {"source": "./path/to/tdm_clusters.csv"},
                "keys": {"tdm_clusters_pkey": {"type": "PK", "columns": ["id"]}},
                "infer_settings": {
                    "size": 300,
                    "run_parallel": True,
                    "reports": ["accuracy"],
                    "batch_size": 200,
                    "random_seed": 1,
                },
                "encryption": {"fernet_key": None},
                "format": {}
            },
            "tdm_models_pk": {
                "train_settings": {"source": "./path/to/tdm_models.csv"},
                "keys": {"tdm_models_pkey": {"type": "PK", "columns": ["id"]}},
                "infer_settings": {
                    "size": 300,
                    "run_parallel": True,
                    "reports": ["accuracy"],
                    "batch_size": 200,
                    "random_seed": 1,
                },
                "encryption": {"fernet_key": None},
                "format": {}
            },
            "tdm_models_fk": {
                "train_settings": {"source": "./path/to/tdm_models.csv"},
                "keys": {
                    "tdm_models_fkey": {
                        "type": "FK",
                        "columns": ["cluster_id"],
                        "references": {"table": "tdm_clusters", "columns": ["id"]},
                    }
                },
                "infer_settings": {
                    "size": 300,
                    "run_parallel": True,
                    "reports": ["accuracy"],
                    "batch_size": 200,
                    "random_seed": 1,
                },
                "encryption": {"fernet_key": None},
                "format": {}
            },
        },
        0.08333333333333333,
        type_of_process="infer"
    )
    assert mock_check_completion_of_training.call_count == 2
    assert mock_check_existence_of_destination.call_count == 2
    mock_validate_fernet_key.assert_not_called()
    assert mock_validate_metadata.call_count == 2
    assert mock_check_access_to_input_data.call_count == 2
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_infer.assert_called_once_with(
        ["tdm_clusters", "tdm_models_pk", "tdm_models_fk"]
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Worker, "_Worker__infer_tables")
def test_launch_infer_without_metadata(
    mock_infer_tables,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_check_access_to_input_data,
    mock_generate_reports,
    mock_collect_metrics_in_infer,
    rp_logger,
):
    """
    Test that 'launch_infer' method calls all necessary methods
    in case the metadata file wasn't provided and the inference process was launched through CLI
    """
    rp_logger.info(
        "Test that 'launch_infer' method calls all necessary methods "
        "in case the metadata file wasn't provided and "
        "the inference process was launched through CLI"
    )
    worker = Worker(
        table_name="table",
        metadata_path=None,
        settings={
            "size": 200,
            "run_parallel": True,
            "random_seed": 2,
            "reports": ["accuracy"],
            "batch_size": 200,
        },
        log_level="INFO",
        type_of_process="infer",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_infer()
    mock_infer_tables.assert_called_once_with(
        ["table"],
        {
            "table": {
                "train_settings": {"source": None},
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": ["accuracy"],
                    "batch_size": 200,
                },
                "encryption": {"fernet_key": None},
                "keys": {},
                "format": {}
            }
        },
        0.25,
        type_of_process="infer"
    )
    mock_check_completion_of_training.assert_called_once_with("table")
    mock_check_existence_of_destination.assert_called_once_with("table")
    mock_validate_metadata.assert_called_once_with("table")
    mock_validate_fernet_key.assert_not_called()
    mock_check_access_to_input_data.assert_called_once_with("table")
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_infer.assert_called_once_with(["table"])
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Worker, "_Worker__infer_tables")
def test_launch_infer_with_metadata_contained_global_settings(
    mock_infer_tables,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_check_access_to_input_data,
    mock_generate_reports,
    mock_collect_metrics_infer,
    rp_logger,
):
    """
    Test that 'launch_infer' method calls all necessary methods
    in case the metadata file of related tables was provided, and it contains global settings
    """
    rp_logger.info(
        "Test that 'launch_infer' method calls all necessary methods "
        "in case the metadata file of related tables was provided, "
        "and it contains global settings"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_with_global_settings.yaml",
        settings={
            "size": 300,
            "run_parallel": True,
            "random_seed": 3,
            "reports": ["accuracy"],
            "batch_size": 300,
        },
        log_level="INFO",
        type_of_process="infer",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_infer()
    mock_infer_tables.assert_called_once_with(
        ["pk_test", "fk_test"],
        {
            "pk_test": {
                "train_settings": {"source": "./path/to/pk_test.csv", "row_limit": 800},
                "infer_settings": {
                    "reports": [],
                    "size": 1000,
                    "run_parallel": True,
                    "random_seed": 3,
                    "batch_size": 300,
                },
                "encryption": {"fernet_key": FERNET_KEY},
                "keys": {"pk_id": {"type": "PK", "columns": ["Id"]}},
                "format": {}
            },
            "fk_test": {
                "keys": {
                    "fk_id": {
                        "type": "FK",
                        "columns": ["Id"],
                        "references": {"table": "pk_test", "columns": ["Id"]},
                    }
                },
                "train_settings": {"source": "./path/to/fk_test.csv"},
                "infer_settings": {
                    "size": 1000,
                    "run_parallel": True,
                    "reports": ["accuracy"],
                    "random_seed": 3,
                    "batch_size": 300,
                },
                "encryption": {"fernet_key": FERNET_KEY},
                "format": {}
            },
        },
        0.125,
        type_of_process="infer",
    )
    assert mock_check_completion_of_training.call_count == 2
    assert mock_check_existence_of_destination.call_count == 2
    assert mock_validate_metadata.call_count == 2
    mock_validate_fernet_key.assert_called_once_with("fk_test", FERNET_KEY)
    mock_check_access_to_input_data.assert_called_once_with("fk_test")
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_infer.assert_called_once_with(["pk_test", "fk_test"])
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Worker, "_Worker__infer_tables")
def test_launch_infer_with_metadata_contained_several_fernet_keys(
    mock_infer_tables,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_check_access_to_input_data,
    mock_generate_reports,
    mock_collect_metrics_infer,
    rp_logger,
):
    """
    Test that 'launch_infer' method calls all necessary methods
    in case the metadata file contained several fernet keys was provided,
    and it contains global settings
    """
    rp_logger.info(
        "Test that 'launch_infer' method calls all necessary methods "
        "in case the metadata file contained several fernet keys was provided, "
        "and it contains global settings"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                      f"metadata_with_several_fernet_keys.yaml",
        settings={
            "size": 300,
            "run_parallel": True,
            "random_seed": 3,
            "reports": ["accuracy"],
            "batch_size": 300,
        },
        log_level="INFO",
        type_of_process="infer",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_infer()
    mock_infer_tables.assert_called_once_with(
        ["pk_test", "fk_test"],
        {
            "pk_test": {
                "train_settings": {
                    "source": "./path/to/pk_test.csv",
                    "epochs": 1,
                    "drop_null": False,
                    "row_limit": 800
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "reports": ["accuracy"],
                    "random_seed": 3,
                    "batch_size": 300
                },
                "encryption": {
                    "fernet_key": FERNET_KEY_2
                },
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"]
                    }
                },
                "format": {}
            },
            "fk_test": {
                "train_settings": {
                    "source": "./path/to/fk_test.csv",
                    "epochs": 5,
                    "drop_null": True,
                    "reports": ["accuracy", "sample"],
                    "row_limit": 600
                },
                "infer_settings": {
                    "size": 90,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": [],
                    "batch_size": 300
                },
                "keys": {
                    "fk_id": {
                        "type": "FK",
                        "columns": ["Id"],
                        "references": {
                            "table": "pk_test",
                            "columns": ["Id"]
                        }
                    }
                },
                "encryption": {
                    "fernet_key": FERNET_KEY
                },
                "format": {}
            }
        },
        0.125,
        type_of_process="infer"
    )
    assert mock_check_completion_of_training.call_count == 2
    assert mock_check_existence_of_destination.call_count == 2
    assert mock_validate_metadata.call_count == 2
    mock_validate_fernet_key.assert_called_once_with("pk_test", FERNET_KEY_2)
    mock_check_access_to_input_data.assert_called_once_with("pk_test")
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_infer.assert_called_once_with(["pk_test", "fk_test"])
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Worker, "_generate_reports")
@patch.object(
    Worker, "_Worker__preprocess_data",
    return_value=(
            pd.DataFrame(
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
            ),
            {"fields": {}, "format": "CSV"})
)
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_infer_table")
@patch.object(Worker, "_train_table")
def test_train_tables_without_generation_reports(
    mock_train_table,
    mock_infer_table,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_preprocess_data,
    mock_generate_reports,
    mock_collect_metrics_in_train,
    rp_logger,
):
    """
    Test the '__train_tables' method of the 'Worker' class
    in case the reports won't be generated
    """
    rp_logger.info(
        "Test the '__train_tables' method of the 'Worker' class "
        "in case the reports won't be generated"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_without_reports.yaml",
        settings={},
        log_level="INFO",
        type_of_process="train",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_train()
    mock_preprocess_data.assert_called_once_with(table_name="test_table")
    mock_gather_existed_columns.assert_called_once_with("test_table")
    mock_check_existence_of_source.assert_called_once_with("test_table")
    mock_check_existence_of_key_columns.assert_called_once_with("test_table")
    mock_check_existence_of_referenced_columns.assert_called_once_with("test_table")
    mock_check_completion_of_training.assert_not_called()
    mock_check_existence_of_destination.assert_not_called()
    mock_validate_fernet_key.assert_not_called()
    mock_validate_metadata.assert_called_once_with("test_table")
    mock_train_table.assert_called_once()
    mock_infer_table.assert_not_called()
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(["test_table"], ["test_table"], False)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Worker, "_generate_reports")
@patch.object(
    Worker, "_Worker__preprocess_data",
    return_value=(
            pd.DataFrame(
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
            ),
            {"fields": {}, "format": "CSV"})
)
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_infer_table")
@patch.object(Worker, "_train_table")
def test_train_tables_with_generation_reports(
    mock_train_table,
    mock_infer_table,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    mock_validate_fernet_key,
    mock_check_existence_of_destination,
    mock_validate_metadata,
    mock_preprocess_data,
    mock_generate_reports,
    mock_collect_metrics_in_train,
    rp_logger,
):
    """
    Test the '__train_tables' method of the 'Worker' class
    in case the reports will be generated
    """
    rp_logger.info(
        "Test the '__train_tables' method of the 'Worker' class "
        "in case the reports will be generated"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_with_reports.yaml",
        settings={},
        log_level="INFO",
        type_of_process="train",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_train()
    mock_check_existence_of_source.assert_called_once_with("test_table")
    mock_gather_existed_columns.assert_called_once_with("test_table")
    mock_check_existence_of_source.assert_called_once_with("test_table")
    mock_check_existence_of_key_columns.assert_called_once_with("test_table")
    mock_check_existence_of_referenced_columns.assert_called_once_with("test_table")
    mock_check_completion_of_training.assert_not_called()
    mock_check_existence_of_destination.assert_not_called()
    mock_validate_fernet_key.assert_not_called()
    mock_validate_metadata.assert_called_once_with("test_table")
    mock_preprocess_data.assert_called_once_with(table_name="test_table")
    mock_train_table.assert_called_once()
    mock_infer_table.assert_called_once_with(
        table="test_table",
        metadata={
            "test_table": {
                "train_settings": {
                    "source": "./path/to/test_table.csv",
                    "epochs": 100,
                    "drop_null": False,
                    "reports": ["accuracy", "sample"],
                    "row_limit": 800,
                    "batch_size": 2000
                },
                "infer_settings": {
                    "destination": "./path/to/test_table_infer.csv",
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": ["accuracy"],
                    "batch_size": 200
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"]
                    }
                },
                "format": {}
            }
        },
        type_of_process="train",
        delta=0.49
    )
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(["test_table"], ["test_table"], True)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_infer_table")
def test_infer_tables_without_generation_reports(
    mock_infer_table,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_metadata,
    mock_validate_fernet_key,
    mock_check_access_to_input_data,
    mock_generate_reports,
    mock_collect_metrics_in_infer,
    rp_logger,
):
    """
    Test the '__infer_tables' method of the 'Worker' class
    in case the reports won't be generated
    """
    rp_logger.info(
        "Test the '__infer_tables' method of the 'Worker' class "
        "in case the reports won't be generated"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_without_reports.yaml",
        settings={
            "size": 300,
            "run_parallel": True,
            "random_seed": 3,
            "reports": ["accuracy"],
            "batch_size": 300
        },
        log_level="INFO",
        type_of_process="infer",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_infer()
    mock_gather_existed_columns.assert_not_called()
    mock_check_existence_of_source.assert_not_called()
    mock_check_existence_of_key_columns.assert_not_called()
    mock_check_existence_of_referenced_columns.assert_not_called()
    mock_check_completion_of_training.assert_called_once_with("test_table")
    mock_check_existence_of_destination.assert_called_once_with("test_table")
    mock_validate_metadata.assert_called_once_with("test_table")
    mock_infer_table.assert_called_once_with(
        table="test_table",
        metadata={
            "test_table": {
                "train_settings": {
                    "source": "./path/to/test_table.csv",
                    "epochs": 100,
                    "drop_null": False,
                    "reports": [],
                    "row_limit": 800,
                    "batch_size": 2000
                },
                "infer_settings": {
                    "destination": "./path/to/test_table_infer.csv",
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": [],
                    "batch_size": 200
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"]
                    }
                },
                "format": {}
            }
        },
        type_of_process="infer",
        delta=0.5
    )
    mock_generate_reports.assert_called_once()
    mock_validate_fernet_key.assert_not_called()
    mock_check_access_to_input_data.assert_not_called()
    mock_collect_metrics_in_infer.assert_called_once_with(["test_table"])
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Worker, "_generate_reports")
@patch.object(
    Worker, "_Worker__preprocess_data",
    return_value=(
            pd.DataFrame(
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
            ),
            {"fields": {}, "format": "CSV"})
)
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_infer_table")
@patch.object(Worker, "_train_table")
def test_train_tables_without_provided_fernet_key(
    mock_train_table,
    mock_infer_table,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_preprocess_data,
    mock_generate_reports,
    mock_collect_metrics_in_train,
    rp_logger,
):
    """
    Test the '__train_tables' method of the 'Worker' class
    in case the Fernet key was not provided
    """
    rp_logger.info(
        "Test the '__train_tables' method of the 'Worker' class "
        "in case the Fernet key was not provided"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_without_reports.yaml",
        settings={},
        log_level="INFO",
        type_of_process="train",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_train()
    mock_preprocess_data.assert_called_once_with(table_name="test_table")
    mock_gather_existed_columns.assert_called_once_with("test_table")
    mock_check_existence_of_source.assert_called_once_with("test_table")
    mock_check_existence_of_key_columns.assert_called_once_with("test_table")
    mock_check_existence_of_referenced_columns.assert_called_once_with("test_table")
    mock_check_completion_of_training.assert_not_called()
    mock_check_existence_of_destination.assert_not_called()
    mock_validate_fernet_key.assert_not_called()
    mock_validate_metadata.assert_called_once_with("test_table")
    mock_train_table.assert_called_once()
    mock_infer_table.assert_not_called()
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(["test_table"], ["test_table"], False)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Worker, "_generate_reports")
@patch.object(
    Worker, "_Worker__preprocess_data",
    return_value=(
            pd.DataFrame(
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
            ),
            {"fields": {}, "format": "CSV"})
)
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_infer_table")
@patch.object(Worker, "_train_table")
def test_train_tables_with_provided_fernet_key(
    mock_train_table,
    mock_infer_table,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_preprocess_data,
    mock_generate_reports,
    mock_collect_metrics_in_train,
    rp_logger,
):
    """
    Test the '__train_tables' method of the 'Worker' class
    in case the Fernet key was provided
    """
    rp_logger.info(
        "Test the '__train_tables' method of the 'Worker' class "
        "in case the Fernet key was provided"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_without_reports.yaml",
        settings={},
        log_level="INFO",
        type_of_process="train",
        encryption_settings=fetch_env_variables({"fernet_key": "FERNET_KEY"})
    )
    worker.launch_train()
    mock_preprocess_data.assert_called_once_with(table_name="test_table")
    mock_gather_existed_columns.assert_called_once_with("test_table")
    mock_check_existence_of_source.assert_called_once_with("test_table")
    mock_check_existence_of_key_columns.assert_called_once_with("test_table")
    mock_check_existence_of_referenced_columns.assert_called_once_with("test_table")
    mock_check_completion_of_training.assert_not_called()
    mock_check_existence_of_destination.assert_not_called()
    mock_validate_fernet_key.assert_called_once_with("test_table", FERNET_KEY)
    mock_validate_metadata.assert_called_once_with("test_table")
    mock_train_table.assert_called_once()
    mock_infer_table.assert_not_called()
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(["test_table"], ["test_table"], False)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables")
def test_launch_train_with_metadata_contained_several_fernet_keys(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_fernet_key,
    mock_validate_metadata,
    mock_generate_reports,
    mock_collect_metrics_in_train,
    rp_logger,
):
    """
    Test that 'launch_train' method calls all necessary methods
    by providing the metadata file of related tables
    in which several fernet keys are provided
    """
    rp_logger.info(
        "Test that 'launch_train' method calls all necessary methods "
        "by providing the metadata file of related tables "
        "in several fernet keys are provided"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                      "metadata_with_several_fernet_keys.yaml",
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "reports": ["accuracy", "sample"],
        },
        log_level="INFO",
        type_of_process="train",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        ["pk_test", "fk_test"],
        ["pk_test", "fk_test"],
        {
            "pk_test": {
                "train_settings": {
                    "source": "./path/to/pk_test.csv",
                    "epochs": 1,
                    "drop_null": False,
                    "row_limit": 800,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"]
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "reports": ["accuracy"]
                },
                "encryption": {
                    "fernet_key": "k64ntCKv3k7ihkNmbjN5cIlkRGxkPoHskJNcKB6bVuI="
                },
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"]
                    }
                },
                "format": {}
            },
            "fk_test": {
                "train_settings": {
                    "source": "./path/to/fk_test.csv",
                    "epochs": 5,
                    "drop_null": True,
                    "reports": ["accuracy", "sample"],
                    "row_limit": 600,
                    "batch_size": 1000
                },
                "infer_settings": {
                    "size": 90,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": []
                },
                "keys": {
                    "fk_id": {
                        "type": "FK",
                        "columns": ["Id"],
                        "references": {
                            "table": "pk_test",
                            "columns": ["Id"]
                        }
                    }
                },
                "encryption": {
                    "fernet_key": "VrToTpXdm35CNT3Tur3EGIa2OZ8bfjo-asHo_b-0DTY="
                },
                "format": {}
            }
        },
        {
            "pk_test": {
                "train_settings": {
                    "source": "./path/to/pk_test.csv",
                    "epochs": 1,
                    "drop_null": False,
                    "row_limit": 800,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"]
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "reports": ["accuracy"]
                },
                "encryption": {
                    "fernet_key": "k64ntCKv3k7ihkNmbjN5cIlkRGxkPoHskJNcKB6bVuI="
                },
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"]
                    }
                },
                "format": {}
            },
            "fk_test": {
                "train_settings": {
                    "source": "./path/to/fk_test.csv",
                    "epochs": 5,
                    "drop_null": True,
                    "reports": ["accuracy", "sample"],
                    "row_limit": 600,
                    "batch_size": 1000
                },
                "infer_settings": {
                    "size": 90,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": []
                },
                "keys": {
                    "fk_id": {
                        "type": "FK",
                        "columns": ["Id"],
                        "references": {
                            "table": "pk_test",
                            "columns": ["Id"]
                        }
                    }
                },
                "encryption": {
                    "fernet_key": "VrToTpXdm35CNT3Tur3EGIa2OZ8bfjo-asHo_b-0DTY="
                },
                "format": {}
            }
        },
        True
    )
    assert mock_gather_existed_columns.call_count == 2
    assert mock_check_existence_of_source.call_count == 2
    assert mock_check_existence_of_key_columns.call_count == 2
    assert mock_check_existence_of_referenced_columns.call_count == 2
    assert mock_validate_fernet_key.call_count == 2
    assert mock_validate_metadata.call_count == 2
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(
        ["pk_test", "fk_test"],
        ["pk_test", "fk_test"],
        True
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_launch_train_with_metadata_contained_non_existent_fernet_key(rp_logger, caplog):
    """
    Test that 'launch_train' method calls all necessary methods
    by providing the metadata file of related tables
    in which one of provided keys is non-existent
    """
    rp_logger.info(
        "Test that 'launch_train' method calls all necessary methods "
        "by providing the metadata file of related tables "
        "in which one of provided keys is non-existent"
    )
    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            worker = Worker(
                table_name=None,
                metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                              "metadata_with_one_nonexistent_fernet_key.yaml",
                settings={
                    "source": None,
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                log_level="INFO",
                type_of_process="train",
                encryption_settings=fetch_env_variables({"fernet_key": None})
            )
            worker.launch_train()
    assert (
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT' wasn't fetched. "
        "Please, check whether it is set correctly."
    ) in str(error.value)
    assert (
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT' wasn't fetched. "
        "Please, check whether it is set correctly."
    ) in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_launch_train_with_metadata_contained_all_non_existent_fernet_key(rp_logger, caplog):
    """
    Test that 'launch_train' method calls all necessary methods
    by providing the metadata file of related tables
    in which all provided keys are non-existent
    """
    rp_logger.info(
        "Test that 'launch_train' method calls all necessary methods "
        "by providing the metadata file of related tables "
        "in which all provided keys are non-existent"
    )
    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            worker = Worker(
                table_name=None,
                metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                              "metadata_with_all_nonexistent_fernet_keys.yaml",
                settings={
                    "source": None,
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                log_level="INFO",
                type_of_process="train",
                encryption_settings=fetch_env_variables({"fernet_key": None})
            )
            worker.launch_train()
    assert (
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT_1' wasn't fetched. "
        "Please, check whether it is set correctly. "
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT_2' wasn't fetched. "
        "Please, check whether it is set correctly."
    ) in str(error.value)
    assert (
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT_1' wasn't fetched. "
        "Please, check whether it is set correctly. "
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT_2' wasn't fetched. "
        "Please, check whether it is set correctly."
    ) in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_infer_table")
def test_infer_tables_with_generation_report_and_without_provided_fernet_key(
    mock_infer_table,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_metadata,
    mock_validate_fernet_key,
    mock_check_access_to_input_data,
    mock_generate_reports,
    mock_collect_metrics_in_infer,
    rp_logger,
):
    """
    Test the '__infer_tables' method of the 'Worker' class
    in case the reports should be generated, and the Fernet key was not provided
    """
    rp_logger.info(
        "Test the '__infer_tables' method of the 'Worker' class "
        "in case the reports will be generated, and the Fernet key was not provided"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_with_reports.yaml",
        settings={
            "size": 300,
            "run_parallel": True,
            "random_seed": 3,
            "reports": ["accuracy"],
            "batch_size": 300,
        },
        log_level="INFO",
        type_of_process="infer",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_infer()
    mock_gather_existed_columns.assert_not_called()
    mock_check_existence_of_source.assert_not_called()
    mock_check_existence_of_key_columns.assert_not_called()
    mock_check_existence_of_referenced_columns.assert_not_called()
    mock_check_completion_of_training.assert_called_once_with("test_table")
    mock_check_existence_of_destination.assert_called_once_with("test_table")
    mock_validate_metadata.assert_called_once_with("test_table")
    mock_infer_table.assert_called_once_with(
        table="test_table",
        metadata={
            "test_table": {
                "train_settings": {
                    "source": "./path/to/test_table.csv",
                    "epochs": 100,
                    "drop_null": False,
                    "reports": ["accuracy", "sample"],
                    "row_limit": 800,
                    "batch_size": 2000
                },
                "infer_settings": {
                    "destination": "./path/to/test_table_infer.csv",
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": ["accuracy"],
                    "batch_size": 200
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"]
                    }
                },
                "format": {}
            }
        },
        type_of_process="infer",
        delta=0.25
    )
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_infer.assert_called_once_with(["test_table"])
    mock_validate_fernet_key.assert_not_called()
    mock_check_access_to_input_data.assert_called_once_with("test_table")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_infer_table")
def test_infer_tables_with_generation_report_and_with_provided_fernet_key(
    mock_infer_table,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_metadata,
    mock_validate_fernet_key,
    mock_check_access_to_input_data,
    mock_generate_reports,
    mock_collect_metrics_in_infer,
    rp_logger,
):
    """
    Test the '__infer_tables' method of the 'Worker' class
    in case the reports should be generated, and the Fernet key was provided
    """
    rp_logger.info(
        "Test the '__infer_tables' method of the 'Worker' class "
        "in case the reports will be generated, and the Fernet key was provided"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_with_reports.yaml",
        settings={
            "size": 300,
            "run_parallel": True,
            "random_seed": 3,
            "reports": ["accuracy"],
            "batch_size": 300,
        },
        log_level="INFO",
        type_of_process="infer",
        encryption_settings=fetch_env_variables({"fernet_key": "FERNET_KEY"})
    )
    worker.launch_infer()
    mock_gather_existed_columns.assert_not_called()
    mock_check_existence_of_source.assert_not_called()
    mock_check_existence_of_key_columns.assert_not_called()
    mock_check_existence_of_referenced_columns.assert_not_called()
    mock_check_completion_of_training.assert_called_once_with("test_table")
    mock_check_existence_of_destination.assert_called_once_with("test_table")
    mock_validate_metadata.assert_called_once_with("test_table")
    mock_infer_table.assert_called_once_with(
        table="test_table",
        metadata={
            "test_table": {
                "train_settings": {
                    "source": "./path/to/test_table.csv",
                    "epochs": 100,
                    "drop_null": False,
                    "reports": ["accuracy", "sample"],
                    "row_limit": 800,
                    "batch_size": 2000
                },
                "infer_settings": {
                    "destination": "./path/to/test_table_infer.csv",
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": ["accuracy"],
                    "batch_size": 200
                },
                "encryption": {"fernet_key": FERNET_KEY},
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"]
                    }
                },
                "format": {}
            }
        },
        type_of_process="infer",
        delta=0.25
    )
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_infer.assert_called_once_with(["test_table"])
    mock_validate_fernet_key.assert_called_once_with("test_table", FERNET_KEY)
    mock_check_access_to_input_data.assert_called_once_with("test_table")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_infer_table")
def test_infer_tables_without_generation_report_and_with_provided_fernet_key(
    mock_infer_table,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_metadata,
    mock_validate_fernet_key,
    mock_check_access_to_input_data,
    mock_generate_reports,
    mock_collect_metrics_in_infer,
    rp_logger,
):
    """
    Test the '__infer_tables' method of the 'Worker' class
    in case the reports won't be generated, and the Fernet key was provided
    """
    rp_logger.info(
        "Test the '__infer_tables' method of the 'Worker' class "
        "in case the reports won't be generated, and the Fernet key was provided"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_without_reports.yaml",
        settings={
            "size": 300,
            "run_parallel": True,
            "random_seed": 3,
            "reports": [],
            "batch_size": 300,
        },
        log_level="INFO",
        type_of_process="infer",
        encryption_settings=fetch_env_variables({"fernet_key": "FERNET_KEY"})
    )
    worker.launch_infer()
    mock_gather_existed_columns.assert_not_called()
    mock_check_existence_of_source.assert_not_called()
    mock_check_existence_of_key_columns.assert_not_called()
    mock_check_existence_of_referenced_columns.assert_not_called()
    mock_check_completion_of_training.assert_called_once_with("test_table")
    mock_check_existence_of_destination.assert_called_once_with("test_table")
    mock_validate_metadata.assert_called_once_with("test_table")
    mock_infer_table.assert_called_once_with(
        table="test_table",
        metadata={
            "test_table": {
                "train_settings": {
                    "source": "./path/to/test_table.csv",
                    "epochs": 100,
                    "drop_null": False,
                    "reports": [],
                    "row_limit": 800,
                    "batch_size": 2000
                },
                "infer_settings": {
                    "destination": "./path/to/test_table_infer.csv",
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": [],
                    "batch_size": 200
                },
                "encryption": {"fernet_key": FERNET_KEY},
                "format": {},
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"],
                    }
                }
            }
        },
        type_of_process="infer",
        delta=0.5
    )
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_infer.assert_called_once_with(["test_table"])
    mock_validate_fernet_key.assert_not_called()
    mock_check_access_to_input_data.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_validate_fernet_key")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_infer_table")
def test_infer_tables_without_generation_report_and_without_provided_fernet_key(
    mock_infer_table,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_metadata,
    mock_validate_fernet_key,
    mock_check_access_to_input_data,
    mock_generate_reports,
    mock_collect_metrics_in_infer,
    rp_logger,
):
    """
    Test the '__infer_tables' method of the 'Worker' class
    in case the reports won't be generated, and the Fernet key wasn't provided
    """
    rp_logger.info(
        "Test the '__infer_tables' method of the 'Worker' class "
        "in case the reports won't be generated, and the Fernet key wasn't provided"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_without_reports.yaml",
        settings={
            "size": 300,
            "run_parallel": True,
            "random_seed": 3,
            "reports": [],
            "batch_size": 300,
        },
        log_level="INFO",
        type_of_process="infer",
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_infer()
    mock_gather_existed_columns.assert_not_called()
    mock_check_existence_of_source.assert_not_called()
    mock_check_existence_of_key_columns.assert_not_called()
    mock_check_existence_of_referenced_columns.assert_not_called()
    mock_check_completion_of_training.assert_called_once_with("test_table")
    mock_check_existence_of_destination.assert_called_once_with("test_table")
    mock_validate_metadata.assert_called_once_with("test_table")
    mock_infer_table.assert_called_once_with(
        table="test_table",
        metadata={
            "test_table": {
                "train_settings": {
                    "source": "./path/to/test_table.csv",
                    "epochs": 100,
                    "drop_null": False,
                    "reports": [],
                    "row_limit": 800,
                    "batch_size": 2000
                },
                "infer_settings": {
                    "destination": "./path/to/test_table_infer.csv",
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": [],
                    "batch_size": 200
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"]
                    }
                },
                "format": {}
            }
        },
        type_of_process="infer",
        delta=0.5
    )
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_infer.assert_called_once_with(["test_table"])
    mock_validate_fernet_key.assert_not_called()
    mock_check_access_to_input_data.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables")
def test_launch_train_with_absent_metadata_and_callback_loader(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_metadata,
    mock_generate_reports,
    mock_collect_metrics_in_train,
    rp_logger
):
    """
    Test the initialization of 'Worker' class
    with the absent metadata and provided callback function
    during the training process
    """
    rp_logger.info(
        "Test the initialization of the instance of 'Worker' class "
        "with the absent metadata and provided callback function during the training process"
    )
    worker = Worker(
        table_name="table",
        metadata_path=None,
        settings={
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "reports": ["accuracy", "sample"],
        },
        log_level="INFO",
        type_of_process="train",
        loader=MagicMock(),
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    assert worker.metadata == {
        "table": {
            "train_settings": {
                "source": None,
                "batch_size": 1000,
                "drop_null": True,
                "epochs": 20,
                "reports": ["accuracy", "sample"],
                "row_limit": 1000,
            },
            "infer_settings": {},
            "encryption": {"fernet_key": None},
            "keys": {},
            "format": {}
        }
    }
    worker.launch_train()
    mock_train_tables.assert_called_with(
        ["table"],
        ["table"],
        {
            "table": {
                "train_settings": {
                    "source": None,
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"]
                },
                "infer_settings": {},
                "encryption": {"fernet_key": None},
                "keys": {},
                "format": {}
            }
        },
        {
            "table": {
                "train_settings": {
                    "source": None,
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"]
                },
                "infer_settings": {},
                "encryption": {"fernet_key": None},
                "keys": {},
                "format": {}
            }
        },
        True
    )
    mock_gather_existed_columns.assert_not_called()
    mock_check_existence_of_source.assert_not_called()
    mock_check_existence_of_key_columns.assert_not_called()
    mock_check_existence_of_referenced_columns.assert_not_called()
    mock_validate_metadata.assert_called_once()
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(
        ["table"],
        ["table"],
        True
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_launch_infer_with_metadata_contained_non_existent_fernet_key(rp_logger, caplog):
    """
    Test that 'launch_infer' method calls all necessary methods
    by providing the metadata file of related tables
    in which one of provided keys is non-existent
    """
    rp_logger.info(
        "Test that 'launch_infer' method calls all necessary methods "
        "by providing the metadata file of related tables "
        "in which one of provided keys is non-existent"
    )
    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            worker = Worker(
                table_name=None,
                metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                              "metadata_with_one_nonexistent_fernet_key.yaml",
                settings={
                    "size": 300,
                    "run_parallel": True,
                    "random_seed": 3,
                    "reports": ["accuracy"],
                    "batch_size": 300,
                },
                log_level="INFO",
                type_of_process="infer",
                encryption_settings=fetch_env_variables({"fernet_key": None})
            )
            worker.launch_infer()
    assert (
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT' wasn't fetched. "
        "Please, check whether it is set correctly."
    ) in str(error.value)
    assert (
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT' wasn't fetched. "
        "Please, check whether it is set correctly."
    ) in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_launch_infer_with_metadata_contained_all_non_existent_fernet_key(rp_logger, caplog):
    """
    Test that 'launch_infer' method calls all necessary methods
    by providing the metadata file of related tables
    in which all provided keys are non-existent
    """
    rp_logger.info(
        "Test that 'launch_infer' method calls all necessary methods "
        "by providing the metadata file of related tables "
        "in which all provided keys are non-existent"
    )
    with pytest.raises(ValueError) as error:
        with caplog.at_level("ERROR"):
            worker = Worker(
                table_name=None,
                metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                              "metadata_with_all_nonexistent_fernet_keys.yaml",
                settings={
                    "size": 300,
                    "run_parallel": True,
                    "random_seed": 3,
                    "reports": ["accuracy"],
                    "batch_size": 300,
                },
                log_level="INFO",
                type_of_process="infer",
                encryption_settings=fetch_env_variables({"fernet_key": None})
            )
            worker.launch_infer()
    assert (
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT_1' wasn't fetched. "
        "Please, check whether it is set correctly. "
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT_2' wasn't fetched. "
        "Please, check whether it is set correctly."
    ) in str(error.value)
    assert (
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT_1' wasn't fetched. "
        "Please, check whether it is set correctly. "
        "The value of the environment variable 'FERNET_KEY_NONEXISTENT_2' wasn't fetched. "
        "Please, check whether it is set correctly."
    ) in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables")
def test_launch_train_with_metadata_without_source_paths_and_loader(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_metadata,
    mock_generate_reports,
    mock_collect_metrics_in_train,
    rp_logger,
):
    """
    Test that 'launch_train' method calls all necessary methods
    in case the metadata file of related tables was provided,
    and as the data will be loaded by using the callback loader,
    it doesn't contain source paths for tables
    """
    rp_logger.info(
        "Test that 'launch_train' method calls all necessary methods "
        "in case the metadata file of related tables was provided, "
        "and as the data will be loaded by using the callback loader, "
        "it doesn't contain source paths for tables"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_without_sources.yaml",
        settings={
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "reports": ["accuracy", "sample"],
        },
        log_level="INFO",
        type_of_process="train",
        loader=MagicMock(),
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        ["pk_test", "fk_test"],
        ["pk_test", "fk_test"],
        {
            "pk_test": {
                "train_settings": {
                    "epochs": 1,
                    "drop_null": False,
                    "row_limit": 800,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"]
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "reports": ["accuracy"]
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"]
                    }
                },
                "format": {}
            },
            "fk_test": {
                "train_settings": {
                    "epochs": 5,
                    "drop_null": True,
                    "reports": ["accuracy", "sample"],
                    "row_limit": 600,
                    "batch_size": 1000
                },
                "infer_settings": {
                    "size": 90,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": []
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "fk_id": {
                        "type": "FK",
                        "columns": ["Id"],
                        "references": {
                            "table": "pk_test",
                            "columns": ["Id"]
                        }
                    }
                },
                "format": {}
            }
        },
        {
            "pk_test": {
                "train_settings": {
                    "epochs": 1,
                    "drop_null": False,
                    "row_limit": 800,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"]
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "reports": ["accuracy"]
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"]
                    }
                },
                "format": {}
            },
            "fk_test": {
                "train_settings": {
                    "epochs": 5,
                    "drop_null": True,
                    "reports": ["accuracy", "sample"],
                    "row_limit": 600,
                    "batch_size": 1000
                },
                "infer_settings": {
                    "size": 90,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": []
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "fk_id": {
                        "type": "FK",
                        "columns": ["Id"],
                        "references": {
                            "table": "pk_test",
                            "columns": ["Id"]
                        }
                    }
                },
                "format": {}
            }
        },
        True
    )
    mock_gather_existed_columns.assert_not_called()
    mock_check_existence_of_source.assert_not_called()
    mock_check_existence_of_key_columns.assert_not_called()
    mock_check_existence_of_referenced_columns.assert_not_called()
    assert mock_validate_metadata.call_count == 2
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(
        ["pk_test", "fk_test"],
        ["pk_test", "fk_test"],
        True
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Worker, "_generate_reports")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables")
def test_launch_train_with_metadata_without_train_settings_and_loader(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_metadata,
    mock_generate_reports,
    mock_collect_metrics_in_train,
    rp_logger,
):
    """
    Test that 'launch_train' method calls all necessary methods
    in case the metadata file of related tables was provided,
    and as the data will be loaded by using the callback loader,
    it might not contain training settings for tables
    """
    rp_logger.info(
        "Test that 'launch_train' method calls all necessary methods "
        "in case the metadata file of related tables was provided, "
        "and as the data will be loaded by using the callback loader, "
        "it might not contain training settings for tables"
    )
    worker = Worker(
        table_name=None,
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/metadata_without_train_settings.yaml",
        settings={
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "reports": ["accuracy", "sample"],
        },
        log_level="INFO",
        type_of_process="train",
        loader=MagicMock(),
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        ["pk_test", "fk_test"],
        ["pk_test", "fk_test"],
        {
            "pk_test": {
                "train_settings": {
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"]
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "reports": ["accuracy"]
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"]
                    }
                },
                "format": {}
            },
            "fk_test": {
                "train_settings": {
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"]
                },
                "infer_settings": {
                    "size": 90,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": []
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "fk_id": {
                        "type": "FK",
                        "columns": ["Id"],
                        "references": {
                            "table": "pk_test",
                            "columns": ["Id"]
                        }
                    }
                },
                "format": {}
            }
        },
        {
            "pk_test": {
                "train_settings": {
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"]
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "reports": ["accuracy"]
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"]
                    }
                },
                "format": {}
            },
            "fk_test": {
                "train_settings": {
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"]
                },
                "infer_settings": {
                    "size": 90,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": []
                },
                "encryption": {"fernet_key": None},
                "keys": {
                    "fk_id": {
                        "type": "FK",
                        "columns": ["Id"],
                        "references": {
                            "table": "pk_test",
                            "columns": ["Id"]
                        }
                    }
                },
                "format": {}
            }
        },
        True
    )
    mock_gather_existed_columns.assert_not_called()
    mock_check_existence_of_source.assert_not_called()
    mock_check_existence_of_key_columns.assert_not_called()
    mock_check_existence_of_referenced_columns.assert_not_called()
    assert mock_validate_metadata.call_count == 2
    mock_collect_metrics_in_train.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(
        ["pk_test", "fk_test"],
        ["pk_test", "fk_test"],
        True
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Worker, "_generate_reports")
@patch.object(Worker, "_infer_table")
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_validate_metadata")
def test_launch_infer_of_pretrained_table(
    mock_validate_metadata,
    mock_check_existence_of_destination,
    mock_check_completion_of_training,
    mock_check_access_to_input_data,
    mock_infer_table,
    mock_generate_reports,
    mock_collect_metrics_in_infer,
    rp_logger,
):
    """
    Test that the inference process has been launched
    after the training process of the table has been finished
    """
    rp_logger.info(
        "Test that the inference process has been launched "
        "after the training process of the table has been finished"
    )
    worker = Worker(
        table_name="table",
        metadata_path=None,
        settings={
            "size": 300,
            "run_parallel": True,
            "random_seed": 3,
            "reports": ["accuracy"],
            "batch_size": 300,
        },
        log_level="INFO",
        type_of_process="infer",
        loader=None,
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    metadata = {
        "table": {
            "train_settings": {
                "source": None
            },
            "infer_settings": {
                "size": 300,
                "run_parallel": True,
                "random_seed": 3,
                "reports": ["accuracy"],
                "batch_size": 300
            },
            "encryption": {"fernet_key": None},
            "keys": {},
            "format": {}
        }
    }
    worker.launch_infer()
    mock_validate_metadata.assert_called_once_with("table")
    mock_check_existence_of_destination.assert_called_once_with("table")
    mock_check_completion_of_training.assert_called_once_with("table")
    mock_check_access_to_input_data.assert_called_once_with("table")
    mock_infer_table.assert_called_once_with(
        table="table",
        metadata=metadata,
        type_of_process="infer",
        delta=0.25
    )
    mock_generate_reports.assert_called_once()
    mock_collect_metrics_in_infer.assert_called_once_with(["table"])


@patch.object(Worker, "_infer_table")
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_validate_metadata")
def test_launch_infer_of_not_pretrained_table_and_absent_success_file(
    mock_validate_metadata,
    mock_check_existence_of_destination,
    mock_check_access_to_input_data,
    mock_infer_table,
    caplog,
    rp_logger,
):
    """
    Test that the inference process hasn't been started
    in case the training process of the table hasn't been finished,
    and the appropriate success file 'message.success' is absent
    """
    rp_logger.info(
        "Test that the inference process hasn't been started "
        "in case the training process of the table hasn't been finished, "
        "and the appropriate success file 'message.success' is absent"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            worker = Worker(
                table_name="table",
                metadata_path=None,
                settings={
                    "size": 300,
                    "run_parallel": True,
                    "random_seed": 3,
                    "reports": ["accuracy"],
                    "batch_size": 300,
                },
                log_level="INFO",
                type_of_process="infer",
                loader=None,
                encryption_settings=fetch_env_variables({"fernet_key": None})
            )
            worker.launch_infer()
            message = (
                "The training of the table - 'table' hasn't been completed. "
                "Please, retrain the table."
            )
            assert message in str(error.value)
            assert message in caplog.text
    mock_check_existence_of_destination.assert_called_once_with("table")
    mock_validate_metadata.assert_called_once_with("table")
    mock_check_access_to_input_data.assert_not_called()
    mock_infer_table.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_infer_table")
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_validate_metadata")
def test_launch_infer_of_not_pretrained_table_and_success_file_with_wrong_content(
    mock_validate_metadata,
    mock_check_existence_of_destination,
    mock_check_access_to_input_data,
    mock_infer_table,
    test_success_file,
    caplog,
    rp_logger,
):
    """
    Test that the inference process hasn't been started
    in case the training process of the table hasn't been finished,
    and the appropriate success file 'train_message.success' is present,
    but the content of the file doesn't correspond to finished training process
    """
    rp_logger.info(
        "Test that the inference process hasn't been started "
        "in case the training process of the table hasn't been finished, "
        "and the appropriate success file 'train_message.success' is present, "
        "but the content of the file doesn't correspond to finished training process"
    )
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            worker = Worker(
                table_name="table",
                metadata_path=None,
                settings={
                    "size": 300,
                    "run_parallel": True,
                    "random_seed": 3,
                    "reports": ["accuracy"],
                    "batch_size": 300,
                },
                log_level="INFO",
                type_of_process="infer",
                loader=None,
                encryption_settings=fetch_env_variables({"fernet_key": None})
            )
            worker.launch_infer()
            message = (
                "The training of the table - 'table' hasn't been completed. "
                "Please, retrain the table."
            )
            assert message in str(error.message)
            assert message in caplog.text
    mock_check_existence_of_destination.assert_called_once_with("table")
    mock_check_access_to_input_data.assert_not_called()
    mock_infer_table.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("metadata, type_of_process, expected_result", [
    (
        {
            "test_table": {
                "train_settings": {
                    "source": "./path/to/test_table.csv",
                    "reports": ["accuracy", "sample"],
                }
            }
        }, "train", True
    ),
    (
        {
            "test_table": {
                "train_settings": {
                    "source": "./path/to/test_table.csv",
                    "reports": ["sample"],
                }
            }
        }, "train", False
    ),
    (
            {
                "test_table": {
                    "train_settings": {
                        "source": "./path/to/test_table.csv",
                        "reports": [],
                    }
                }
            }, "train", False
    ),
    (
            {
                "test_table": {
                    "train_settings": {
                        "source": "./path/to/test_table.csv",
                        "reports": ["accuracy", "sample"],
                    }
                }
            }, "infer", False
    ),
    (
            {
                "test_table": {
                    "train_settings": {
                        "source": "./path/to/test_table.csv",
                        "reports": ["accuracy", "sample"],
                    },
                    "infer_settings": {
                        "reports": ["accuracy"],
                    }
                }
            }, "infer", True
    ),
    (
            {
                "test_table": {
                    "train_settings": {
                        "source": "./path/to/test_table.csv",
                        "reports": ["accuracy", "sample"],
                    },
                    "infer_settings": {
                        "reports": [],
                    }
                }
            }, "infer", False
    ),

])
@patch.object(Worker, "__attrs_post_init__")
def test_should_generate_reports(
    mock_post_init,
    metadata,
    type_of_process,
    expected_result,
    rp_logger
):
    rp_logger.info("Test the method '_should_generate_data' of the 'Worker' class")
    worker = Worker(
        table_name="table",
        metadata_path=None,
        settings={},
        log_level="INFO",
        type_of_process=type_of_process,
        loader=None,
        encryption_settings=fetch_env_variables({"fernet_key": None})
    )
    worker._should_generate_data(metadata, type_of_process) == expected_result
    rp_logger.info(SUCCESSFUL_MESSAGE)
