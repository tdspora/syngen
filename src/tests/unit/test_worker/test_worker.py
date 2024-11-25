from unittest.mock import patch, MagicMock

import pytest

from syngen.ml.worker import Worker
from syngen.ml.config import Validator

from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME


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
        table_name="test_table",
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
    )
    assert worker.metadata == {
        "test_table": {
            "train_settings": {"source": None},
            "infer_settings": {
                "size": 100,
                "run_parallel": False,
                "batch_size": 100,
                "reports": [],
                "random_seed": 1,
            },
            "keys": {},
            "format": {}
        }
    }
    mock_validator_run.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "run")
def test_init_worker_with_metadata_path(rp_logger):
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
    )
    assert worker.metadata == {
        "global": {},
        "test_table": {
            "train_settings": {
                "source": "./path/to/test_table.csv",
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
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["Id"]
                }
            },
            "format": {}
        },
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "run")
def test_init_worker_with_empty_settings_in_metadata_in_train_process(rp_logger):
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
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                      "metadata_with_empty_settings.yaml",
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
    )
    assert worker.metadata == {
        "global": {},
        "test_table": {
            "train_settings": {
                "source": "./path/to/test_table.csv",
                "epochs": 20,
                "drop_null": True,
                "reports": ["accuracy", "sample"],
                "row_limit": 1000,
                "batch_size": 1000,
            },
            "infer_settings": {},
            "keys": {},
            "format": {}
        },
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "run")
def test_init_worker_with_empty_settings_in_metadata_in_infer_process(rp_logger):
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
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                      "metadata_with_empty_settings.yaml",
        settings={
            "size": 200,
            "run_parallel": False,
            "batch_size": 200,
            "reports": ["accuracy"],
            "random_seed": 5,
        },
        log_level="INFO",
        type_of_process="infer",
    )
    assert worker.metadata == {
        "global": {},
        "test_table": {
            "train_settings": {
                "source": "./path/to/test_table.csv"
            },
            "infer_settings": {
                "size": 200,
                "run_parallel": False,
                "batch_size": 200,
                "random_seed": 5,
                "reports": ["accuracy"]
            },
            "keys": {},
            "format": {}
        },
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "run")
def test_init_worker_for_training_with_metadata_with_global_settings(rp_logger):
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
    )
    assert worker.metadata == {
        "global": {
            "train_settings": {
                "drop_null": True,
                "epochs": 5,
                "row_limit": 500
            },
            "infer_settings": {
                "reports": ["accuracy"],
                "run_parallel": True,
                "size": 1000,
            },
        },
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
            "format": {}
        },
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "run")
def test_init_worker_for_inference_with_metadata_with_global_settings(rp_logger):
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
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                      "metadata_with_global_settings.yaml",
        settings={
            "size": 200,
            "run_parallel": False,
            "batch_size": 200,
            "reports": [],
            "random_seed": 5,
        },
        log_level="INFO",
        type_of_process="infer",
    )
    assert worker.metadata == {
        "global": {
            "train_settings": {
                "drop_null": True,
                "epochs": 5,
                "row_limit": 500
            },
            "infer_settings": {
                "reports": ["accuracy"],
                "run_parallel": True,
                "size": 1000,
            },
        },
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
            "format": {}
        },
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables", return_value=None)
def test_launch_train_with_metadata(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_metadata,
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
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        ["test_table"],
        ["test_table"],
        {
            "test_table": {
                "train_settings": {
                    "source": "./path/to/test_table.csv",
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
                "keys": {"pk_id": {"type": "PK", "columns": ["Id"]}},
                "format": {}
            }
        },
        {
            "test_table": {
                "train_settings": {
                    "source": "./path/to/test_table.csv",
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
                "keys": {"pk_id": {"type": "PK", "columns": ["Id"]}},
                "format": {}
            }
        },
        False
    )
    mock_gather_existed_columns.assert_called_once()
    mock_check_existence_of_source.assert_called_once()
    mock_check_existence_of_key_columns.assert_called_once()
    mock_check_existence_of_referenced_columns.assert_called_once()
    mock_validate_metadata.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(
        ["test_table"],
        ["test_table"],
        False
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables", return_value=None)
def test_launch_train_with_metadata_of_related_tables(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_metadata,
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
    assert mock_validate_metadata.call_count == 2
    mock_collect_metrics_in_train.assert_called_once_with(
        ["pk_test", "fk_test"],
        ["pk_test", "fk_test"],
        True
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables", return_value=None)
def test_launch_train_with_metadata_of_related_tables_with_diff_keys(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_metadata,
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
    mock_collect_metrics_in_train.assert_called_once_with(
        ["tdm_models", "tdm_clusters"],
        ["tdm_clusters", "tdm_models_pk", "tdm_models_fk"],
        True
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables", return_value=None)
def test_launch_train_without_metadata(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_metadata,
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
        table_name="test_table",
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
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        ["test_table"],
        ["test_table"],
        {
            "test_table": {
                "train_settings": {
                    "source": "./path/to/source.csv",
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "infer_settings": {},
                "keys": {},
                "format": {}
            }
        },
        {
            "test_table": {
                "train_settings": {
                    "source": "./path/to/source.csv",
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"],
                },
                "infer_settings": {},
                "keys": {},
                "format": {}
            }
        },
        True
    )
    mock_gather_existed_columns.assert_called_once()
    mock_check_existence_of_source.assert_called_once()
    mock_check_existence_of_key_columns.assert_called_once()
    mock_check_existence_of_referenced_columns.assert_called_once()
    mock_validate_metadata.assert_called_once()
    mock_collect_metrics_in_train.assert_called_once_with(
        ["test_table"],
        ["test_table"],
        True
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables", return_value=None)
def test_launch_train_with_metadata_contained_global_settings(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_metadata,
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
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                      "metadata_with_global_settings.yaml",
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
    mock_collect_metrics_in_train.assert_called_once_with(
        ["pk_test", "fk_test"],
        ["pk_test", "fk_test"],
        True
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Worker, "_Worker__infer_tables", return_value=None)
def test_launch_infer_with_metadata(
    mock_infer_tables,
    mock_check_existence_of_destination,
    mock_validate_metadata,
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
    )
    worker.launch_infer()
    mock_infer_tables.assert_called_once_with(
        ["test_table"],
        {
            "test_table": {
                "train_settings": {
                    "source": "./path/to/test_table.csv",
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
                "keys": {"pk_id": {"type": "PK", "columns": ["Id"]}},
                "format": {}
            }
        },
        0.25,
        type_of_process="infer"
    )
    mock_check_existence_of_destination.assert_called_once()
    mock_validate_metadata.assert_called_once()
    mock_collect_metrics_in_infer.assert_called_once_with(["test_table"])
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Worker, "_Worker__infer_tables", return_value=None)
def test_launch_infer_with_metadata_of_related_tables(
    mock_infer_tables,
    mock_check_existence_of_destination,
    mock_validate_metadata,
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
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                      "metadata_of_related_tables.yaml",
        settings={
            "size": 300,
            "run_parallel": True,
            "reports": ["accuracy"],
            "batch_size": 200,
            "random_seed": 1,
        },
        log_level="INFO",
        type_of_process="infer",
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
    assert mock_check_existence_of_destination.call_count == 2
    assert mock_validate_metadata.call_count == 2
    mock_collect_metrics_in_infer.assert_called_once_with(["pk_test", "fk_test"])
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Worker, "_Worker__infer_tables", return_value=None)
def test_launch_infer_with_metadata_of_related_tables_with_diff_keys(
    mock_infer_tables,
    mock_check_existence_of_destination,
    mock_validate_metadata,
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
                "format": {}
            },
        },
        0.08333333333333333,
        type_of_process="infer"
    )
    assert mock_check_existence_of_destination.call_count == 2
    assert mock_validate_metadata.call_count == 2
    mock_collect_metrics_in_infer.assert_called_once_with(
        ["tdm_clusters", "tdm_models_pk", "tdm_models_fk"]
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Worker, "_Worker__infer_tables", return_value=None)
def test_launch_infer_without_metadata(
    mock_infer_tables,
    mock_check_existence_of_destination,
    mock_validate_metadata,
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
        table_name="test_table",
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
    )
    worker.launch_infer()
    mock_infer_tables.assert_called_once_with(
        ["test_table"],
        {
            "test_table": {
                "train_settings": {"source": None},
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "reports": ["accuracy"],
                    "batch_size": 200,
                },
                "keys": {},
                "format": {}
            }
        },
        0.25,
        type_of_process="infer"
    )
    mock_check_existence_of_destination.assert_called_once()
    mock_validate_metadata.assert_called_once()
    mock_collect_metrics_in_infer.assert_called_once_with(["test_table"])
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Worker, "_Worker__infer_tables", return_value=None)
def test_launch_infer_with_metadata_contained_global_settings(
    mock_infer_tables,
    mock_check_existence_of_destination,
    mock_validate_metadata,
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
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                      "metadata_with_global_settings.yaml",
        settings={
            "size": 300,
            "run_parallel": True,
            "random_seed": 3,
            "reports": ["accuracy"],
            "batch_size": 300,
        },
        log_level="INFO",
        type_of_process="infer",
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
                "format": {}
            },
        },
        0.125,
        type_of_process="infer",
    )
    assert mock_check_existence_of_destination.call_count == 2
    assert mock_validate_metadata.call_count == 2
    mock_collect_metrics_infer.assert_called_once_with(["pk_test", "fk_test"])
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables", return_value=None)
def test_init_worker_for_training_process_with_absent_metadata_and_callback_loader(
        mock_train_tables,
        mock_gather_existed_columns,
        mock_check_existence_of_source,
        mock_check_existence_of_key_columns,
        mock_check_existence_of_referenced_columns,
        mock_validate_metadata,
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
        table_name="test_table",
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
        loader=MagicMock()
    )
    assert worker.metadata == {
        "test_table": {
            "train_settings": {
                "source": None,
                "batch_size": 1000,
                "drop_null": True,
                "epochs": 20,
                "reports": ["accuracy", "sample"],
                "row_limit": 1000,
            },
            "infer_settings": {},
            "keys": {},
            "format": {}
        }
    }
    worker.launch_train()
    mock_train_tables.assert_called_with(
        ["test_table"],
        ["test_table"],
        {
            "test_table": {
                "train_settings": {
                    "source": None,
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"]
                },
                "infer_settings": {},
                "keys": {},
                "format": {}
            }
        },
        {
            "test_table": {
                "train_settings": {
                    "source": None,
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "reports": ["accuracy", "sample"]
                },
                "infer_settings": {},
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
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables", return_value=None)
def test_launch_train_with_metadata_without_source_paths(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_metadata,
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
        loader=MagicMock()
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
    mock_collect_metrics_in_train.assert_called_once_with(
        ["pk_test", "fk_test"],
        ["pk_test", "fk_test"],
        True
    )
    mock_check_existence_of_source.assert_not_called()
    mock_check_existence_of_key_columns.assert_not_called()
    mock_check_existence_of_referenced_columns.assert_not_called()
    assert mock_validate_metadata.call_count == 2
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_collect_metrics_in_train")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_check_existence_of_source")
@patch.object(Validator, "_gather_existed_columns")
@patch.object(Worker, "_Worker__train_tables", return_value=None)
def test_launch_train_with_metadata_without_train_settings(
    mock_train_tables,
    mock_gather_existed_columns,
    mock_check_existence_of_source,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_metadata,
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
        metadata_path=f"{DIR_NAME}/unit/test_worker/fixtures/"
                      "metadata_without_train_settings.yaml",
        settings={
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "reports": ["accuracy", "sample"],
        },
        log_level="INFO",
        type_of_process="train",
        loader=MagicMock()
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
    mock_collect_metrics_in_train.assert_called_once_with(
        ["pk_test", "fk_test"],
        ["pk_test", "fk_test"],
        True
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)



@patch.object(Worker, "_generate_reports")
@patch.object(Worker, "_check_completion_of_training", return_value=None)
@patch.object(Worker, "_infer_table")
@patch.object(Worker, "_collect_metrics_in_infer")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_destination")
def test_launch_infer_pretrained_table(
    mock_check_existence_of_destination,
    mock_validate_metadata,
    mock_collect_metrics_in_infer,
    mock_infer_table,
    mock_check_completion_of_training,
    mock_generate_reports,
    rp_logger,
):
    """
    Test that the inference process has been launched
    if the training process of the table has been finished
    """
    rp_logger.info(
        "Test that the inference process has been launched "
        "if the training process of the table has been finished"
    )
    worker = Worker(
        table_name="test_table",
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
        loader=None
    )
    worker.launch_infer()
    mock_check_existence_of_destination.assert_called_once()
    mock_validate_metadata.assert_called_once_with("test_table")
    mock_infer_table.assert_called_once_with(
        table="test_table",
        metadata={
            "test_table": {
                "train_settings": {
                    "source": None
                },
                'infer_settings': {
                    "size": 300,
                    "run_parallel": True,
                    "random_seed": 3,
                    "reports": ["accuracy"],
                    "batch_size": 300
                },
                "keys": {},
                "format": {}
            }
        },
        type_of_process="infer",
        delta=0.25
    )
    mock_collect_metrics_in_infer.assert_called_once_with(["test_table"])
    mock_generate_reports.assert_called_once()


@patch.object(Worker, "_infer_table")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_destination")
def test_launch_infer_not_pretrained_table(
    mock_check_existence_of_destination,
    mock_validate_metadata,
    mock_collect_metrics_in_infer,
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
    worker = Worker(
        table_name="test_table",
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
        loader=None
    )
    with pytest.raises(FileNotFoundError):
        with caplog.at_level("ERROR"):
            worker.launch_infer()
            assert (
                "The training of the table - 'test-table' hasn't been completed. "
                "Please, retrain the table."
                in caplog.text
            )
    mock_infer_table.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_infer_table")
@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_destination")
def test_launch_infer_not_pretrained_table(
    mock_check_existence_of_destination,
    mock_validate_metadata,
    mock_infer_table,
    test_success_file,
    caplog,
    rp_logger,
):
    """
    Test that the inference process hasn't been started
    in case the training process of the table hasn't been finished,
    and the appropriate success file 'message.success' is present,
    but the content of the file doesn't correspond to finished training process
    """
    rp_logger.info(
        "Test that the inference process hasn't been started "
        "in case the training process of the table hasn't been finished, " 
        "and the appropriate success file 'message.success' is present, "
        "but the content of the file doesn't correspond to finished training process"
    )
    worker = Worker(
        table_name="test_table",
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
        loader=None
    )
    with pytest.raises(ValueError):
        with caplog.at_level("ERROR"):
            worker.launch_infer()
            assert (
                "The training of the table - 'test-table' hasn't been completed. "
                "Please, retrain the table."
                in caplog.text
            )
    mock_infer_table.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)

