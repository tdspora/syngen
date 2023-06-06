from unittest.mock import patch

from syngen.ml.worker import Worker
from syngen.ml.strategies.strategies import TrainStrategy, InferStrategy

from tests.conftest import SUCCESSFUL_MESSAGE


def test_init_worker_for_training_process_with_absent_metadata(rp_logger):
    """
    Test the initialization of 'Worker' class with absent metadata
    during a training process
    """
    rp_logger.info("Test the initialization of the instance of 'Worker' class "
                   "with the absent metadata during a training process")
    worker = Worker(
        table_name="test_table",
        metadata_path=None,
        settings={
            "source": "path/to/source",
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "print_report": True
        },
        log_level="INFO",
        type="train"
    )
    assert isinstance(worker.train_strategy, TrainStrategy) is True
    assert isinstance(worker.infer_strategy, InferStrategy) is True
    assert worker.metadata =={
        "test_table": {
            "source": "path/to/source",
            "train_settings": {
                "batch_size": 1000,
                "drop_null": True,
                "epochs": 20,
                "print_report": True,
                "row_limit": 1000
            },
            "infer_settings": {},
            "keys": {}
        }
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_init_worker_for_infer_process_with_absent_metadata(rp_logger):
    """
    Test the initialization of 'Worker' class with absent metadata
    during an inference process
    """
    rp_logger.info("Test the initialization of the instance of 'Worker' class "
                   "with the absent metadata during an inference process")
    worker = Worker(
        table_name="test_table",
        metadata_path=None,
        settings={
            "size": 100,
            "run_parallel": False,
            "batch_size": 100,
            "print_report": False,
            "random_seed": 1
        },
        log_level="INFO",
        type="infer"
    )
    assert isinstance(worker.train_strategy, TrainStrategy) is True
    assert isinstance(worker.infer_strategy, InferStrategy) is True
    assert worker.metadata =={
        "test_table": {
            "source": None,
            "train_settings": {},
            "infer_settings": {
                "size": 100,
                "run_parallel": False,
                "batch_size": 100,
                "print_report": False,
                "random_seed": 1
            },
            "keys": {}
        }
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_init_worker_with_metadata(rp_logger):
    """
    Test the initialization of 'Worker' class with metadata
    during a training process
    """
    rp_logger.info("Test the initialization of the instance of 'Worker' class "
                   "with provided metadata during a training process")
    worker = Worker(
        table_name=None,
        metadata_path=r"./tests/unit/worker/fixtures/metadata.yaml",
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "print_report": True
        },
        log_level="INFO",
        type="train"
    )
    assert isinstance(worker.train_strategy, TrainStrategy) is True
    assert isinstance(worker.infer_strategy, InferStrategy) is True
    assert worker.metadata == {
        "test_table": {
            "source": "./path/to/test_table.csv",
            "train_settings": {
                "epochs": 100,
                "drop_null": False,
                "print_report": False,
                "row_limit": 800,
                "batch_size": 2000
            },
            "infer_settings": {
                "size": 200,
                "run_parallel": True,
                "random_seed": 2,
                "print_report": True,
                "batch_size": 200
            },
            "keys": {
                "pk_id": {
                    "columns": ["Id"],
                    "type": "PK"
                }
            }
        }
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_init_worker_with_empty_settings_in_metadata(rp_logger):
    """
    Test the initialization during a training process
    of 'Worker' class with metadata in which
    training and inference settings are empty
    """
    rp_logger.info("Test the initialization of the instance of 'Worker' class with provided metadata "
                   "in which training and inference settings are empty during a training process")
    worker = Worker(
        table_name=None,
        metadata_path="./tests/unit/worker/fixtures/metadata_with_empty_settings.yaml",
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "print_report": True
        },
        log_level="INFO",
        type="train"
    )
    assert isinstance(worker.train_strategy, TrainStrategy) is True
    assert isinstance(worker.infer_strategy, InferStrategy) is True
    assert worker.metadata == {
        "test_table": {
            "source": "./path/to/test_table.csv",
            "train_settings": {
                "epochs": 20,
                "drop_null": True,
                "print_report": True,
                "row_limit": 1000,
                "batch_size": 1000
            },
            "infer_settings": {},
            "keys": {
                "pk_id": {
                    "columns": ["Id"],
                    "type": "PK"
                }
            }
        }
    }
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_generate_reports", return_value=None)
@patch.object(Worker, "_Worker__train_tables", return_value=None)
def test_launch_train_with_metadata(mock_train_tables, mock_generate_reports, rp_logger):
    """
    Test that 'launch_train' method calls all necessary methods in case the metadata file was provided
    """
    rp_logger.info("Test that 'launch_train' method calls all necessary methods "
                   "in case the metadata file was provided")
    worker = Worker(
        table_name=None,
        metadata_path="./tests/unit/worker/fixtures/metadata.yaml",
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "print_report": True
        },
        log_level="INFO",
        type="train"
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        (["test_table"],
         {
             "test_table": {
                "train_settings": {
                    "epochs": 100,
                    "drop_null": False,
                    "print_report": False,
                    "row_limit": 800,
                    "batch_size": 2000
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "print_report": True,
                    "batch_size": 200},
                "source": "./path/to/test_table.csv",
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"]
                    }
                }
            }
        }
        ),
        (["test_table"],
         {
             "test_table": {
                 "train_settings": {
                     "epochs": 100,
                     "drop_null": False,
                     "print_report": False,
                     "row_limit": 800,
                     "batch_size": 2000
                 },
                'infer_settings': {
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "print_report": True,
                    "batch_size": 200
                },
                 "source": "./path/to/test_table.csv",
                 "keys": {
                     "pk_id": {
                         "type": "PK",
                         "columns": ["Id"]
                     }
                 }
             }
         }
         )
    )
    mock_generate_reports.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_generate_reports", return_value=None)
@patch.object(Worker, "_Worker__train_tables", return_value=None)
def test_launch_train_with_metadata_of_related_tables(mock_train_tables, mock_generate_reports, rp_logger):
    """
    Test that 'launch_train' method calls all necessary methods
    in case the metadata file of related tables was provided
    """
    rp_logger.info("Test that 'launch_train' method calls all necessary methods in case "
                   "the metadata file of related tables was provided")
    worker = Worker(
        table_name=None,
        metadata_path="./tests/unit/worker/fixtures/metadata_of_related_tables.yaml",
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "print_report": True
        },
        log_level="INFO",
        type="train"
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        (
            ["pk_test", "fk_test"],
            {
                "pk_test": {
                    "train_settings": {
                        "epochs": 1,
                        "drop_null": False,
                        "row_limit": 800,
                        "batch_size": 1000,
                        "print_report": True
                    },
                    "infer_settings": {
                        "size": 200,
                        "run_parallel": True,
                        "print_report": True
                    },
                    "source": "./path/to/pk_test.csv",
                    "keys": {
                        "pk_id": {
                            "type": "PK",
                            "columns": ["Id"]
                        }
                    }
                },
                "fk_test": {
                    "train_settings": {
                        "epochs": 5,
                        "drop_null": True,
                        "print_report": True,
                        "row_limit": 600,
                        "batch_size": 1000
                    },
                    "infer_settings": {
                        "size": 90,
                        "run_parallel": True,
                        "random_seed": 2,
                        "print_report": False
                    },
                    "source": "./path/to/fk_test.csv",
                    "keys": {
                        "fk_id": {
                            "type": "FK",
                            "columns":["Id"],
                            "references": {
                                "table": "pk_test",
                                "columns": ["Id"]}
                        }
                    }
                }
            }
        ),
        (
            ["pk_test", "fk_test"],
            {
                "pk_test": {
                    "train_settings": {
                        "epochs": 1,
                        "drop_null": False,
                        "row_limit": 800,
                        "batch_size": 1000,
                        "print_report": True
                    },
                    "infer_settings":
                        {
                            "size": 200,
                            "run_parallel": True,
                            "print_report": True
                        },
                    "source": "./path/to/pk_test.csv",
                    "keys": {
                        "pk_id": {
                            "type": "PK",
                            "columns": ["Id"]}
                    }
                },
                "fk_test": {
                    "train_settings": {
                        "epochs": 5,
                        "drop_null": True,
                        "print_report": True,
                        "row_limit": 600,
                        "batch_size": 1000
                    },
                    "infer_settings": {
                        "size": 90,
                        "run_parallel": True,
                        "random_seed": 2,
                        "print_report": False
                    },
                    "source": "./path/to/fk_test.csv",
                    "keys": {
                        "fk_id": {
                            "type": "FK",
                            "columns": ["Id"],
                            "references": {
                                "table": "pk_test",
                                "columns": ["Id"]}
                        }
                    }
                }
            }
        )
    )
    mock_generate_reports.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_generate_reports", return_value=None)
@patch.object(Worker, "_Worker__train_tables", return_value=None)
def test_launch_train_with_metadata_of_related_tables(mock_train_tables, mock_generate_reports, rp_logger):
    """
    Test that 'launch_train' method calls all necessary methods
    in case the metadata file of related tables was provided
    """
    rp_logger.info("Test that 'launch_train' method calls all necessary methods in case "
                   "the metadata file of related tables was provided")
    worker = Worker(
        table_name=None,
        metadata_path="./tests/unit/worker/fixtures/metadata_of_related_tables.yaml",
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "print_report": True
        },
        log_level="INFO",
        type="train"
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        (
            ["pk_test", "fk_test"],
            {
                "pk_test": {
                    "train_settings": {
                        "epochs": 1,
                        "drop_null": False,
                        "row_limit": 800,
                        "batch_size": 1000,
                        "print_report": True
                    },
                    "infer_settings": {
                        "size": 200,
                        "run_parallel": True,
                        "print_report": True
                    },
                    "source": "./path/to/pk_test.csv",
                    "keys": {
                        "pk_id": {
                            "type": "PK",
                            "columns": ["Id"]
                        }
                    }
                },
                "fk_test": {
                    "train_settings": {
                        "epochs": 5,
                        "drop_null": True,
                        "print_report": True,
                        "row_limit": 600,
                        "batch_size": 1000
                    },
                    "infer_settings": {
                        "size": 90,
                        "run_parallel": True,
                        "random_seed": 2,
                        "print_report": False
                    },
                    "source": "./path/to/fk_test.csv",
                    "keys": {
                        "fk_id": {
                            "type": "FK",
                            "columns":["Id"],
                            "references": {
                                "table": "pk_test",
                                "columns": ["Id"]}
                        }
                    }
                }
            }
        ),
        (
            ["pk_test", "fk_test"],
            {
                "pk_test": {
                    "train_settings": {
                        "epochs": 1,
                        "drop_null": False,
                        "row_limit": 800,
                        "batch_size": 1000,
                        "print_report": True
                    },
                    "infer_settings":
                        {
                            "size": 200,
                            "run_parallel": True,
                            "print_report": True
                        },
                    "source": "./path/to/pk_test.csv",
                    "keys": {
                        "pk_id": {
                            "type": "PK",
                            "columns": ["Id"]}
                    }
                },
                "fk_test": {
                    "train_settings": {
                        "epochs": 5,
                        "drop_null": True,
                        "print_report": True,
                        "row_limit": 600,
                        "batch_size": 1000
                    },
                    "infer_settings": {
                        "size": 90,
                        "run_parallel": True,
                        "random_seed": 2,
                        "print_report": False
                    },
                    "source": "./path/to/fk_test.csv",
                    "keys": {
                        "fk_id": {
                            "type": "FK",
                            "columns": ["Id"],
                            "references": {
                                "table": "pk_test",
                                "columns": ["Id"]}
                        }
                    }
                }
            }
        )
    )
    mock_generate_reports.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)

@patch.object(Worker, "_generate_reports", return_value=None)
@patch.object(Worker, "_Worker__train_tables", return_value=None)
def test_launch_train_with_metadata_of_related_tables_with_diff_keys(mock_train_tables, mock_generate_reports, rp_logger):
    """
    Test that 'launch_train' method calls all necessary methods
    in case the metadata file of related tables in which the table might contain several different keys was provided
    """
    rp_logger.info("Test that 'launch_train' method calls all necessary methods in case "
                   "the metadata file of related tables was provided")
    worker = Worker(
        table_name=None,
        metadata_path="./tests/unit/worker/fixtures/metadata_of_related_tables_with_diff_keys.yaml",
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "print_report": True
        },
        log_level="INFO",
        type="train"
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        (
            ["tdm_models", "tdm_clusters"],
            {
                "tdm_models": {
                    "source": "./path/to/tdm_models.csv",
                    "keys": {
                        "tdm_models_pkey": {
                            "type": "PK",
                            "columns": ["id"]
                        },
                        "tdm_models_fkey": {
                            "type": "FK",
                            "columns": ["cluster_id"],
                            "references": {
                                "table": "tdm_clusters",
                                "columns": ["id"]
                            }
                        }
                    },
                    "train_settings": {
                        "epochs": 20,
                        "drop_null": True,
                        "row_limit": 1000,
                        "batch_size": 1000,
                        "print_report": True
                    },
                    "infer_settings": {}
                },
                "tdm_clusters": {
                    'source': './path/to/tdm_clusters.csv',
                    'keys': {
                        'tdm_clusters_pkey': {
                            'type': 'PK',
                            'columns':['id']
                        }
                    },
                    "train_settings": {
                        "epochs": 20,
                        "drop_null": True,
                        "row_limit": 1000,
                        "batch_size": 1000,
                        "print_report": True
                    },
                    "infer_settings": {}
                }
            }
        ),
        (
            ["tdm_clusters", "tdm_models_pk", "tdm_models_fk"],
            {
                "tdm_clusters": {
                    "source": "./path/to/tdm_clusters.csv",
                    "keys": {
                        "tdm_clusters_pkey": {
                            "type": "PK",
                            "columns": ["id"]}
                    },
                    "train_settings": {
                        "epochs": 20,
                        "drop_null": True,
                        "row_limit": 1000,
                        "batch_size": 1000,
                        "print_report": True
                    },
                    "infer_settings": {}
                },
                "tdm_models_pk": {
                    "source": "./path/to/tdm_models.csv",
                    "keys": {
                        "tdm_models_pkey": {
                            "type": "PK",
                            "columns":["id"]
                        }
                    },
                    "train_settings": {
                        "epochs": 20,
                        "drop_null": True,
                        "row_limit": 1000,
                        "batch_size": 1000,
                        "print_report": True
                    },
                    "infer_settings": {}
                },
                "tdm_models_fk": {
                    "source": "./path/to/tdm_models.csv",
                    "keys": {
                        "tdm_models_fkey": {
                            "type": "FK",
                            "columns": ["cluster_id"],
                            "references": {
                                "table": "tdm_clusters",
                                "columns": ["id"]}
                        }
                    },
                    "train_settings": {
                        "epochs": 20,
                        "drop_null": True,
                        "row_limit": 1000,
                        "batch_size": 1000,
                        "print_report": True
                    },
                    "infer_settings": {}
                }
            }
        )
    )
    mock_generate_reports.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)



@patch.object(Worker, "_generate_reports", return_value=None)
@patch.object(Worker, "_Worker__train_tables", return_value=None)
def test_launch_train_without_metadata(mock_train_tables, mock_generate_reports, rp_logger):
    """
    Test that 'launch_train' method calls all necessary methods
    in case the metadata file wasn't provided and training process was launched through CLI
    """
    rp_logger.info("Test that 'launch_train' method calls all necessary methods "
                   "in case the metadata file wasn't provided and a training process was launched through CLI")
    worker = Worker(
        table_name="test_table",
        metadata_path=None,
        settings={
            "source": None,
            "epochs": 20,
            "drop_null": True,
            "row_limit": 1000,
            "batch_size": 1000,
            "print_report": True
        },
        log_level="INFO",
        type="train"
    )
    worker.launch_train()
    mock_train_tables.assert_called_once_with(
        (["test_table"],
        {
            "test_table": {
                "train_settings": {
                    "epochs": 20,
                    "drop_null": True,
                    "row_limit": 1000,
                    "batch_size": 1000,
                    "print_report": True
                },
                "infer_settings": {},
                "keys": {},
                "source": None
            }
        }),
        (["test_table"],
         {
             "test_table": {
                 "train_settings": {
                     "epochs": 20,
                     "drop_null": True,
                     "row_limit": 1000,
                     "batch_size": 1000,
                     "print_report": True
                 },
                 "infer_settings": {},
                 "keys": {},
                 "source": None
             }
         })
    )
    mock_generate_reports.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)

@patch.object(Worker, "_generate_reports", return_value=None)
@patch.object(Worker, "_Worker__infer_tables", return_value=None)
def test_launch_infer_with_metadata(mock_infer_tables, mock_generate_reports, rp_logger):
    """
    Test that 'launch_infer' method calls all necessary methods in case the metadata file was provided
    """
    rp_logger.info("Test that 'launch_infer' method calls all necessary methods in case the metadata file was provided")
    worker = Worker(
        table_name=None,
        metadata_path="./tests/unit/worker/fixtures/metadata.yaml",
        settings={
            "size": 200,
            "run_parallel": True,
            "random_seed": 2,
            "print_report": True,
            "batch_size": 200
        },
        log_level="INFO",
        type="infer"
    )
    worker.launch_infer()
    mock_infer_tables.assert_called_once_with(
        ["test_table"],
        {
            "test_table": {
                "train_settings": {
                    "epochs": 100,
                    "drop_null": False,
                    "print_report": False,
                    "row_limit": 800,
                    "batch_size": 2000,
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "print_report": True,
                    "batch_size": 200
                },
                "source": "./path/to/test_table.csv",
                "keys": {
                    "pk_id": {
                        "type": "PK",
                        "columns": ["Id"]}
                }
            }
        }
    )
    mock_generate_reports.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_generate_reports", return_value=None)
@patch.object(Worker, "_Worker__infer_tables", return_value=None)
def test_launch_infer_with_metadata_of_related_tables(mock_infer_tables, mock_generate_reports, rp_logger):
    """
    Test that 'launch_infer' method calls all necessary methods
    in case the metadata file of related tables was provided
    """
    rp_logger.info("Test that 'launch_infer' method calls all necessary methods in case "
                   "the metadata file of related tables was provided")
    worker = Worker(
        table_name=None,
        metadata_path="./tests/unit/worker/fixtures/metadata_of_related_tables.yaml",
        settings={
            "size": 300,
            "run_parallel": True,
            "print_report": True,
            "batch_size": 200,
            "random_seed": 1
        },
        log_level="INFO",
        type="infer"
    )
    worker.launch_infer()
    mock_infer_tables.assert_called_once_with(
        ["pk_test", "fk_test"],
        {
            "pk_test": {
                "train_settings": {
                    "epochs": 1,
                    "drop_null": False,
                    "row_limit": 800
                },
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "print_report": True,
                    "batch_size": 200,
                    "random_seed": 1
                },
                "source": "./path/to/pk_test.csv",
                "keys": {
                    "pk_id": {
                        "type": "PK", "columns": ["Id"]
                    }
                }
            },
            "fk_test": {
                "train_settings": {
                    "epochs": 5,
                    "drop_null": True,
                    "print_report": True,
                    "row_limit": 600
                },
                "infer_settings": {
                    "size": 90,
                    "run_parallel": True,
                    "random_seed": 2,
                    "print_report": False,
                    "batch_size": 200
                },
                "source": "./path/to/fk_test.csv",
                "keys": {
                    "fk_id": {
                        "type": "FK",
                        "columns": ["Id"],
                        "references": {
                            "table": "pk_test",
                            "columns": ["Id"]
                        }
                    }
                }
            }
        }
    )
    mock_generate_reports.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_generate_reports", return_value=None)
@patch.object(Worker, "_Worker__infer_tables", return_value=None)
def test_launch_infer_with_metadata_of_related_tables_with_diff_keys(mock_infer_tables, mock_generate_reports, rp_logger):
    """
    Test that 'launch_infer' method calls all necessary methods
    in case the metadata file of related tables in which the table might contain several different keys was provided
    """
    rp_logger.info("Test that 'launch_infer' method calls all necessary methods in case "
                   "the metadata file of related tables was provided")
    worker = Worker(
        table_name=None,
        metadata_path="./tests/unit/worker/fixtures/metadata_of_related_tables_with_diff_keys.yaml",
        settings={
            "size": 300,
            "run_parallel": True,
            "print_report": True,
            "batch_size": 200,
            "random_seed": 1
        },
        log_level="INFO",
        type="infer"
    )
    worker.launch_infer()
    mock_infer_tables.assert_called_once_with(
        ["tdm_clusters", "tdm_models_pk", "tdm_models_fk"],
        {
            "tdm_clusters": {
                "source": "./path/to/tdm_clusters.csv",
                "keys": {
                    "tdm_clusters_pkey": {
                        "type": "PK",
                        "columns": ["id"]
                    }
                },
                "train_settings": {},
                "infer_settings": {
                    "size": 300,
                    "run_parallel": True,
                    "print_report": True,
                    "batch_size": 200,
                    "random_seed": 1
                }
            },
            "tdm_models_pk": {
                "source": "./path/to/tdm_models.csv",
                "keys": {
                    "tdm_models_pkey": {
                        "type": "PK",
                        "columns": ["id"]
                    }
                },
                "train_settings": {},
                "infer_settings": {
                    "size": 300,
                    "run_parallel": True,
                    "print_report": True,
                    "batch_size": 200,
                    "random_seed": 1
                }
            },
            "tdm_models_fk": {
                "source": "./path/to/tdm_models.csv",
                "keys": {
                    "tdm_models_fkey": {
                        "type": "FK",
                        "columns": ["cluster_id"],
                        "references": {
                            "table": "tdm_clusters",
                            "columns": ["id"]}
                    }
                },
                "train_settings": {},
                "infer_settings": {
                    "size": 300,
                    "run_parallel": True,
                    "print_report": True,
                    "batch_size": 200,
                    "random_seed": 1
                }
            }
        }
    )
    mock_generate_reports.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Worker, "_generate_reports", return_value=None)
@patch.object(Worker, "_Worker__infer_tables", return_value=None)
def test_launch_infer_without_metadata(mock_infer_tables, mock_generate_reports, rp_logger):
    """
    Test that 'launch_infer' method calls all necessary methods
    in case the metadata file wasn't provided and inference process was launched through CLI
    """
    rp_logger.info("Test that 'launch_infer' method calls all necessary methods "
                   "in case the metadata file wasn't provided and inference process was launched through CLI")
    worker = Worker(
        table_name="test_table",
        metadata_path=None,
        settings={
            "size": 200,
            "run_parallel": True,
            "random_seed": 2,
            "print_report": True,
            "batch_size": 200
        },
        log_level="INFO",
        type="infer"
    )
    worker.launch_infer()
    mock_infer_tables.assert_called_once_with(
        ["test_table"],
        {
            "test_table": {
                "train_settings": {},
                "infer_settings": {
                    "size": 200,
                    "run_parallel": True,
                    "random_seed": 2,
                    "print_report": True,
                    "batch_size": 200
                },
                "keys": {},
                "source": None
            }
        }
    )
    mock_generate_reports.assert_called_once()
    rp_logger.info(SUCCESSFUL_MESSAGE)
