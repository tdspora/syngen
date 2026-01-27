from unittest.mock import patch, call, MagicMock
import pytest
from collections import defaultdict

from syngen.ml.config.validation import Validator
from syngen.ml.utils import ValidationError
from tests.conftest import SUCCESSFUL_MESSAGE, DIR_NAME, get_dataframe

FAKE_METADATA_PATH = "path/to/metadata.yaml"


@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_metadata_of_one_table_without_fk_key_in_train_process(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_referential_integrity,
    mock_check_completion_of_training,
    rp_logger
):
    """
    Test the validation of the metadata of one table contained only the primary key
    during the training process
    """
    rp_logger.info(
        "Test the validation of the metadata of one table contained the primary key "
        "during the training process"
    )
    test_metadata = {
        "test_table": {
            "train_settings": {
                "source": "path/to/test_table.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        }
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="train",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {}
    assert validator.merged_metadata == test_metadata
    mock_get_columns_of_source.assert_called_once_with(
        path_to_source="path/to/test_table.csv", table_name="test_table"
    )
    mock_get_columns_by_loader.assert_not_called()
    mock_check_existence_of_key_columns.assert_called_once_with("test_table")
    mock_check_existence_of_referenced_columns.assert_called_once_with("test_table")
    mock_validate_referential_integrity.assert_not_called()
    mock_check_completion_of_training.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_metadata_of_one_table_without_fk_key_in_train_process_with_loader(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_referential_integrity,
    mock_check_completion_of_training,
    rp_logger
):
    """
    Test the validation of the metadata of one table contained only the primary key
    during the training process with the provided 'loader'
    """
    rp_logger.info(
        "Test the validation of the metadata of one table "
        "contained only the primary key during the training process "
        "with the provided 'loader'"
    )
    test_metadata = {
        "test_table": {
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            },
        }
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="train",
        metadata_path=FAKE_METADATA_PATH,
        loader=get_dataframe
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {}
    assert validator.merged_metadata == test_metadata
    mock_get_columns_of_source.assert_not_called()
    mock_get_columns_by_loader.assert_called_once_with("test_table")
    mock_check_existence_of_key_columns.assert_called_once_with("test_table")
    mock_check_existence_of_referenced_columns.assert_called_once_with("test_table")
    mock_validate_referential_integrity.assert_not_called()
    mock_check_completion_of_training.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_check_key_column_in_pk(rp_logger):
    """
    Test that the column of the primary key exists in the source table
    """
    rp_logger.info(
        "Test that the column of the primary key exists in the source table"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/table_with_data.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        }
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="train",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {}
    assert validator.merged_metadata == test_metadata
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_check_key_column_in_fk(rp_logger):
    """
    Test that the column of the foreign key exists in the parent table
    """
    rp_logger.info(
        "Test that the column of the foreign key exists in the parent table"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": f"{DIR_NAME}/unit/data_loaders/fixtures/"
                          "csv_tables/table_with_data.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        },
        "table_b": {
            "train_settings": {
                "source": f"{DIR_NAME}/unit/data_loaders/fixtures/"
                          "csv_tables/child_table_with_data.csv"
            },
            "keys": {
                "fk_id": {
                    "type": "FK",
                    "columns": ["Id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                },
            }
        }
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="train",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {
        "fk_id": {
            "parent_columns": ["id"],
            "parent_table": "table_a"
        }
    }
    assert validator.merged_metadata == test_metadata
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_metadata_of_related_tables_with_fk_key_in_train_process(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_referential_integrity,
    mock_check_completion_of_training,
    rp_logger
):
    """
    Test the validation of the metadata of related tables
    contained the primary key and the foreign key
    during the training process
    """
    rp_logger.info(
        "Test the validation of the metadata of related tables "
        "contained only the primary key and the foreign key during the training process"
    )
    test_metadata = {
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "fk_id": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        },
        "table_a": {
            "train_settings": {
                "source": "path/to/table_a.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        },
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="train",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {
        "fk_id": {
            "parent_table": "table_a",
            "parent_columns": ["id"]
        }
    }
    assert mock_get_columns_of_source.call_count == 2
    mock_get_columns_by_loader.assert_not_called()
    assert mock_check_existence_of_key_columns.call_count == 2
    assert mock_check_existence_of_referenced_columns.call_count == 2
    mock_validate_referential_integrity.assert_called_once_with(
        fk_name="fk_id",
        fk_config={
            "type": "FK",
            "columns": ["id"],
            "references": {
                "table": "table_a",
                "columns": ["id"]
            }
        },
        parent_config={
            "train_settings": {
                "source": "path/to/table_a.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        }
    )
    mock_check_completion_of_training.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_metadata_of_related_tables_with_fk_key_in_train_process_with_loader(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_referential_integrity,
    mock_check_completion_of_training,
    rp_logger
):
    """
    Test the validation of the metadata of related tables
    contained the primary key and the foreign key
    during the training process with the provided 'loader'
    """
    rp_logger.info(
        "Test the validation of the metadata of related tables "
        "contained the primary key and the foreign key "
        "during the training process with the provided 'loader'"
    )
    test_metadata = {
        "table_a": {
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        },
        "table_b": {
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "fk_id": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="train",
        metadata_path=FAKE_METADATA_PATH,
        loader=get_dataframe
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {
        "fk_id": {
            "parent_table": "table_a",
            "parent_columns": ["id"]
        }
    }
    assert validator.merged_metadata == test_metadata
    mock_get_columns_of_source.assert_not_called()
    assert mock_get_columns_by_loader.call_count == 2
    assert mock_check_existence_of_key_columns.call_count == 2
    assert mock_check_existence_of_referenced_columns.call_count == 2
    mock_validate_referential_integrity.assert_called_once_with(
        fk_name="fk_id",
        fk_config={
            "type": "FK",
            "columns": ["id"],
            "references": {
                "table": "table_a",
                "columns": ["id"]
            }
        },
        parent_config={
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        }
    )
    mock_check_completion_of_training.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_metadata_of_related_tables_with_several_fk_key_in_train_process(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_referential_integrity,
    mock_check_completion_of_training,
    rp_logger
):
    """
    Test the validation of the metadata of related tables
    contained several foreign keys
    during the training process
    """
    rp_logger.info(
        "Test the validation of the metadata of related tables contained several foreign keys "
        "during the training process"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": "path/to/table_a.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "uq_id": {
                    "type": "UQ",
                    "columns": ["name"]
                }
            }
        },
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv"
            },
            "keys": {
                "fk_1": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                },
                "fk_2": {
                    "type": "FK",
                    "columns": ["name"],
                    "references": {
                        "table": "table_a",
                        "columns": ["name"]
                    }
                }
            }
        }
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="train",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {
        "fk_1": {
            "parent_table": "table_a",
            "parent_columns": ["id"]
        },
        "fk_2": {
            "parent_table": "table_a",
            "parent_columns": ["name"]
        }
    }
    assert validator.merged_metadata == test_metadata
    assert mock_get_columns_of_source.call_count == 2
    mock_get_columns_by_loader.assert_not_called()
    assert mock_check_existence_of_key_columns.call_count == 2
    assert mock_check_existence_of_referenced_columns.call_count == 2
    assert mock_validate_referential_integrity.call_count == 2
    mock_check_completion_of_training.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_metadata_of_related_tables_with_several_fk_key_in_train_with_loader(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_referential_integrity,
    mock_check_completion_of_training,
    rp_logger
):
    """
    Test the validation of the metadata of related tables
    contained several foreign keys during the training process
    with the provided 'loader'
    """
    rp_logger.info(
        "Test the validation of the metadata of related tables contained several foreign keys "
        "during the training process with the provided 'loader'"
    )
    test_metadata = {
        "table_a": {
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "uq_id": {
                    "type": "UQ",
                    "columns": ["name"]
                }
            }
        },
        "table_b": {
            "keys": {
                "fk_1": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                },
                "fk_2": {
                    "type": "FK",
                    "columns": ["name"],
                    "references": {
                        "table": "table_a",
                        "columns": ["name"]
                    }
                }
            }
        }
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="train",
        metadata_path=FAKE_METADATA_PATH,
        loader=get_dataframe
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {
        "fk_1": {
            "parent_table": "table_a",
            "parent_columns": ["id"]
        },
        "fk_2": {
            "parent_table": "table_a",
            "parent_columns": ["name"]
        }
    }
    assert validator.merged_metadata == test_metadata
    mock_get_columns_of_source.assert_not_called()
    assert mock_get_columns_by_loader.call_count == 2
    assert mock_check_existence_of_key_columns.call_count == 2
    assert mock_check_existence_of_referenced_columns.call_count == 2
    assert mock_validate_referential_integrity.call_count == 2
    mock_check_completion_of_training.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_metadata_of_one_table_without_fk_key_in_infer_process(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_referential_integrity,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_check_existence_of_generated_data,
    rp_logger
):
    """
    Test the validation of the metadata of one table
    contained the primary key during the inference process
    """
    rp_logger.info(
        "Test the validation of the metadata of one table contained the primary key "
        "during the inference process"
    )
    test_metadata = {
        "test_table": {
            "train_settings": {
                "source": "path/to/test_table.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        }
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="infer",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {}
    assert validator.merged_metadata == test_metadata
    mock_get_columns_of_source.assert_not_called()
    mock_get_columns_by_loader.assert_not_called()
    mock_check_existence_of_key_columns.assert_not_called()
    mock_check_existence_of_referenced_columns.assert_not_called()
    mock_validate_referential_integrity.assert_not_called()
    mock_check_completion_of_training.assert_called_once_with("test_table")
    mock_check_existence_of_destination.assert_called_once_with("test_table")
    mock_check_existence_of_generated_data.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_metadata_of_related_tables_without_fk_key_in_infer_process(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_referential_integrity,
    mock_check_existence_of_generated_data,
    rp_logger
):
    """
    Test the validation of the metadata of related tables
    contained only the primary key and the unique key
    during the inference process
    """
    rp_logger.info(
        "Test the validation of the metadata of related tables "
        "contained only the primary key and the unique key during the inference process"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": "path/to/table_a.csv"
            },
            "infer_settings": {
                "destination": "path/to/generated_table_a.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        },
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "uq_id": {
                    "type": "UQ",
                    "columns": ["name"]
                }
            }
        }
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="infer",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {}
    assert validator.merged_metadata == test_metadata
    mock_get_columns_of_source.assert_not_called()
    mock_get_columns_by_loader.assert_not_called()
    mock_check_existence_of_key_columns.assert_not_called()
    mock_check_existence_of_referenced_columns.assert_not_called()
    assert mock_check_completion_of_training.call_count == 2
    assert mock_check_existence_of_destination.call_count == 2
    mock_validate_referential_integrity.assert_not_called()
    mock_check_existence_of_generated_data.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_metadata_of_related_tables_with_fk_key_in_infer_process(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_referential_integrity,
    mock_check_existence_of_generated_data,
    rp_logger
):
    """
    Test the validation of the metadata of related tables
    contained the primary key and the foreign key
    during the inference process
    """
    rp_logger.info(
        "Test the validation of the metadata of related tables "
        "contained only the primary key and the foreign key during the inference process"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": "path/to/table_a.csv"
            },
            "infer_settings": {
                "destination": "path/to/generated_table_a.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        },
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "fk_id": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="infer",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {
        "fk_id": {
            "parent_table": "table_a",
            "parent_columns": ["id"]
        }
    }
    assert validator.merged_metadata == test_metadata
    mock_get_columns_of_source.assert_not_called()
    mock_get_columns_by_loader.assert_not_called()
    mock_check_existence_of_key_columns.assert_not_called()
    mock_check_existence_of_referenced_columns.assert_not_called()
    assert mock_check_completion_of_training.call_count == 2
    assert mock_check_existence_of_destination.call_count == 2
    mock_validate_referential_integrity.assert_called_once_with(
        fk_name="fk_id",
        fk_config={
            "type": "FK",
            "columns": ["id"],
            "references": {
                "table": "table_a",
                "columns": ["id"]
            }
        },
        parent_config={
            "train_settings": {
                "source": "path/to/table_a.csv"
            },
            "infer_settings": {
                "destination": "path/to/generated_table_a.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        }
    )
    mock_check_existence_of_generated_data.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_metadata_of_related_tables_with_several_fk_key_in_infer_process(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_validate_referential_integrity,
    mock_check_existence_of_generated_data,
    rp_logger
):
    """
    Test the validation of the metadata of related tables
    contained several foreign keys during the inference process
    """
    rp_logger.info(
        "Test the validation of the metadata of related tables contained several foreign keys "
        "during the inference process"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": "path/to/table_a.csv"
            },
            "infer_settings": {
                "destination": "path/to/generated_table_a.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "uq_id": {
                    "type": "UQ",
                    "columns": ["name"]
                }
            }
        },
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv"
            },
            "keys": {
                "fk_1": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                },
                "fk_2": {
                    "type": "FK",
                    "columns": ["name"],
                    "references": {
                        "table": "table_a",
                        "columns": ["name"]
                    }
                }
            }
        }
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="infer",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {
        "fk_1": {
            "parent_table": "table_a",
            "parent_columns": ["id"]
        },
        "fk_2": {
            "parent_table": "table_a",
            "parent_columns": ["name"]
        }
    }
    assert validator.merged_metadata == test_metadata
    mock_get_columns_of_source.assert_not_called()
    mock_get_columns_by_loader.assert_not_called()
    mock_check_existence_of_key_columns.assert_not_called()
    mock_check_existence_of_referenced_columns.assert_not_called()
    assert mock_check_completion_of_training.call_count == 2
    assert mock_check_existence_of_destination.call_count == 2
    assert mock_validate_referential_integrity.call_count == 2
    mock_check_existence_of_generated_data.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_incomplete_metadata_contained_fk_key_in_train_process_without_reports(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_referential_integrity,
    mock_check_completion_of_training,
    mock_check_existence_of_generated_data,
    test_metadata_storage,
    rp_logger
):
    """
    Test the validation of the incomplete metadata of one table
    contained the foreign key but not contained the information of the parent table.
    It's used in the training process without the generation of reports
    """
    rp_logger.info(
        "Test the validation of the incomplete metadata of one table "
        "contained the foreign key but not contained the information of the parent table. "
        "It used in the training process without the generation of reports"
    )
    metadata = {
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv",
                "reports": []
            },
            "keys": {
                "fk_key": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    validator = Validator(
        metadata=metadata,
        type_of_process="train",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {
        "fk_key": {
            "parent_table": "table_a",
            "parent_columns": ["id"]
        }
    }
    assert validator.merged_metadata == {
        "table_a": {
            "train_settings": {
                "source": "path/to/table_a.csv",
                "reports": ["accuracy", "sample"]
            },
            "infer_settings": {
                "destination": "path/to/generated_table_a.csv"
            },
            "encryption": {},
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "uq_id": {
                    "type": "UQ",
                    "columns": ["name"]
                }
            },
            "format": {}
        },
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv",
                "reports": []
            },
            "keys": {
                "fk_key": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    assert mock_get_columns_of_source.call_count == 2
    mock_get_columns_by_loader.assert_not_called()
    mock_check_existence_of_key_columns.assert_called_once_with("table_b")
    mock_check_existence_of_referenced_columns.assert_called_once_with("table_b")
    mock_validate_referential_integrity.assert_called_once_with(
        fk_name="fk_key",
        fk_config={
            "type": "FK",
            "columns": ["id"],
            "references": {
                "table": "table_a",
                "columns": ["id"]
            }
        },
        parent_config={
            "train_settings": {
                "source": "path/to/table_a.csv",
                "reports": ["accuracy", "sample"]
            },
            "infer_settings": {
                "destination": "path/to/generated_table_a.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "uq_id": {
                    "type": "UQ",
                    "columns": ["name"]
                }
            },
            "encryption": {},
            "format": {}
        }
    )
    mock_check_completion_of_training.assert_called_once_with("table_a")
    mock_check_existence_of_generated_data.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("value", [
    ["accuracy", "sample"],
    ["accuracy", "metrics_only"],
    ["accuracy"],
    ["metrics_only"]
])
@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_incomplete_metadata_contained_fk_key_in_train_process_with_gen_data_and_reports(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_referential_integrity,
    mock_check_completion_of_training,
    mock_check_existence_of_generated_data,
    test_metadata_storage,
    value,
    rp_logger
):
    """
    Test the validation of the incomplete metadata of one table
    contained the foreign key but not contained the information of the parent table.
    It's used in the training process with the generation reports
    that requires the generation of the synthetic data
    """
    rp_logger.info(
        "Test the validation of the incomplete metadata of one table "
        "contained the foreign key but not contained the information of the parent table. "
        "It's used in the training process with the generation reports "
        "that requires the generation of the synthetic data "
    )
    metadata = {
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv",
                "reports": value
            },
            "keys": {
                "fk_key": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    validator = Validator(
        metadata=metadata,
        type_of_process="train",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {
        "fk_key": {
            "parent_columns": ["id"],
            "parent_table": "table_a"
        }
    }
    assert validator.merged_metadata == {
        "table_a": {
            "train_settings": {
                "source": "path/to/table_a.csv",
                "reports": ["accuracy", "sample"]
            },
            "infer_settings": {
                "destination": "path/to/generated_table_a.csv"
            },
            "encryption": {},
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "uq_id": {
                    "type": "UQ",
                    "columns": ["name"]
                }
            },
            "format": {}
        },
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv",
                "reports": value
            },
            "keys": {
                "fk_key": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    assert mock_get_columns_of_source.call_count == 2
    mock_get_columns_by_loader.assert_not_called()
    mock_check_existence_of_key_columns.assert_called_once_with("table_b")
    mock_check_existence_of_referenced_columns.assert_called_once_with("table_b")
    mock_validate_referential_integrity.assert_called_once_with(
        fk_name="fk_key",
        fk_config={
            "type": "FK",
            "columns": ["id"],
            "references": {
                "table": "table_a",
                "columns": ["id"]
            }
        },
        parent_config={
            "train_settings": {
                "source": "path/to/table_a.csv",
                "reports": ["accuracy", "sample"]
            },
            "infer_settings": {
                "destination": "path/to/generated_table_a.csv"
            },
            "encryption": {},
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "uq_id": {
                    "type": "UQ",
                    "columns": ["name"]
                }
            },
            "format": {}
        }
    )
    mock_check_completion_of_training.assert_called_once_with("table_a")
    mock_check_existence_of_generated_data.assert_called_once_with("table_a")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_incomplete_metadata_contained_fk_key_in_train_process_with_gen_sample_report(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_referential_integrity,
    mock_check_completion_of_training,
    mock_check_existence_of_generated_data,
    test_metadata_storage,
    rp_logger
):
    """
    Test the validation of the incomplete metadata of one table
    contained the foreign key but not contained the information of the parent table.
    It's used in the training process with the generation only a 'sample' report
    that doesn't require the generation of the synthetic data
    """
    rp_logger.info(
        "Test the validation of the incomplete metadata of one table "
        "contained the foreign key but not contained the information of the parent table. "
        "It's used in the training process with the generation only a 'sample' report "
        "that doesn't require the generation of the synthetic data"
    )
    metadata = {
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv",
                "reports": ["sample"]
            },
            "keys": {
                "fk_key": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    validator = Validator(
        metadata=metadata,
        type_of_process="train",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {
        "fk_key": {
            "parent_table": "table_a",
            "parent_columns": ["id"]
        }
    }
    assert validator.merged_metadata == {
        "table_a": {
            "train_settings": {
                "source": "path/to/table_a.csv",
                "reports": ["accuracy", "sample"]
            },
            "infer_settings": {
                "destination": "path/to/generated_table_a.csv"
            },
            "encryption": {},
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "uq_id": {
                    "type": "UQ",
                    "columns": ["name"]
                }
            },
            "format": {}
        },
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv",
                "reports": ["sample"]
            },
            "keys": {
                "fk_key": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    assert mock_get_columns_of_source.call_count == 2
    mock_get_columns_by_loader.assert_not_called()
    mock_check_existence_of_key_columns.assert_called_once_with("table_b")
    mock_check_existence_of_referenced_columns.assert_called_once_with("table_b")
    mock_validate_referential_integrity.assert_called_once_with(
        fk_name="fk_key",
        fk_config={
            "type": "FK",
            "columns": ["id"],
            "references": {
                "table": "table_a",
                "columns": ["id"]
            }
        },
        parent_config={
            "train_settings": {
                "source": "path/to/table_a.csv",
                "reports": ["accuracy", "sample"]
            },
            "infer_settings": {
                "destination": "path/to/generated_table_a.csv"
            },
            "encryption": {},
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "uq_id": {
                    "type": "UQ",
                    "columns": ["name"]
                }
            },
            "format": {}
        }
    )
    mock_check_completion_of_training.assert_called_once_with("table_a")
    mock_check_existence_of_generated_data.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_incomplete_metadata_in_infer_process(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_existence_of_destination,
    mock_validate_referential_integrity,
    mock_check_completion_of_training,
    mock_check_existence_of_generated_data,
    test_metadata_storage,
    rp_logger
):
    """
    Test the validation of the incomplete metadata of one table
    contained the foreign key but not contained the information of the parent table.
    It's used in the inference process
    """
    rp_logger.info(
        "Test the validation of the incomplete metadata of one table "
        "contained the foreign key but not contained the information of the parent table. "
        "It used in the inference process"
    )
    metadata = {
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv",
            },
            "keys": {
                "fk_key": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    validator = Validator(
        metadata=metadata,
        type_of_process="infer",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    assert validator.mapping == {
        "fk_key": {
            "parent_table": "table_a",
            "parent_columns": ["id"]
        }
    }
    assert validator.merged_metadata == {
        "table_a": {
            "train_settings": {
                "source": "path/to/table_a.csv",
                "reports": ["accuracy", "sample"]
            },
            "infer_settings": {
                "destination": "path/to/generated_table_a.csv"
            },
            "encryption": {},
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "uq_id": {
                    "type": "UQ",
                    "columns": ["name"]
                }
            },
            "format": {}
        },
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv"
            },
            "keys": {
                "fk_key": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    mock_get_columns_of_source.assert_not_called()
    mock_get_columns_by_loader.assert_not_called()
    mock_check_existence_of_key_columns.assert_not_called()
    mock_check_existence_of_referenced_columns.assert_not_called()
    mock_check_existence_of_destination.assert_called_once_with("table_b")
    mock_validate_referential_integrity.assert_called_once_with(
        fk_name="fk_key",
        fk_config={
            "type": "FK",
            "columns": ["id"],
            "references": {
                "table": "table_a",
                "columns": ["id"]
            }
        },
        parent_config={
            "train_settings": {
                "source": "path/to/table_a.csv",
                "reports": ["accuracy", "sample"]
            },
            "infer_settings": {
                "destination": "path/to/generated_table_a.csv"
            },
            "encryption": {},
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                },
                "uq_id": {
                    "type": "UQ",
                    "columns": ["name"]
                }
            },
            "format": {}
        }
    )
    assert mock_check_completion_of_training.call_args_list == [call("table_b"), call("table_a")]
    mock_check_existence_of_generated_data.assert_called_once_with("table_a")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
def test_validate_metadata_with_not_existent_source_in_train_process(
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_referential_integrity,
    caplog,
    rp_logger
):
    """
    Test the validation of the metadata of one table during the training process.
    The source of the table is not existent.
    """
    rp_logger.info(
        "Test the validation of the metadata of one table during the training process. "
        "The source of the table is not existent."
    )
    test_metadata = {
        "test_table": {
            "train_settings": {
                "source": "path/to/test_table.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            validator = Validator(
                metadata=test_metadata,
                type_of_process="train",
                metadata_path=FAKE_METADATA_PATH,
                loader=None
            )
            validator.run()
            assert validator.mapping == {}
            assert validator.merged_metadata == test_metadata
            mock_check_existence_of_key_columns.assert_not_called()
            mock_check_existence_of_referenced_columns.assert_not_called()
            mock_validate_referential_integrity.assert_not_called()
            message = (
                "The validation of the metadata has been failed. The error(s) found in - "
                "\"check existence of the source\": {\"test_table\": \"It seems that the path "
                "to the source of the table - 'test_table' isn't correct. Please, check the path "
                "to the source of the table - 'test_table'\"}"
            )
            assert str(error.value) == message
            assert message in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_validate_incomplete_metadata_with_absent_parent_metadata_in_metadata_storage(
    test_metadata_storage,
    caplog,
    rp_logger
):
    """
    Test the validation of the incomplete metadata of one table contained the foreign key.
    The information of the parent table is absent in the metadata storage.
    """
    rp_logger.info(
        "Test the validation of the incomplete metadata of one table "
        "contained the foreign key. "
        "The information of the parent table is absent in the metadata storage."
    )
    metadata = {
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv",
            },
            "keys": {
                "fk_key": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_c",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            validator = Validator(
                metadata=metadata,
                type_of_process="train",
                metadata_path=FAKE_METADATA_PATH,
                loader=None
            )
            validator.errors = defaultdict(defaultdict)
            validator.run()
            assert validator.mapping == {
                "fk_key": {
                    "parent_table": "table_a",
                    "parent_columns": ["id"]
                }
            }
            message = (
                "The metadata of the parent table - 'table_c' hasn't been found. "
                "Please, check whether the information of the parent table - "
                "'{parent_table}' exists in the current metadata file or in the metadata files "
                "stored in 'model_artifacts/metadata' directory"
            )
            assert str(error.value) == message
            assert message in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_incomplete_metadata_with_wrong_referential_integrity(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    test_metadata_storage,
    caplog,
    rp_logger
):
    """
    Test the validation of the incomplete metadata of one table contained the foreign key.
    The information of the parent table is present in the metadata storage,
    but the parent table contains the PK which contains the list of columns
    which not correspond to the list of the columns of the FK.
    """
    rp_logger.info(
        "Test the validation of the incomplete metadata of one table contained "
        "the foreign key. The information of the parent table is present in the metadata storage, "
        "but the parent table contains the PK which contains the list of columns "
        "which not correspond to the list of the columns of the FK"
    )
    metadata = {
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv",
            },
            "keys": {
                "fk_key": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_d",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            validator = Validator(
                metadata=metadata,
                type_of_process="train",
                metadata_path=FAKE_METADATA_PATH,
                loader=None
            )
            validator.run()
            assert validator.mapping == {
                "fk_key": {
                    "parent_table": "table_d",
                    "parent_columns": ["id"]
                }
            }
            assert validator.merged_metadata == {
                "table_b": {
                    "train_settings": {
                        "source": "path/to/table_b.csv",
                    },
                    "keys": {
                        "fk_key": {
                            "type": "FK",
                            "columns": ["id"],
                            "references": {
                                "table": "table_d",
                                "columns": ["id"]
                            }
                        }
                    }
                },
                "table_d": {
                    "train_settings": {
                        "source": "path/to/table_a.csv",
                        "reports": ["accuracy", "sample"]
                    },
                    "infer_settings": {
                        "destination": "path/to/generated_table_a.csv"
                    },
                    "keys": {
                        "pk_id": {
                            "type": "PK",
                            "columns": ["another_id"]
                        },
                    }
                }
            }
            assert mock_get_columns_of_source.call_count == 2
            mock_get_columns_by_loader.assert_not_called()
            assert mock_check_existence_of_key_columns.call_count == 2
            assert mock_check_existence_of_referenced_columns.call_count == 2
            mock_check_completion_of_training.assert_called_once_with("table_d")
            message = (
                "The validation of the metadata has been failed. "
                "The error(s) found in - \"validate referential integrity\": "
                "{\"fk_key\": \"The primary key columns associated with the columns "
                "of the foreign key - 'fk_key' is not the same\"}"
            )
            assert str(error.value) == message
            assert message in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_validate_referential_integrity")
def test_validate_metadata_with_not_existent_destination(
    mock_validate_referential_integrity,
    caplog,
    rp_logger
):
    """
    Test the validation of the metadata of one table during the inference process.
    The destination for the generated data is not existent.
    """
    rp_logger.info(
        "Test the validation of the metadata of one table during the inference process. "
        "The destination of the table is not existent."
    )
    test_metadata = {
        "test_table": {
            "train_settings": {
                "source": "path/to/test_table.csv"
            },
            "infer_settings": {
                "destination": "path/to/generated_test_table.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            validator = Validator(
                metadata=test_metadata,
                type_of_process="infer",
                metadata_path=FAKE_METADATA_PATH,
                loader=None
            )
            validator.run()
            assert validator.mapping == {}
            assert validator.merged_metadata == test_metadata
            mock_validate_referential_integrity.assert_called_once_with()
            message = (
                "The validation of the metadata has been failed. The error(s) found in - "
                "\"check existence of the destination\": {\"test_table\": \"It seems that "
                "the directory path for storing the generated data of table 'test_table' "
                "isn't correct. Please, verify the destination path\"}"
            )
            assert str(error.value) == message
            assert message in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_incomplete_metadata_with_absent_success_file_of_parent_table_in_train_process(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_referential_integrity,
    mock_check_completion_of_training,
    mock_check_existence_of_generated_data,
    test_metadata_storage,
    caplog,
    rp_logger
):
    """
    Test the validation of the incomplete metadata of one table contained the foreign key
    during the training process.
    The information of the parent table is present in the metadata storage,
    but the parent table hasn't been trained previously
    that's why the success file of the parent table is absent
    """
    rp_logger.info(
        "Test the validation of the incomplete metadata of one table contained the foreign key "
        "during the training process. The information of the parent table is present "
        "in the metadata storage, but the parent table hasn't been trained previously "
        "that's why the success file of the parent table is absent"
    )
    metadata = {
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv",
            },
            "keys": {
                "fk_key": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            validator = Validator(
                metadata=metadata,
                type_of_process="train",
                metadata_path=FAKE_METADATA_PATH,
                loader=None
            )
            validator.run()
            mock_get_columns_of_source.assert_called_once_with(
                path_to_source="path/to/table_b.csv", table_name="table_b"
            )
            mock_get_columns_by_loader.assert_not_called()
            mock_check_existence_of_key_columns.assert_called_once_with("table_b")
            mock_check_existence_of_referenced_columns.assert_called_once_with("table_b")
            mock_validate_referential_integrity.assert_called_once_with(
                fk_name="fk_key",
                fk_config={
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                },
                parent_metadata={
                    "train_settings": {
                        "source": "path/to/table_a.csv",
                        "reports": ["accuracy", "sample"]
                    },
                    "infer_settings": {
                        "destination": "path/to/generated_table_a.csv"
                    },
                    "keys": {
                        "pk_id": {
                            "type": "PK",
                            "columns": ["id"]
                        },
                        "uq_id": {
                            "type": "UQ",
                            "columns": ["name"]
                        }
                    }
                }
            )
            mock_check_completion_of_training.assert_called_once_with("table_a")
            mock_check_existence_of_generated_data.assert_not_called()
            assert validator.mapping == {
                "fk_key": {
                    "parent_table": "table_a",
                    "parent_columns": ["id"]
                }
            }
            assert validator.merged_metadata == {
                "table_b": {
                    "train_settings": {
                        "source": "path/to/table_b.csv",
                    },
                    "keys": {
                        "fk_key": {
                            "type": "FK",
                            "columns": ["id"],
                            "references": {
                                "table": "table_a",
                                "columns": ["id"]
                            }
                        }
                    }
                },
                "table_a": {
                    "train_settings": {
                        "source": "path/to/table_a.csv",
                        "reports": ["accuracy", "sample"]
                    },
                    "infer_settings": {
                        "destination": "path/to/generated_table_a.csv"
                    },
                    "keys": {
                        "pk_id": {
                            "type": "PK",
                            "columns": ["id"]
                        },
                        "uq_id": {
                            "type": "UQ",
                            "columns": ["name"]
                        }
                    }
                }
            }
            message = (
                "The validation of the metadata has been failed. The error(s) found in - "
                "\"check completion of the training process\": "
                "{\"table_a\": \"The training of the table  - 'table_a'"
                "hasn't been completed. Please, retrain the table.\"}"
            )
            assert str(error.value) == message
            assert message in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
def test_validate_incomplete_metadata_with_absent_gen_data_of_parent_table_in_infer_process(
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_check_existence_of_generated_data,
    mock_validate_referential_integrity,
    test_metadata_storage,
    caplog,
    rp_logger
):
    """
    Test the validation of the incomplete metadata of one table contained the foreign key
    during the inference process.
    The information of the parent table is present in the metadata storage,
    but the generated data of the parent table hasn't been generated previously
    """
    rp_logger.info(
        "Test the validation of the incomplete metadata of one table contained the foreign key "
        "during the inference process. The information of the parent table is present "
        "in the metadata storage, but the generated data of the parent table hasn't been "
        "generated previously"
    )
    metadata = {
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv",
            },
            "keys": {
                "fk_key": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            validator = Validator(
                metadata=metadata,
                type_of_process="infer",
                metadata_path=FAKE_METADATA_PATH,
                loader=None
            )
            validator.run()
            assert mock_check_existence_of_destination.call_count == 2
            assert validator.mapping == {
                "fk_key": {
                    "parent_table": "table_a",
                    "parent_columns": ["id"]
                }
            }
            assert validator.merged_metadata == {
                "table_b": {
                    "train_settings": {
                        "source": "path/to/table_b.csv",
                    },
                    "keys": {
                        "fk_key": {
                            "type": "FK",
                            "columns": ["id"],
                            "references": {
                                "table": "table_a",
                                "columns": ["id"]
                            }
                        }
                    }
                },
                "table_a": {
                    "train_settings": {
                        "source": "path/to/table_a.csv",
                        "reports": ["accuracy", "sample"]
                    },
                    "infer_settings": {
                        "destination": "path/to/generated_table_a.csv"
                    },
                    "keys": {
                        "pk_id": {
                            "type": "PK",
                            "columns": ["id"]
                        },
                        "uq_id": {
                            "type": "UQ",
                            "columns": ["name"]
                        }
                    },
                    "format": {}
                }
            }
            assert mock_check_existence_of_destination.assert_called_once_with("table_b")
            mock_validate_referential_integrity.assert_called_once_with(
                fk_name="fk_key",
                fk_config={
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                },
                parent_config={
                    "train_settings": {
                        "source": "path/to/table_a.csv",
                        "reports": ["accuracy", "sample"]
                    },
                    "infer_settings": {
                        "destination": "path/to/generated_table_a.csv"
                    },
                    "keys": {
                        "pk_id": {
                            "type": "PK",
                            "columns": ["id"]
                        },
                        "uq_id": {
                            "type": "UQ",
                            "columns": ["name"]
                        }
                    },
                    "format": {}
                }
            )
            mock_check_existence_of_generated_data.assert_called_once_with("table_a")
            message = (
                "The validation of the metadata has been failed. "
                "The error(s) found in - \"check existence of the generated data\": {"
                "\"table_a\": \"The generated data of the table - 'table_a' "
                "hasn't been generated. "
                "Please, generate the data related to the table - 'table_a' first\"}"
            )
            assert str(error.value) == message
            assert message in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


@pytest.mark.parametrize("value", [
    ["accuracy", "sample"],
    ["accuracy", "metrics_only"],
    ["sample", "metrics_only"],
    ["accuracy"],
    ["metrics_only"],
])
@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_incomplete_metadata_without_gen_parent_table_in_train_process_with_reports(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_referential_integrity,
    mock_check_completion_of_training,
    mock_check_existence_of_generated_data,
    test_metadata_storage,
    caplog,
    value,
    rp_logger
):
    """
    Test the validation of the incomplete metadata of one table contained the foreign key
    during the training process with the generation of reports that requires
    the generation of the synthetic data.
    The information of the parent table is present in the metadata storage,
    but the generated data of the parent table hasn't been generated previously
    """
    rp_logger.info(
        "Test the validation of the incomplete metadata of one table "
        "contained the foreign key during the training process with "
        "the generation of reports that requires the generation of the synthetic data. "
        "The information of the parent table is present in the metadata storage, "
        "but the generated data of the parent table hasn't been generated previously"
    )
    metadata = {
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv",
                "reports": value
            },
            "keys": {
                "fk_key": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        with caplog.at_level("ERROR"):
            validator = Validator(
                metadata=metadata,
                type_of_process="train",
                metadata_path=FAKE_METADATA_PATH,
                loader=None
            )
            validator.run()
            assert validator.mapping == {
                "fk_key": {
                    "parent_table": "table_a",
                    "parent_columns": ["id"]
                }
            }
            assert validator.merged_metadata == {
                "table_b": {
                    "train_settings": {
                        "source": "path/to/table_b.csv",
                    },
                    "keys": {
                        "fk_key": {
                            "type": "FK",
                            "columns": ["id"],
                            "references": {
                                "table": "table_a",
                                "columns": ["id"]
                            }
                        }
                    }
                },
                "table_a": {
                    "train_settings": {
                        "source": "path/to/table_a.csv",
                        "reports": ["accuracy", "sample"]
                    },
                    "infer_settings": {
                        "destination": "path/to/generated_table_a.csv"
                    },
                    "keys": {
                        "pk_id": {
                            "type": "PK",
                            "columns": ["id"]
                        },
                        "uq_id": {
                            "type": "UQ",
                            "columns": ["name"]
                        }
                    },
                    "format": {}
                }
            }
            mock_get_columns_of_source.assert_called_once_with(
                path_to_source="path/to/table_b.csv", table_name="table_b"
            )
            mock_get_columns_by_loader.assert_not_called()
            mock_check_existence_of_key_columns.assert_called_once_with("table_b")
            mock_check_existence_of_referenced_columns.assert_called_once_with("table_b")
            mock_validate_referential_integrity.assert_called_once_with("table_b")
            mock_check_completion_of_training.assert_called_once_with("table_a")
            mock_check_existence_of_generated_data.assert_called_once_with("table_a")
            message = (
                "The validation of the metadata has been failed. "
                "The error(s) found in - \"check existence of the generated data\": {"
                "\"table_a\": \"The generated data of the table - 'table_a' "
                "hasn't been generated. Please, generate the data related "
                "to the table 'table_a' first\"}"
            )
            assert str(error.value) == message
            assert message in caplog.text
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_check_not_existent_key_column_in_pk(rp_logger):
    """
    Test that the column of the primary key doesn't exist in the source table
    """
    rp_logger.info(
        "Test that the column of the primary key doesn't exist in the source table"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/table_with_data.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["non-existent column"]
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        validator = Validator(
            metadata=test_metadata,
            type_of_process="train",
            metadata_path=FAKE_METADATA_PATH,
            loader=None
        )
        validator.run()
        assert validator.mapping == {}
        assert validator.merged_metadata == test_metadata
        assert str(error.value) == (
            "The validation of the metadata has been failed. "
            "The error(s) found in - \n\"check existence of the key columns "
            "in 'columns'\": {\n    \"pk_id\": \"The columns "
            "of the PK 'pk_id' - 'non-existent column' don\'t exist "
            "in the source of the table - 'table_a'\"\n}"
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_check_not_existent_key_column_in_uq(rp_logger):
    """
    Test that the column of the unique key doesn't exist in the source table
    """
    rp_logger.info(
        "Test that the column of the unique key doesn't exist in the source table"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/table_with_data.csv"
            },
            "keys": {
                "uq_id": {
                    "type": "UQ",
                    "columns": ["non-existent column"]
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        validator = Validator(
            metadata=test_metadata,
            type_of_process="train",
            metadata_path=FAKE_METADATA_PATH,
            loader=None
        )
        validator.run()
        assert validator.mapping == {}
        assert validator.merged_metadata == test_metadata
        assert str(error.value) == (
            "The validation of the metadata has been failed. "
            "The error(s) found in - \n\"check existence of the key columns "
            "in 'columns'\": {\n    \"uq_id\": \"The columns "
            "of the UQ 'uq_id' - 'non-existent column' don\'t exist "
            "in the source of the table - 'table_a'\"\n}"
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_check_not_existent_key_column_in_fk(rp_logger):
    """
    Test that the column of the foreign key doesn't exist in the child table
    """
    rp_logger.info(
        "Test that the column of the foreign key doesn't exist in the child table"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/table_with_data.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        },
        "table_b": {
            "train_settings": {
                "source": f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/"
                          "child_table_with_data.csv"
            },
            "keys": {
                "fk_id": {
                    "type": "FK",
                    "columns": ["non-existent column"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        validator = Validator(
            metadata=test_metadata,
            type_of_process="train",
            metadata_path=FAKE_METADATA_PATH,
            loader=None
        )
        validator.run()
        assert validator.mapping == {}
        assert validator.merged_metadata == test_metadata
        assert str(error.value) == (
            "The validation of the metadata has been failed. "
            "The error(s) found in - \n\"check existence of the key columns in 'columns'\": {"
            "\n    \"fk_id\": \"The columns of the FK 'fk_id' - 'non-existent column' "
            "don\'t exist in the source of the table - \'table_b\'\"\n}"
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_check_not_existent_referenced_table_in_fk(test_metadata_storage, rp_logger):
    """
    Test that the referenced table of the foreign key doesn't exist in the metadata
    """
    rp_logger.info(
        "Test that the table of the foreign key doesn't exist in the metadata"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/table_with_data.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        },
        "table_b": {
            "train_settings": {
                "source": f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/"
                          "child_table_with_data.csv"
            },
            "keys": {
                "fk_id": {
                    "type": "FK",
                    "columns": ["non-existent column"],
                    "references": {
                        "table": "non-existent table",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        validator = Validator(
            metadata=test_metadata,
            type_of_process="train",
            metadata_path=FAKE_METADATA_PATH,
            loader=None
        )
        validator.errors = defaultdict(defaultdict)
        validator.run()
        assert validator.mapping == {}
        assert validator.merged_metadata == test_metadata
        assert str(error.value) == (
            "The metadata of the parent table - 'non-existent table' hasn't been found. "
            "Please, check whether the information of the parent table - 'non-existent table' "
            "exists in the current metadata file or in the metadata files stored in "
            "'model_artifacts/metadata' directory"
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


def test_check_not_existent_referenced_columns_in_fk(rp_logger):
    """
    Test that the referenced columns of the foreign key doesn't exist in the parent table
    """
    rp_logger.info(
        "Test that the referenced columns of the foreign key doesn't exist in the parent table"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/table_with_data.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            }
        },
        "table_b": {
            "train_settings": {
                "source": f"{DIR_NAME}/unit/data_loaders/fixtures/csv_tables/"
                          "child_table_with_data.csv"
                },
            "keys": {
                "fk_id": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["non-existent column"]
                    }
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        validator = Validator(
            metadata=test_metadata,
            type_of_process="train",
            metadata_path=FAKE_METADATA_PATH,
            loader=None
        )
        validator.run()
        assert validator.mapping == {}
        assert validator.merged_metadata == test_metadata
        assert str(error.value) == (
            "The validation of the metadata has been failed. The error(s) found in - \n"
            "\"check existence of the key columns in 'references.columns'\": {\n    \"fk_id\": "
            "\"The 'references.columns' of the FK 'fk_id' - 'non-existent column' don't exist "
            "in the referenced table - 'table_b'\"}"
        )
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_metadata_of_related_tables_with_missed_pk_key_in_parent_table_in_train_process(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    rp_logger
):
    """
    Test the validation of the metadata of related tables
    contained the parent table without mentioned PK key in the metadata
    during the training process
    """
    rp_logger.info(
        "Test the validation of the metadata of related tables "
        "contained the parent table without mentioned PK key "
        "in the metadata during the training process"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": "path/to/table_a.csv"
            },
        },
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv"
            },
            "keys": {
                "fk_id": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        validator = Validator(
            metadata=test_metadata,
            type_of_process="train",
            metadata_path=FAKE_METADATA_PATH,
            loader=None
        )
        validator.run()
        assert validator.mapping == {
            "fk_id": {
                "parent_table": "table_a",
                "parent_columns": ["id"]
            }
        }
        assert validator.merged_metadata == test_metadata
        assert (
            "The information about columns of the primary or unique key "
            "associated with the columns of the 'FK' key - 'fk_id' "
            "wasn't found in the metadata of the parent table"
        ) in str(error.value)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_metadata_of_related_tables_with_wrong_pk_key_in_parent_table_in_train_process(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_check_completion_of_training,
    rp_logger
):
    """
    Test the validation of the metadata of related tables
    contained the parent table with the wrong PK key in the metadata
    during the training process
    """
    rp_logger.info(
        "Test the validation of the metadata of related tables "
        "contained the parent table with the wrong PK key "
        "in the metadata during the training process"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": "path/to/table_a.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["wrong_id"]
                }
            }
        },
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv"
            },
            "keys": {
                "fk_id": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        validator = Validator(
            metadata=test_metadata,
            type_of_process="train",
            metadata_path=FAKE_METADATA_PATH,
            loader=None
        )
        validator.run()
        assert validator.mapping == {
            "fk_id": {
                "parent_table": "table_a",
                "parent_columns": ["id"]
            }
        }
        assert validator.merged_metadata == test_metadata
        assert (
            "The columns of primary or unique key associated with the columns of "
            "the 'FK' key - 'fk_id' aren't the same"
        ) in str(error.value)
        assert (
            "The columns of the PK 'pk_id' - 'wrong_id' don't exist in the table - 'pk_test'"
        ) in str(error.value)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_completion_of_training")
def test_validate_metadata_of_related_tables_with_missed_pk_key_in_parent_table_in_infer_process(
    mock_check_completion_of_training,
    rp_logger
):
    """
    Test the validation of the metadata of related tables
    contained the parent table without mentioned PK key in the metadata
    during the inference process
    """
    rp_logger.info(
        "Test the validation of the metadata of related tables "
        "contained the parent table without mentioned PK key "
        "in the metadata during the inference process"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": "path/to/table_a.csv"
            },
        },
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv"
            },
            "keys": {
                "fk_id": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        validator = Validator(
            metadata=test_metadata,
            type_of_process="infer",
            metadata_path=FAKE_METADATA_PATH,
            loader=None
        )
        validator.run()
        assert validator.mapping == {
            "fk_id": {
                "parent_table": "table_a",
                "parent_columns": ["id"]
            }
        }
        assert validator.merged_metadata == test_metadata
        assert (
            "The information about columns of the primary or unique key "
            "associated with the columns of the 'FK' key - 'fk_id' "
            "wasn't found in the metadata of the parent table"
        ) in str(error.value)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_completion_of_training")
def test_validate_metadata_of_related_tables_with_wrong_pk_key_in_parent_table_in_infer_process(
    mock_check_completion_of_training,
    rp_logger
):
    """
    Test the validation of the metadata of related tables
    contained the parent table with the wrong PK key in the metadata
    during the inference process
    """
    rp_logger.info(
        "Test the validation of the metadata of related tables "
        "contained the parent table with the wrong PK key "
        "in the metadata during the inference process"
    )
    test_metadata = {
        "table_a": {
            "train_settings": {
                "source": "path/to/table_a.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["wrong_id"]
                }
            }
        },
        "table_b": {
            "train_settings": {
                "source": "path/to/table_b.csv"
            },
            "keys": {
                "fk_id": {
                    "type": "FK",
                    "columns": ["id"],
                    "references": {
                        "table": "table_a",
                        "columns": ["id"]
                    }
                }
            }
        }
    }
    with pytest.raises(ValidationError) as error:
        validator = Validator(
            metadata=test_metadata,
            type_of_process="infer",
            metadata_path=FAKE_METADATA_PATH,
            loader=None
        )
        validator.run()
        assert validator.mapping == {
            "fk_id": {
                "parent_table": "table_a",
                "parent_columns": ["id"]
            }
        }
        assert validator.merged_metadata == test_metadata
        assert (
            "The columns of primary or unique key associated with the columns of "
            "the 'FK' key - 'fk_id' aren't the same"
        ) in str(error.value)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_metadata_if_valid_fernet_key_in_train_process(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_metadata,
    valid_fernet_key,
    rp_logger
):
    """
    Test the validation of the metadata of a table during training process
    if encryption is turned on and the valid Fernet key is provided
    """
    rp_logger.info(
        "Test the validation of the metadata of a table during training process "
        "if encryption is turned on and the valid Fernet key is provided"
    )
    test_metadata = {
        "table": {
            "train_settings": {
                "source": "path/to/table.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            },
            "encryption": {
                "fernet_key": valid_fernet_key
            }
        },
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="train",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    mock_get_columns_of_source.assert_called_once_with(
        path_to_source="path/to/table.csv", table_name="table"
    )
    mock_get_columns_by_loader.assert_not_called()
    mock_check_existence_of_key_columns.assert_called_once_with("table")
    mock_check_existence_of_referenced_columns.assert_called_once_with("table")
    mock_validate_metadata.assert_called_once_with("table")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_validate_metadata")
@patch.object(Validator, "_check_existence_of_referenced_columns")
@patch.object(Validator, "_check_existence_of_key_columns")
@patch.object(Validator, "_get_columns_by_loader")
@patch.object(Validator, "_get_columns_of_source")
def test_validate_metadata_if_invalid_fernet_key_in_train_process(
    mock_get_columns_of_source,
    mock_get_columns_by_loader,
    mock_check_existence_of_key_columns,
    mock_check_existence_of_referenced_columns,
    mock_validate_metadata,
    invalid_fernet_key,
    rp_logger
):
    """
    Test the validation of the metadata of a table during training process
    if encryption is turned on and the invalid Fernet key is provided
    """
    rp_logger.info(
        "Test the validation of the metadata of a table during training process "
        "if encryption is turned on and the invalid Fernet key is provided"
    )
    test_metadata = {
        "table": {
            "train_settings": {
                "source": "path/to/table.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            },
            "encryption": {
                "fernet_key": invalid_fernet_key
            }
        },
    }
    with pytest.raises(ValidationError) as error:
        validator = Validator(
            metadata=test_metadata,
            type_of_process="train",
            metadata_path=FAKE_METADATA_PATH,
            loader=None
        )
        validator.errors = defaultdict(defaultdict)
        validator.run()
        mock_get_columns_of_source.assert_called_once_with(
            path_to_source="path/to/table.csv", table_name="table"
        )
        mock_get_columns_by_loader.assert_not_called()
        mock_check_existence_of_key_columns.assert_called_once_with("table")
        mock_check_existence_of_referenced_columns.assert_called_once_with("table")
        mock_validate_metadata.assert_called_once_with("table")
    assert "Fernet key must be 32 url-safe base64-encoded bytes." in str(error.value)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
def test_validate_metadata_if_valid_fernet_key_with_generation_reports_in_infer_process(
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_check_access_to_input_data,
    valid_fernet_key,
    rp_logger
):
    """
    Test the validation of the metadata of a table during inference process
    if encryption is turned on, reports should be generated, and the valid Fernet key is provided
    """
    rp_logger.info(
        "Test the validation of the metadata of a table during inference process "
        "if encryption is turned on, reports should be generated, "
        "and the valid Fernet key is provided"
    )
    fernet_key = valid_fernet_key
    test_metadata = {
        "table": {
            "train_settings": {
                "source": "path/to/table.csv"
            },
            "infer_settings": {
                "reports": ["accuracy"],
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            },
            "encryption": {
                "fernet_key": fernet_key
            }
        },
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="infer",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    mock_check_completion_of_training.assert_called_once_with("table")
    mock_check_existence_of_destination.assert_called_once_with("table")
    mock_check_access_to_input_data.assert_called_once_with("table")
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_fernet_key")
def test_validate_metadata_if_valid_fernet_key_without_generation_reports_in_infer_process(
    mock_validate_fernet_key,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_check_access_to_input_data,
    valid_fernet_key,
    rp_logger
):
    """
    Test the validation of the metadata of a table during inference process
    if encryption is turned on, reports won't be generated, and the valid Fernet key is provided
    """
    rp_logger.info(
        "Test the validation of the metadata of a table during inference process "
        "if encryption is turned on, reports won't be generated, "
        "and the valid Fernet key is provided"
    )
    test_metadata = {
        "table": {
            "train_settings": {
                "source": "path/to/table.csv"
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            },
            "encryption": {
                "fernet_key": valid_fernet_key
            }
        },
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="infer",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    mock_validate_fernet_key.assert_not_called()
    mock_check_completion_of_training.assert_called_once_with("table")
    mock_check_existence_of_destination.assert_called_once_with("table")
    mock_check_access_to_input_data.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
def test_validate_metadata_if_invalid_fernet_key_in_infer_process_with_reports_generation(
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_check_access_to_input_data,
    invalid_fernet_key,
    rp_logger
):
    """
    Test the validation of the metadata of a table during inference process
    if encryption is turned on, reports should be generated, and the invalid Fernet key is provided
    """
    rp_logger.info(
        "Test the validation of the metadata of a table during inference process "
        "if encryption is turned on, reports should be generated, "
        "and the invalid Fernet key is provided"
    )
    test_metadata = {
        "table": {
            "train_settings": {
                "source": "path/to/table.csv"
            },
            "infer_settings": {
                "reports": ["accuracy", "metrics_only"],
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            },
            "encryption": {
                "fernet_key": invalid_fernet_key
            }
        },
    }
    with pytest.raises(ValidationError) as error:
        validator = Validator(
            metadata=test_metadata,
            type_of_process="infer",
            metadata_path=FAKE_METADATA_PATH,
            loader=None
        )
        validator.errors = defaultdict(defaultdict)
        validator.run()
        mock_check_completion_of_training.assert_called_once_with("table")
        mock_check_existence_of_destination.assert_called_once_with("table")
        mock_check_access_to_input_data.assert_not_called()
    assert "Fernet key must be 32 url-safe base64-encoded bytes." in str(error.value)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_fernet_key")
def test_validate_metadata_if_invalid_fernet_key_in_infer_process_without_reports_generation(
    mock_validate_fernet_key,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_check_access_to_input_data,
    invalid_fernet_key,
    rp_logger
):
    """
    Test the validation of the metadata of a table during inference process
    if encryption is turned on, reports won't be generated, and the invalid Fernet key is provided
    """
    rp_logger.info(
        "Test the validation of the metadata of a table during inference process "
        "if encryption is turned on, reports won't be generated, "
        "and the invalid Fernet key is provided"
    )
    test_metadata = {
        "table": {
            "train_settings": {
                "source": "path/to/table.csv"
            },
            "infer_settings": {
                "reports": [],
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            },
            "encryption": {
                "fernet_key": invalid_fernet_key
            }
        },
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="infer",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    mock_validate_fernet_key.assert_not_called()
    mock_check_completion_of_training.assert_called_once_with("table")
    mock_check_existence_of_destination.assert_called_once_with("table")
    mock_check_access_to_input_data.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch(
    "syngen.ml.config.validation.fetch_config",
    return_value=MagicMock(paths={"input_data_path": "path/to/nonexistent/input_data.dat"})
)
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
def test_validate_metadata_if_valid_fernet_key_in_infer_with_reports_and_absent_input_data(
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_fetch_config,
    valid_fernet_key,
    rp_logger
):
    """
    Test the validation of the metadata of a table during inference process
    if encryption is turned on, reports should be generated, the valid Fernet key is provided,
    and the input data is absent
    """
    rp_logger.info(
        "Test the validation of the metadata of a table during inference process "
        "if encryption is turned on, reports should be generated, "
        "the valid Fernet key is provided, and the input data is absent"
    )
    test_metadata = {
        "table": {
            "train_settings": {
                "source": "path/to/table.csv"
            },
            "infer_settings": {
                "reports": ["accuracy"],
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            },
            "encryption": {
                "fernet_key": valid_fernet_key
            }
        },
    }
    with pytest.raises(ValidationError) as error:
        validator = Validator(
            metadata=test_metadata,
            type_of_process="infer",
            metadata_path=FAKE_METADATA_PATH,
            loader=None
        )
        validator.errors = defaultdict(defaultdict)
        validator.run()
        mock_check_completion_of_training.assert_called_once_with("table")
        mock_check_existence_of_destination.assert_called_once_with("table")
    assert "No such file or directory: 'path/to/nonexistent/input_data.dat'" in str(error.value)
    rp_logger.info(SUCCESSFUL_MESSAGE)


@patch(
    "syngen.ml.config.validation.fetch_config",
    return_value=MagicMock(paths={"input_data_path": "path/to/nonexistent/input_data.dat"})
)
@patch.object(Validator, "_check_access_to_input_data")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_completion_of_training")
@patch.object(Validator, "_validate_fernet_key")
def test_validate_metadata_if_valid_fernet_key_in_infer_without_reports_and_absent_input_data(
    mock_validate_fernet_key,
    mock_check_completion_of_training,
    mock_check_existence_of_destination,
    mock_check_access_to_input_data,
    mock_fetch_config,
    valid_fernet_key,
    rp_logger
):
    """
    Test the validation of the metadata of a table during inference process
    if encryption is turned on, reports won't be generated, the valid Fernet key is provided,
    and the input data is absent
    """
    rp_logger.info(
        "Test the validation of the metadata of a table during inference process "
        "if encryption is turned on, reports won't be generated, "
        "the valid Fernet key is provided, and the input data is absent"
    )
    test_metadata = {
        "table": {
            "train_settings": {
                "source": "path/to/table.csv"
            },
            "infer_settings": {
                "reports": [],
            },
            "keys": {
                "pk_id": {
                    "type": "PK",
                    "columns": ["id"]
                }
            },
            "encryption": {
                "fernet_key": valid_fernet_key
            }
        },
    }
    validator = Validator(
        metadata=test_metadata,
        type_of_process="infer",
        metadata_path=FAKE_METADATA_PATH,
        loader=None
    )
    validator.errors = defaultdict(defaultdict)
    validator.run()
    mock_validate_fernet_key.assert_not_called()
    mock_check_completion_of_training.assert_called_once_with("table")
    mock_check_existence_of_destination.assert_called_once_with("table")
    mock_check_access_to_input_data.assert_not_called()
    rp_logger.info(SUCCESSFUL_MESSAGE)
