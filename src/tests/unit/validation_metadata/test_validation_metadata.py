from unittest.mock import patch

import pytest

from marshmallow import ValidationError
from syngen.ml.config.validation import Validator

FAKE_METADATA_PATH = "path/to/metadata.yaml"


@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_source")
def test_validate_metadata_of_one_table_without_fk_key(mock_check_existence_of_source,
                                                       mock_validate_referential_integrity,
                                                       mock_check_existence_of_success_file,
                                                       rp_logger):
    """
    Test the validation of the metadata of one table contained only the primary key used in the training process
    """
    rp_logger.info("Test the validation of the metadata of one table contained the primary key "
                   "used in the training process")
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
    validator = Validator(metadata=test_metadata, type_of_process="train", metadata_path=FAKE_METADATA_PATH)
    validator.run()
    assert validator.mapping == {}
    assert validator.merged_metadata == test_metadata
    mock_check_existence_of_source.assert_called_once()
    mock_validate_referential_integrity.assert_not_called()
    mock_check_existence_of_success_file.assert_not_called()


@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_source")
def test_validate_metadata_of_related_tables_without_fk_key(mock_check_existence_of_source,
                                                            mock_validate_referential_integrity,
                                                            mock_check_existence_of_success_file,
                                                            rp_logger):
    """
    Test the validation of the metadata of related tables
    contained only the primary key and the unique key
    used in the training process
    """
    rp_logger.info("Test the validation of the metadata of related tables "
                   "contained only the primary key and the unique key "
                   "used in the training process")
    test_metadata = {
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
    validator = Validator(metadata=test_metadata, type_of_process="train", metadata_path=FAKE_METADATA_PATH)
    validator.run()
    assert validator.mapping == {}
    assert validator.merged_metadata == test_metadata
    assert mock_check_existence_of_source.call_count == 2
    mock_validate_referential_integrity.assert_not_called()
    mock_check_existence_of_success_file.assert_not_called()


@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_source")
def test_validate_metadata_of_related_tables_with_fk_key(mock_check_existence_of_source,
                                                         mock_validate_referential_integrity,
                                                         mock_check_existence_of_success_file,
                                                         rp_logger):
    """
    Test the validation of the metadata of related tables
    contained the primary key and the foreign key
    used in the training process
    """
    rp_logger.info("Test the validation of the metadata of related tables "
                   "contained only the primary key and the foreign key "
                   "used in the training process")
    test_metadata = {
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
    validator = Validator(metadata=test_metadata, type_of_process="train", metadata_path=FAKE_METADATA_PATH)
    validator.run()
    assert validator.mapping == {
        "fk_id": {
            "parent_columns": ["id"],
            "parent_table": "table_a"
        }
    }
    assert validator.merged_metadata == test_metadata
    assert mock_check_existence_of_source.call_count == 2
    assert mock_validate_referential_integrity.call_count == 1
    mock_check_existence_of_success_file.assert_not_called()


@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_source")
def test_validate_metadata_of_related_tables_with_several_fk_key(mock_check_existence_of_source,
                                                                 mock_validate_referential_integrity,
                                                                 mock_check_existence_of_success_file,
                                                                 rp_logger):
    """
    Test the validation of the metadata of related tables
    contained several foreign keys
    used in the training process
    """
    rp_logger.info("Test the validation of the metadata of related tables contained several foreign keys "
                   "used in the training process")
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
    validator = Validator(metadata=test_metadata, type_of_process="train", metadata_path=FAKE_METADATA_PATH)
    validator.run()
    assert validator.mapping == {
        "fk_1": {
            "parent_columns": ["id"],
            "parent_table": "table_a"
        },
        "fk_2": {
            "parent_columns": ["name"],
            "parent_table": "table_a"
        }
    }
    assert validator.merged_metadata == test_metadata
    assert mock_check_existence_of_source.call_count == 2
    assert mock_validate_referential_integrity.call_count == 2
    mock_check_existence_of_success_file.assert_not_called()


@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_validate_referential_integrity")
def test_validate_metadata_of_one_table_without_fk_key(mock_validate_referential_integrity,
                                                       mock_check_existence_of_destination,
                                                       mock_check_existence_of_success_file,
                                                       mock_check_existence_of_generated_data,
                                                       rp_logger):
    """
    Test the validation of the metadata of one table
    contained the primary key used in the inference process
    """
    rp_logger.info("Test the validation of the metadata of one table contained the primary key "
                   "used in the inference process")
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
    validator = Validator(metadata=test_metadata, type_of_process="infer", metadata_path=FAKE_METADATA_PATH)
    validator.run()
    assert validator.mapping == {}
    assert validator.merged_metadata == test_metadata
    mock_check_existence_of_destination.assert_called_once()
    mock_validate_referential_integrity.assert_not_called()
    mock_check_existence_of_success_file.assert_not_called()
    mock_check_existence_of_generated_data.assert_not_called()


@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_destination")
def test_validate_metadata_of_related_tables_without_fk_key(mock_check_existence_of_destination,
                                                            mock_validate_referential_integrity,
                                                            mock_check_existence_of_success_file,
                                                            mock_check_existence_of_generated_data,
                                                            rp_logger):
    """
    Test the validation of the metadata of related tables
    contained only the primary key and the unique key
    used in the inference process
    """
    rp_logger.info("Test the validation of the metadata of related tables "
                   "contained only the primary key and the unique key "
                   "used in the inference process")
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
    validator = Validator(metadata=test_metadata, type_of_process="infer", metadata_path=FAKE_METADATA_PATH)
    validator.run()
    assert validator.mapping == {}
    assert validator.merged_metadata == test_metadata
    assert mock_check_existence_of_destination.call_count == 2
    mock_validate_referential_integrity.assert_not_called()
    mock_check_existence_of_success_file.assert_not_called()
    mock_check_existence_of_generated_data.assert_not_called()
    mock_check_existence_of_success_file.assert_not_called()


@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_destination")
def test_validate_metadata_of_related_tables_with_fk_key(mock_check_existence_of_destination,
                                                         mock_validate_referential_integrity,
                                                         mock_check_existence_of_success_file,
                                                         mock_check_existence_of_generated_data,
                                                         rp_logger):
    """
    Test the validation of the metadata of related tables
    contained the primary key and the foreign key
    used in the inference process
    """
    rp_logger.info("Test the validation of the metadata of related tables "
                   "contained only the primary key and the foreign key "
                   "used in the inference process")
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
    validator = Validator(metadata=test_metadata, type_of_process="infer", metadata_path=FAKE_METADATA_PATH)
    validator.run()
    assert validator.mapping == {
        "fk_id": {
            "parent_columns": ["id"],
            "parent_table": "table_a"
        }
    }
    assert validator.merged_metadata == test_metadata
    assert mock_check_existence_of_destination.call_count == 2
    assert mock_validate_referential_integrity.call_count == 1
    mock_check_existence_of_success_file.assert_not_called()
    mock_check_existence_of_generated_data.assert_not_called()


@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_destination")
def test_validate_metadata_of_related_tables_with_several_fk_key(mock_check_existence_of_destination,
                                                                 mock_validate_referential_integrity,
                                                                 mock_check_existence_of_success_file,
                                                                 mock_check_existence_of_generated_data,
                                                                 rp_logger):
    """
    Test the validation of the metadata of related tables
    contained several foreign keys
    used in the inference process
    """
    rp_logger.info("Test the validation of the metadata of related tables contained several foreign keys "
                   "used in the inference process")
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
    validator = Validator(metadata=test_metadata, type_of_process="infer", metadata_path=FAKE_METADATA_PATH)
    validator.run()
    assert validator.mapping == {
        "fk_1": {
            "parent_columns": ["id"],
            "parent_table": "table_a"
        },
        "fk_2": {
            "parent_columns": ["name"],
            "parent_table": "table_a"
        }
    }
    assert validator.merged_metadata == test_metadata
    assert mock_check_existence_of_destination.call_count == 2
    assert mock_validate_referential_integrity.call_count == 2
    mock_check_existence_of_success_file.assert_not_called()
    mock_check_existence_of_generated_data.assert_not_called()


@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_source")
def test_validate_incomplete_metadata_contained_fk_key_in_train_process_without_print_report(
        mock_check_existence_of_source,
        mock_validate_referential_integrity,
        mock_check_existence_of_success_file,
        test_metadata_storage,
        rp_logger):
    """
    Test the validation of the incomplete metadata of one table
    contained the foreign key but not contained the information of the parent table.
    It's used in the training process with the parameter 'print_report' set to False
    """
    rp_logger.info("Test the validation of the incomplete metadata of one table "
                   "contained the foreign key but not contained the information of the parent table. "
                   "It used in the training process with the parameter 'print_report' set to False")
    metadata = {
            "table_b": {
                "train_settings": {
                    "source": "path/to/table_b.csv",
                    "print_report": False
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
    validator = Validator(metadata=metadata, type_of_process="train", metadata_path=FAKE_METADATA_PATH)
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
                    "print_report": True
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
                "source": "path/to/table_b.csv",
                "print_report": False
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
    assert mock_check_existence_of_source.call_count == 2
    mock_validate_referential_integrity.assert_called_once()
    mock_check_existence_of_success_file.assert_called_once()


@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_source")
def test_validate_incomplete_metadata_contained_fk_key_in_train_process_with_print_report(
        mock_check_existence_of_source,
        mock_validate_referential_integrity,
        mock_check_existence_of_success_file,
        mock_check_existence_of_generated_data,
        test_metadata_storage,
        rp_logger):
    """
    Test the validation of the incomplete metadata of one table
    contained the foreign key but not contained the information of the parent table.
    It's used in the training process with the parameter 'print_report' set to True
    """
    rp_logger.info("Test the validation of the incomplete metadata of one table "
                   "contained the foreign key but not contained the information of the parent table. "
                   "It used in the training process with the parameter 'print_report' set to True")
    metadata = {
            "table_b": {
                "train_settings": {
                    "source": "path/to/table_b.csv",
                    "print_report": True
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
    validator = Validator(metadata=metadata, type_of_process="train", metadata_path=FAKE_METADATA_PATH)
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
                    "print_report": True
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
                "source": "path/to/table_b.csv",
                "print_report": True
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
    assert mock_check_existence_of_source.call_count == 2
    mock_validate_referential_integrity.assert_called_once()
    mock_check_existence_of_success_file.assert_called_once()
    mock_check_existence_of_generated_data.assert_called_once()


@patch.object(Validator, "_check_existence_of_generated_data")
@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_destination")
def test_validate_incomplete_metadata_in_infer_process(mock_check_existence_of_destination,
                                                       mock_validate_referential_integrity,
                                                       mock_check_existence_of_success_file,
                                                       mock_check_existence_of_generated_data,
                                                       test_metadata_storage,
                                                       rp_logger):
    """
    Test the validation of the incomplete metadata of one table
    contained the foreign key but not contained the information of the parent table.
    It's used in the inference process
    """
    rp_logger.info("Test the validation of the incomplete metadata of one table "
                   "contained the foreign key but not contained the information of the parent table. "
                   "It used in the inference process")
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
    validator = Validator(metadata=metadata, type_of_process="infer", metadata_path=FAKE_METADATA_PATH)
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
                    "print_report": True
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
    assert mock_check_existence_of_destination.call_count == 2
    mock_validate_referential_integrity.assert_called_once()
    mock_check_existence_of_success_file.assert_called_once()
    mock_check_existence_of_generated_data.assert_called_once()


@patch.object(Validator, "_validate_referential_integrity")
def test_validate_metadata_with_not_existent_source(mock_validate_referential_integrity, caplog, rp_logger):
    """
    Test the validation of the metadata of one table.
    The source of the table is not existent.
    """
    rp_logger.info("Test the validation of the metadata of one table. The source of the table is not existent.")
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
            validator = Validator(metadata=test_metadata, type_of_process="train", metadata_path=FAKE_METADATA_PATH)
            validator.run()
            assert validator.mapping == {}
            assert validator.merged_metadata == test_metadata
            mock_validate_referential_integrity.assert_called_once()
            message = "The validation of the metadata has been failed. The error(s) found in - " \
                      "\"check existence of the source\": {\"test_table\": \"It seems that the path " \
                      "to the source of the table - 'test_table' isn't correct. Please, check the path " \
                      "to the source of the table - 'test_table'\"}"
            assert str(error.value) == message
            assert message in caplog.text


@patch.object(Validator, "_check_existence_of_source")
def test_validate_incomplete_metadata_with_absent_parent_metadata_in_metadata_storage(
        mock_check_existence_of_source,
        test_metadata_storage,
        caplog,
        rp_logger):
    """
    Test the validation of the incomplete metadata of one table contained the foreign key.
    The information of the parent table is absent in the metadata storage.
    """
    rp_logger.info("Test the validation of the incomplete metadata of one table contained the foreign key. "
                   "The information of the parent table is absent in the metadata storage.")
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
            validator = Validator(metadata=metadata, type_of_process="train", metadata_path=FAKE_METADATA_PATH)
            validator.run()
            mock_check_existence_of_source.assert_called_once()
            assert validator.mapping == {
                "fk_key": {
                    "parent_columns": ["id"],
                    "parent_table": "table_a"
                }
            }
            message = "The metadata of the parent table - 'table_c' hasn't been found. Please, check " \
                      "whether the information of the parent table - '{parent_table}' exists in the current " \
                      "metadata file or in the metadata files stored in 'model_artifacts/metadata' directory"
            assert str(error.value) == message
            assert message in caplog.text


@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_check_existence_of_source")
def test_validate_incomplete_metadata_with_wrong_referential_integrity(
        mock_check_existence_of_source,
        mock_check_existence_of_success_file,
        test_metadata_storage,
        caplog,
        rp_logger):
    """
    Test the validation of the incomplete metadata of one table contained the foreign key.
    The information of the parent table is present in the metadata storage, but the parent table
    contains the PK which contains the list of columns which not correspond to the list of the columns
    of the FK.
    """
    rp_logger.info("Test the validation of the incomplete metadata of one table contained the foreign key. "
                   "The information of the parent table is present in the metadata storage, but the parent table "
                   "contains the PK which contains the list of columns which not correspond to the list of the "
                   "columns of the FK")
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
            validator = Validator(metadata=metadata, type_of_process="train", metadata_path=FAKE_METADATA_PATH)
            validator.run()
            assert validator.mapping == {
                "fk_key": {
                    "parent_columns": ["id"],
                    "parent_table": "table_d"
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
                        "print_report": True
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
            assert mock_check_existence_of_source.call_count == 2
            mock_check_existence_of_success_file.assert_called_once()
            message = "The validation of the metadata has been failed. " \
                      "The error(s) found in - \"validate referential integrity\": " \
                      "{\"fk_key\": \"The primary key columns associated with the columns " \
                      "of the foreign key - 'fk_key' is not the same\"}"
            assert str(error.value) == message
            assert message in caplog.text


@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_source")
def test_validate_incomplete_metadata_with_absent_success_file_of_parent_table_in_train_process(
        mock_check_existence_of_source,
        mock_validate_referential_integrity,
        test_metadata_storage,
        caplog,
        rp_logger
):
    """
    Test the validation of the incomplete metadata of one table contained the foreign key used in the training process.
    The information of the parent table is present in the metadata storage, but the parent table
    hasn't been trained previously that's why the success file of the parent table is absent
    """
    rp_logger.info("Test the validation of the incomplete metadata of one table contained the foreign key "
                   "used in the training process. The information of the parent table is present "
                   "in the metadata storage, but the parent table hasn't been trained previously "
                   "that's why the success file of the parent table is absent")
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
            validator = Validator(metadata=metadata, type_of_process="train", metadata_path=FAKE_METADATA_PATH)
            validator.run()
            assert validator.mapping == {
                "fk_key": {
                    "parent_columns": ["id"],
                    "parent_table": "table_a"
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
                        "print_report": True
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
            assert mock_check_existence_of_source.call_count == 2
            mock_validate_referential_integrity.assert_called_once()
            message = "The validation of the metadata has been failed. The error(s) found in - " \
                      "\"check existence of the success file\": {\"table_a\": \"The table - 'table_a'" \
                      "hasn't been trained completely. Please, retrain this table first\"}"
            assert str(error.value) == message
            assert message in caplog.text


@patch.object(Validator, "_validate_referential_integrity")
def test_validate_metadata_with_not_existent_destination(mock_validate_referential_integrity, caplog, rp_logger):
    """
    Test the validation of the metadata of one table used in the inference process.
    The destination for the generated data is not existent.
    """
    rp_logger.info("Test the validation of the metadata of one table used in the inference process. "
                   "The destination of the table is not existent.")
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
            validator = Validator(metadata=test_metadata, type_of_process="infer", metadata_path=FAKE_METADATA_PATH)
            validator.run()
            assert validator.mapping == {}
            assert validator.merged_metadata == test_metadata
            mock_validate_referential_integrity.assert_called_once()
            message = "The validation of the metadata has been failed. The error(s) found in - \"check existence " \
                      "of the destination\": {\"test_table\": \"It seems that the directory path for storing " \
                      "the generated data of table 'test_table' isn't correct. Please, verify the destination path\"}"
            assert str(error.value) == message
            assert message in caplog.text


@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_destination")
def test_validate_incomplete_metadata_with_absent_success_file_of_parent_table_in_infer_process(
        mock_check_existence_of_destination,
        mock_validate_referential_integrity,
        test_metadata_storage,
        caplog,
        rp_logger
):
    """
    Test the validation of the incomplete metadata of one table contained the foreign key
    used in the inference process. The information of the parent table is present in the metadata storage,
    but the parent table hasn't been trained previously that's why the success file of the parent table is absent
    """
    rp_logger.info("Test the validation of the incomplete metadata of one table contained the foreign key "
                   "used in the inference process. The information of the parent table is present "
                   "in the metadata storage, but the parent table hasn't been trained previously "
                   "that's why the success file of the parent table is absent")
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
            validator = Validator(metadata=metadata, type_of_process="train", metadata_path=FAKE_METADATA_PATH)
            validator.run()
            assert mock_check_existence_of_destination.call_count == 2
            mock_validate_referential_integrity.assert_called_once()
            assert validator.mapping == {
                "fk_key": {
                    "parent_columns": ["id"],
                    "parent_table": "table_a"
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
                        "print_report": True
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
            message = "The validation of the metadata has been failed. The error(s) found in - " \
                      "\"check existence of the success file\": {\"table_a\": \"The table - 'table_a'" \
                      "hasn't been trained completely. Please, retrain this table first\"}"
            assert str(error.value) == message
            assert message in caplog.text


@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_check_existence_of_destination")
def test_validate_incomplete_metadata_with_absent_generated_of_parent_table_in_infer_process(
        mock_check_existence_of_destination,
        mock_check_existence_of_success_file,
        mock_validate_referential_integrity,
        test_metadata_storage,
        caplog,
        rp_logger
):
    """
    Test the validation of the incomplete metadata of one table contained the foreign key
    used in the inference process. The information of the parent table is present in the metadata storage,
    but the generated data of the parent table hasn't been generated previously
    """
    rp_logger.info("Test the validation of the incomplete metadata of one table contained the foreign key"
                   "used in the inference process. The information of the parent table is present "
                   "in the metadata storage, but the generated data of the parent table hasn't been "
                   "generated previously")
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
                metadata_path=FAKE_METADATA_PATH
            )
            validator.run()
            assert mock_check_existence_of_destination.call_count == 2
            assert validator.mapping == {
                "fk_key": {
                    "parent_columns": ["id"],
                    "parent_table": "table_a"
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
                        "print_report": True
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
            assert mock_check_existence_of_destination.call_count == 2
            mock_validate_referential_integrity.assert_called_once()
            mock_check_existence_of_success_file.assert_called_once()
            message = "The validation of the metadata has been failed. The error(s) found in - \"check existence " \
                      "of the generated data\": {\"table_a\": \"The generated data of the table - 'table_a' " \
                      "hasn't been generated. Please, generate the data related to the table 'table_a' first\"}"
            assert str(error.value) == message
            assert message in caplog.text


@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_destination")
@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_check_existence_of_source")
def test_validate_incomplete_metadata_with_absent_generated_of_parent_table_in_train_process_with_print_report(
        mock_check_existence_of_source,
        mock_check_existence_of_success_file,
        mock_check_existence_of_destination,
        mock_validate_referential_integrity,
        test_metadata_storage,
        caplog,
        rp_logger
):
    """
    Test the validation of the incomplete metadata of one table contained the foreign key
    used in the training process with the parameter 'print_report' set to True.
    The information of the parent table is present in the metadata storage,
    but the generated data of the parent table hasn't been generated previously
    """
    rp_logger.info("Test the validation of the incomplete metadata of one table contained the foreign key"
                   "used in the training process with the parameter 'print_report' set to True. "
                   "The information of the parent table is present in the metadata storage, "
                   "but the generated data of the parent table hasn't been generated previously")
    metadata = {
            "table_b": {
                "train_settings": {
                    "source": "path/to/table_b.csv",
                    "print_report": True
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
                metadata_path=FAKE_METADATA_PATH
            )
            validator.run()
            assert mock_check_existence_of_destination.call_count == 2
            assert validator.mapping == {
                "fk_key": {
                    "parent_columns": ["id"],
                    "parent_table": "table_a"
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
                        "print_report": True
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
            assert mock_check_existence_of_source.call_count == 2
            assert mock_check_existence_of_destination.call_count == 2
            mock_check_existence_of_success_file.assert_called_once()
            mock_validate_referential_integrity.assert_called_once()
            message = "The validation of the metadata has been failed. The error(s) found in - \"check existence " \
                      "of the generated data\": {\"table_a\": \"The generated data of the table - 'table_a' " \
                      "hasn't been generated. Please, generate the data related to the table 'table_a' first\"}"
            assert str(error.value) == message
            assert message in caplog.text