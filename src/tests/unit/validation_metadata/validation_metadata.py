from unittest.mock import patch
from syngen.ml.config.validation import Validator


@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_source", return_value=True)
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
    validator = Validator(metadata=test_metadata, type_of_process="train")
    validator.run()
    assert validator.mapping == {}
    assert validator.merged_metadata == test_metadata
    mock_check_existence_of_source.assert_called_once()
    mock_validate_referential_integrity.assert_not_called()
    mock_check_existence_of_success_file.assert_not_called()


@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_source", return_value=True)
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
    validator = Validator(metadata=test_metadata, type_of_process="train")
    validator.run()
    assert validator.mapping == {}
    assert validator.merged_metadata == test_metadata
    assert mock_check_existence_of_source.call_count == 2
    mock_validate_referential_integrity.assert_not_called()
    mock_check_existence_of_success_file.assert_not_called()


@patch.object(Validator, "_check_existence_of_success_file")
@patch.object(Validator, "_validate_referential_integrity")
@patch.object(Validator, "_check_existence_of_source", return_value=True)
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
    validator = Validator(metadata=test_metadata, type_of_process="train")
    validator.run()
    assert validator.mapping == {
        "fk_id": {
            "child_columns": ["id"],
            "child_table": "table_b",
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
@patch.object(Validator, "_check_existence_of_source", return_value=True)
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
    validator = Validator(metadata=test_metadata, type_of_process="train")
    validator.run()
    assert validator.mapping == {
        "fk_1": {
            "child_columns": ["id"],
            "child_table": "table_b",
            "parent_columns": ["id"],
            "parent_table": "table_a"
        },
        "fk_2": {
            "child_columns": ["name"],
            "child_table": "table_b",
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
@patch.object(Validator, "_check_existence_of_destination", return_value=True)
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
    validator = Validator(metadata=test_metadata, type_of_process="infer")
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
@patch.object(Validator, "_check_existence_of_destination", return_value=True)
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
    validator = Validator(metadata=test_metadata, type_of_process="infer")
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
@patch.object(Validator, "_check_existence_of_destination", return_value=True)
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
    validator = Validator(metadata=test_metadata, type_of_process="infer")
    validator.run()
    assert validator.mapping == {
        "fk_id": {
            "child_columns": ["id"],
            "child_table": "table_b",
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
@patch.object(Validator, "_check_existence_of_destination", return_value=True)
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
    validator = Validator(metadata=test_metadata, type_of_process="infer")
    validator.run()
    assert validator.mapping == {
        "fk_1": {
            "child_columns": ["id"],
            "child_table": "table_b",
            "parent_columns": ["id"],
            "parent_table": "table_a"
        },
        "fk_2": {
            "child_columns": ["name"],
            "child_table": "table_b",
            "parent_columns": ["name"],
            "parent_table": "table_a"
        }
    }
    assert validator.merged_metadata == test_metadata
    assert mock_check_existence_of_destination.call_count == 2
    assert mock_validate_referential_integrity.call_count == 2
    mock_check_existence_of_success_file.assert_not_called()
    mock_check_existence_of_generated_data.assert_not_called()
