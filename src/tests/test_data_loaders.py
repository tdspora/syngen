from unittest import TestCase
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from syngen.ml.data_loaders import (
    DataLoader,
    CSVLoader,
    AvroLoader
)


class TestDataLoader(TestCase):
    def test_initialize_data_loader_for_single_csv_table_with_existed_path(self):
        test_data_loader = DataLoader("tests/fixtures/csv_tables/table_with_data.csv")
        assert test_data_loader.path == "tests/fixtures/csv_tables/table_with_data.csv"
        assert test_data_loader.has_existed_path is True
        assert isinstance(test_data_loader.file_loader, CSVLoader)

    def test_initialize_data_loader_for_single_csv_table_with_not_existed_path(self):
        test_data_loader = DataLoader("path/to/table.csv")
        assert test_data_loader.path == "path/to/table.csv"
        assert test_data_loader.has_existed_path is False
        assert isinstance(test_data_loader.file_loader, CSVLoader)

    def test_initialize_data_loader_for_single_avro_table_with_existed_path(self):
        test_data_loader = DataLoader("tests/fixtures/avro_tables/table_with_data.avro")
        assert test_data_loader.path == "tests/fixtures/avro_tables/table_with_data.avro"
        assert test_data_loader.has_existed_path is True
        assert isinstance(test_data_loader.file_loader, AvroLoader)

    def test_initialize_data_loader_for_single_avro_table_with_not_existed_path(self):
        test_data_loader = DataLoader("path/to/table.avro")
        assert test_data_loader.path == "path/to/table.avro"
        assert test_data_loader.has_existed_path is False
        assert isinstance(test_data_loader.file_loader, AvroLoader)

    def test_initialize_data_loader_for_single_table_with_unsupported_format(self):
        with pytest.raises(NotImplementedError) as error:
            DataLoader("path/to/table.test")
            assert str(error.value) == "File format not supported"

    def test_initialize_data_loader_for_single_table_with_empty_path(self):
        with pytest.raises(ValueError):
            with self.assertLogs(level="ERROR") as captured_log:
                DataLoader("")
            self.assertIn(
                captured_log.output, "It seems that the information of source is absent"
            )

    def test_load_data_from_table_in_csv_format(self):
        df, schema = DataLoader("tests/fixtures/csv_tables/table_with_data.csv").load_data()
        assert assert_frame_equal(
            df, pd.DataFrame(
                   {"gender": [0, 1, 0, 1],
                    "height": [157.18518021548246, 166.7731072622863, 162.91821942384928, 173.51448996432848],
                    "id": [925, 84, 821, 383]}
                )
        ) is None

        assert isinstance(df, pd.DataFrame)
        assert schema == {"format": "CSV"}

    def test_load_data_from_empty_table_in_csv_format(self):
        with pytest.raises(ValueError):
            with self.assertLogs(level="ERROR") as captured_log:
                DataLoader("tests/fixtures/csv_tables/empty_table.csv").load_data()
            self.assertIn(
                captured_log.output, "It seems that empty file was provided. Unable to train"
            )

    def test_load_data_from_table_in_csv_format_in_not_utf_8(self):
        path = "tests/fixtures/csv_tables/table_in_iso_encoding.csv"
        with pytest.raises(ValueError) as error:
            with self.assertLogs(level="ERROR") as captured_log:
                DataLoader(path).load_data()
            self.assertIn(
                captured_log.output,
                f"It seems that the content of the data in the path - '{path}' "
                f"doesn't have the encoding UTF-8. The details of the error - {error}.\n"
                f"Please, use the data in UTF-8 encoding"
            )

    def test_load_data_from_table_in_avro_format(self):
        df, schema = DataLoader("tests/fixtures/avro_tables/table_with_data.avro").load_data()
        assert assert_frame_equal(
            df, pd.DataFrame(
                   {"gender": [0, 1, 0, 1],
                    "height": [157.18518021548246, 166.7731072622863, 162.91821942384928, 173.51448996432848],
                    "id": [925, 84, 821, 383]}
                )
        ) is None

        assert isinstance(df, pd.DataFrame)
        assert schema == {'fields': {'gender': 'int', 'height': 'float', 'id': 'int'}, 'format': 'Avro'}

    def test_load_data_from_empty_table_in_avro_format(self):
        with pytest.raises(ValueError):
            with self.assertLogs(level="ERROR") as captured_log:
                DataLoader("tests/fixtures/avro_tables/empty_table.avro").load_data()
            self.assertIn(
                captured_log.output, "It seems that empty file was provided. Unable to train."
            )