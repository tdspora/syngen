from unittest.mock import patch
import pytest

from syngen.ml.handlers import VaeInferHandler
from syngen.ml.data_loaders import MetadataLoader


@patch("os.path.exists", return_value=True)
@pytest.mark.parametrize("path_to_metadata, expected_path, type_of_process", [
    ("tests/unit/handlers/fixtures/metadata.yaml", "path/to/merged_infer_parent-table.csv", "train"),
    ("tests/unit/handlers/fixtures/metadata_with_absent_destination.yaml",
     "path/to/merged_infer_parent-table.csv", "train"),
    ("tests/unit/handlers/fixtures/metadata.yaml", "../data/parent_table_generated.csv", "infer"),
    ("tests/unit/handlers/fixtures/metadata_with_absent_destination.yaml",
     "model_artifacts/tmp_store/parent-table/merged_infer_parent-table.csv", "infer"),
])
def test_get_pk_path(mock_os_path_exists, path_to_metadata, expected_path, type_of_process):
    """
    Test set_pk method
    """
    with patch.object(VaeInferHandler, "__post_init__", lambda x: None):
        metadata = MetadataLoader(path_to_metadata).load_data()
        handler = VaeInferHandler(
            metadata=metadata,
            table_name="parent_table",
            paths={"path_to_merged_infer": "path/to/merged_infer_parent-table.csv"},
            metadata_path=path_to_metadata,
            random_seed=0,
            size=100,
            batch_size=100,
            run_parallel=False,
            print_report=False,
            wrapper_name="MMDVAEWrapper",
            log_level="INFO",
            type_of_process=type_of_process
        )
        assert handler._get_pk_path("parent_table", "child_table") == expected_path
