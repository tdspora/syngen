import os
from collections import Counter
from typing import List, Tuple, Dict
import json
from json import JSONDecodeError
import shutil
from slugify import slugify

from loguru import logger
import pandas as pd
import numpy as np
from flatten_json import flatten

from syngen.ml.data_loaders import DataLoader, MetadataLoader
from syngen.ml.utils import fetch_unique_root


PATH_TO_MODEL_ARTIFACTS = f"{os.getcwd()}/model_artifacts"


class PreprocessHandler:
    """
    The class for the preprocessing of the data before the training process
    """
    def __init__(self, path_to_metadata: str):
        self.metadata_path = path_to_metadata
        self.path_to_flatten_metadata = (
            f"{PATH_TO_MODEL_ARTIFACTS}/tmp_store/flatten_configs/"
            f"flatten_metadata_{fetch_unique_root(None, self.metadata_path)}.json"
        )

    @staticmethod
    def _remove_existed_artifacts(table_name: str):
        """
        Remove existed artifacts from previous train process
        """
        resources_path = f"{PATH_TO_MODEL_ARTIFACTS}/resources/{slugify(table_name)}/"
        tmp_store_path = f"{PATH_TO_MODEL_ARTIFACTS}/tmp_store/{slugify(table_name)}/"
        if os.path.exists(resources_path):
            shutil.rmtree(resources_path)
            logger.info(
                f"The artifacts located in the path - '{resources_path}' "
                f"were removed"
            )
        if os.path.exists(tmp_store_path):
            shutil.rmtree(tmp_store_path)
            logger.info(
                f"The artifacts located in the path - '{tmp_store_path}' "
                f"were removed"
            )

    @staticmethod
    def _prepare_dirs(table_name: str):
        """
        Create main directories for saving original, synthetic data and model artifacts
        """
        resources_path = f"{PATH_TO_MODEL_ARTIFACTS}/resources/{slugify(table_name)}/"
        tmp_store_path = f"{PATH_TO_MODEL_ARTIFACTS}/tmp_store/{slugify(table_name)}/"
        state_path = f"{PATH_TO_MODEL_ARTIFACTS}/resources/{slugify(table_name)}/vae/checkpoints"
        flatten_config_path = f"{PATH_TO_MODEL_ARTIFACTS}/tmp_store/flatten_configs/"
        os.makedirs(resources_path, exist_ok=True)
        os.makedirs(tmp_store_path, exist_ok=True)
        os.makedirs(state_path, exist_ok=True)
        os.makedirs(flatten_config_path, exist_ok=True)

    @staticmethod
    def _run_script():
        """
        Run the script before the training process
        if it exists in the predefined path
        """
        path_to_script = f"{os.getcwd()}/model_artifacts/script.py"
        if os.path.exists(path_to_script):
            os.system(f"python3 {path_to_script}")

    @staticmethod
    def _get_json_columns(data: pd.DataFrame) -> List[str]:
        """
        Get the list of columns which contain JSON data
        """
        json_columns = list()
        for column in data.columns.to_list():
            try:
                if pd.isnull(data[column]).all():
                    continue
                data[column].dropna().apply(lambda v: json.loads(v))
                json_columns.append(column)
            except (TypeError, JSONDecodeError):
                continue
        return json_columns

    @staticmethod
    def _get_flattened_df(data: pd.DataFrame, json_columns: List) -> Tuple[pd.DataFrame, Dict, List]:
        """
        Flatten the JSON columns in the dataframe
        """
        df_list = list()
        flattening_mapping = dict()
        for column in json_columns:
            flattened_data = pd.DataFrame(
                [
                    flatten(json.loads(i), ".")
                    for i in data[column]
                ]
            )
            flattening_mapping[column] = flattened_data.columns.to_list()
            df_list.append(flattened_data)
        flattened_data = pd.concat([data, *df_list], axis=1)
        duplicated_columns = [
            key
            for key, value in dict(Counter(flattened_data.columns.to_list())).items()
            if value > 1
        ]
        flattened_data.drop(columns=flattening_mapping.keys(), inplace=True)
        flattened_df = flattened_data.T.loc[~flattened_data.T.index.duplicated(), :].T
        flattened_df = flattened_df.applymap(lambda x: np.NaN if x in [list(), dict()] else x)
        return flattened_df, flattening_mapping, duplicated_columns

    def _save_flatten_metadata(self, metadata: Dict):
        """
        Save the metadata of the flattening process
        """
        with open(f"{self.path_to_flatten_metadata}", "w") as f:
            json.dump(metadata, f)

    def _handle_json_columns(self):
        """
        Preprocess the data contained JSON columns before the training process
        """
        metadata = MetadataLoader(self.metadata_path).load_data()
        flatten_metadata = dict()
        for table, settings in metadata.items():
            if table == "global":
                continue
            path_to_input_data = (f"{PATH_TO_MODEL_ARTIFACTS}/tmp_store/{slugify(table)}/"
                                  f"input_data_{slugify(table)}.pkl")
            source = settings.get("train_settings", {}).get("source", "")
            data, schema = DataLoader(source).load_data()
            self._remove_existed_artifacts(table_name=table)
            self._prepare_dirs(table)
            if json_columns := self._get_json_columns(data):
                logger.info(
                    f"The table '{table}' contains JSON columns: {', '.join(json_columns)}"
                )
                logger.info(f"Flattening the JSON columns in the table - '{table}'")
                (flattened_data,
                 flattening_mapping,
                 duplicated_columns) = self._get_flattened_df(data, json_columns)
                DataLoader(path_to_input_data).save_data(path_to_input_data, flattened_data)
                flatten_metadata[table] = {
                    "flattening_mapping": flattening_mapping,
                    "duplicated_columns": duplicated_columns
                }
                logger.info(f"The table '{table}' has been successfully flattened")
        if flatten_metadata:
            self._save_flatten_metadata(flatten_metadata)

    def run(self):
        """
        Launch the preprocessing process:
        """
        self._run_script()
        self._handle_json_columns()
