import os
from collections import Counter
from typing import List, Tuple, Dict
import json
from json import JSONDecodeError

from loguru import logger
import pandas as pd
import numpy as np
from flatten_json import flatten

from syngen.ml.data_loaders import DataLoader, MetadataLoader


class PreprocessHandler:
    """
    The class for the preprocessing of the data before the training process
    """
    def __init__(self, path_to_metadata: str):
        self.metadata_path = path_to_metadata
        self.metadata = MetadataLoader(self.metadata_path).load_data()
        self.data_loader = DataLoader.__name__

    @staticmethod
    def run_script():
        """
        Run the script before the training process
        if it exists in the predefined path
        """
        path_to_script = f"{os.getcwd()}/model_artifacts/script.py"
        if os.path.exists(path_to_script):
            os.system(f"python3 {path_to_script}")

    @staticmethod
    def get_json_columns(data: pd.DataFrame) -> List[str]:
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
    def get_flattened_df(data: pd.DataFrame, json_columns: List) -> Tuple[pd.DataFrame, Dict]:
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
        flattened_data.drop(columns=flattening_mapping.keys(), inplace=True)
        flattened_df = flattened_data.T.loc[~flattened_data.T.index.duplicated(), :].T
        flattened_df = flattened_df.applymap(lambda x: np.NaN if x in [list(), dict()] else x)
        return flattened_df, flattening_mapping

    @staticmethod
    def save_flatten_metadata(table_name: str, flattening_mapping: Dict, duplicated_columns: List):
        """
        Save the metadata of the flattening process
        """
        os.makedirs(f"{os.getcwd()}/model_artifacts", exist_ok=True)
        with open(f"{os.getcwd()}/model_artifacts/flatten_metadata.json", "a") as f:
            metadata = {
                table_name: {
                    "flattening_mapping": flattening_mapping,
                    "duplicated_columns": duplicated_columns
                }
            }
            json.dump(metadata, f)

    def handle_json_columns(self):
        """
        Preprocess the data contained JSON columns before the training process
        """
        for table, settings in self.metadata.items():
            if table == "global":
                continue
            source = settings.get("train_settings", {}).get("source", "")
            data, schema = globals()[self.data_loader](source).load_data()
            if json_columns := self.get_json_columns(data):
                logger.info(
                    f"The table '{table}' contains JSON columns: {', '.join(json_columns)}"
                )
                logger.info(f"Flattening the JSON columns in the table - '{table}'")
                flattened_data, flattening_mapping = self.get_flattened_df(data, json_columns)
                DataLoader(source).save_data(source, flattened_data)
                duplicated_columns = [
                    key
                    for key, value in dict(Counter(flattened_data.columns.to_list())).items()
                    if value > 1
                ]
                self.save_flatten_metadata(table, flattening_mapping, duplicated_columns)
                logger.info(f"The table '{table}' has been successfully flattened")

    def run(self):
        """
        Launch the preprocessing process:
        """
        self.run_script()
        self.handle_json_columns()
