import os
import shutil
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional
import json
from json import JSONDecodeError
from slugify import slugify

from loguru import logger
import pandas as pd
import numpy as np
from flatten_json import flatten, unflatten_list

from syngen.ml.data_loaders import DataLoader, MetadataLoader
from syngen.ml.utils import fetch_unique_root
from syngen.ml.context import get_context


PATH_TO_MODEL_ARTIFACTS = f"{os.getcwd()}/model_artifacts"


class Processor:
    """
    The base class for the preprocessing and postprocessing of the data
    """
    def __init__(
        self,
        path_to_metadata: Optional[str],
        table_name: Optional[str],
        settings: Dict
    ):
        self.metadata_path = path_to_metadata
        self.table_name = table_name
        self.settings = settings
        self.metadata = self._fetch_metadata()
        self.path_to_flatten_metadata = (
            f"{PATH_TO_MODEL_ARTIFACTS}/tmp_store/flatten_configs/"
            f"flatten_metadata_{fetch_unique_root(self.table_name, self.metadata_path)}.json"
        )

    def _fetch_metadata(self) -> Dict:
        metadata = dict()
        if self.metadata_path:
            metadata = MetadataLoader(self.metadata_path).load_data()
        if self.table_name:
            metadata = {
                self.table_name: {
                    "train_settings": {
                        "source": self.settings.get("source", "")
                    }
                }
            }
        return metadata


class PreprocessHandler(Processor):
    """
    The class for the preprocessing of the data before the training process
    """
    def __init__(
        self,
        path_to_metadata: Optional[str],
        table_name: Optional[str],
        settings: Dict
    ):
        super().__init__(path_to_metadata, table_name, settings)

    @staticmethod
    def _remove_existed_artifact(path_to_artifact: str):
        """
        Remove the existed artifact from the previous train process
        """
        if os.path.exists(path_to_artifact):
            shutil.rmtree(path_to_artifact)
            logger.info(f"The artifacts located in the path - '{path_to_artifact}' was removed")

    def _remove_existed_artifacts(self, table_name: str):
        """
        Remove existed artifacts from previous train process
        """
        resources_path = f"{PATH_TO_MODEL_ARTIFACTS}/resources/{slugify(table_name)}/"
        tmp_store_path = f"{PATH_TO_MODEL_ARTIFACTS}/tmp_store/{slugify(table_name)}/"
        self._remove_existed_artifact(resources_path)
        self._remove_existed_artifact(tmp_store_path)

    @staticmethod
    def _prepare_dirs(table_name: str):
        """
        Create main directories for saving original, synthetic data and model artifacts
        """
        resources_path = f"{PATH_TO_MODEL_ARTIFACTS}/resources/{slugify(table_name)}/"
        tmp_store_path = f"{PATH_TO_MODEL_ARTIFACTS}/tmp_store/{slugify(table_name)}/"
        state_path = (
            f"{PATH_TO_MODEL_ARTIFACTS}/resources/{slugify(table_name)}/vae/checkpoints"
        )
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
    def _get_flattened_df(
        data: pd.DataFrame,
        json_columns: List,
    ) -> Tuple[pd.DataFrame, Dict, List]:
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

    @staticmethod
    def _load_source(source: str, *args):
        """
        Load the data from the predefined source
        """
        return DataLoader(source).load_data()

    @staticmethod
    def _save_input_data(flattened_data: pd.DataFrame, table_name: str):
        """
        Save the input data to the predefined path
        """
        path_to_input_data = (
            f"{PATH_TO_MODEL_ARTIFACTS}/tmp_store/{slugify(table_name)}/"
            f"input_data_{slugify(table_name)}.pkl"
        )
        DataLoader(path_to_input_data).save_data(flattened_data)

    def _handle_json_columns(self):
        """
        Preprocess the data contained JSON columns before the training process
        """
        flatten_metadata = dict()
        for table, settings in self.metadata.items():
            if table == "global":
                continue
            source = settings.get("train_settings", {}).get("source", "")
            data, schema = self._load_source(source, table)
            order_of_columns = data.columns.to_list()
            self._remove_existed_artifacts(table)
            self._prepare_dirs(table)
            if json_columns := self._get_json_columns(data):
                logger.info(
                    f"The table '{table}' contains JSON columns: {', '.join(json_columns)}"
                )
                logger.info(f"Flattening the JSON columns in the table - '{table}'")
                (flattened_data,
                 flattening_mapping,
                 duplicated_columns) = self._get_flattened_df(data, json_columns)
                self._save_input_data(flattened_data, table)
                flatten_metadata[table] = {
                    "flattening_mapping": flattening_mapping,
                    "duplicated_columns": duplicated_columns,
                    "order_of_columns": order_of_columns
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


class PostprocessHandler(Processor):
    """
    Class for the postprocessing of the generated data
    """

    def __init__(
        self,
        metadata_path: Optional[str],
        table_name: Optional[str],
        settings: Dict
    ):
        super().__init__(metadata_path, table_name, settings)

    def _fetch_flatten_config(self, table_name: str) -> Dict:
        """
        Fetch the metadata of the flattening process
        """
        with open(self.path_to_flatten_metadata, "r") as f:
            flatten_metadata = json.load(f).get(table_name)
            return flatten_metadata

    @staticmethod
    def _check_none_values(x) -> bool:
        """
        Check if the value is None or np.NaN
        """
        return x is None or (isinstance(x, float) and np.isnan(x))

    def _remove_none_from_struct(self, input_data: Any):
        """
        Remove np.NaN or None values from the nested structure
        """
        if isinstance(input_data, list):
            output = [
                self._remove_none_from_struct(v)
                for v in input_data
                if not self._check_none_values(v) and v != {}
            ]
            output = [v for v in output if v != {}]
            return output
        elif isinstance(input_data, dict):
            output = {
                k: self._remove_none_from_struct(v)
                for k, v in input_data.items()
                if not self._check_none_values(v) and v != {}
            }
            return output
        else:
            return input_data

    @staticmethod
    def _restore_empty_values(df: pd.DataFrame):
        """
        Restore the empty dictionary values in the dataframe for nested fields
        """
        df = df.astype(object)
        for i, row in df.iterrows():
            for pos_col, col in enumerate(df.columns.to_list()):
                if pd.isna(df.iloc[i, pos_col]):
                    columns_set = set(df.columns)
                    columns_set.remove(col)
                    nested_fields = [
                        "".join(i.split(f"{col}.")) for i in columns_set if f"{col}." in i
                    ]

                    if nested_fields:
                        df.at[i, col] = dict()
        return df

    @staticmethod
    def _load_generated_data(
        path_to_generated_data: str,
        *args
    ) -> pd.DataFrame:
        """
        Load generated data from the predefined path
        """
        data, schema = DataLoader(path_to_generated_data).load_data()
        return data

    def _post_process_generated_data(
            self,
            data: pd.DataFrame,
            flattening_mapping: Dict,
            duplicated_columns: List
    ) -> pd.DataFrame:
        """
        Postprocess the generated data
        """
        for old_column, new_columns in flattening_mapping.items():
            data[new_columns] = self._restore_empty_values(data[new_columns])

            batch_size = 10
            data["batch_id"] = np.arange(len(data)) // batch_size
            for batch_id, batch_data in data.groupby("batch_id"):
                data.loc[batch_data.index, old_column] = batch_data[new_columns].apply(
                    lambda row: unflatten_list(row.to_dict(), "."), axis=1
                )
            data = data.drop(columns=["batch_id"])

            data[old_column] = data[old_column]. \
                apply(lambda row: self._remove_none_from_struct(row))
            data[old_column] = data[old_column]. \
                apply(lambda row: json.dumps(row, ensure_ascii=False))
            dropped_columns = set(i for i in new_columns if i not in duplicated_columns)
            data.drop(dropped_columns, axis=1, inplace=True)
        return data

    def run(self):
        """
        Launch the postprocessing of generated data,
        and save the processed data to the predefined path
        """

        if os.path.exists(self.path_to_flatten_metadata):
            logger.info("Start postprocessing of the generated data")
            for table in self.metadata.keys():
                if table == "global":
                    continue
                flatten_metadata = self._fetch_flatten_config(table)
                flattening_mapping = flatten_metadata.get("flattening_mapping")
                duplicated_columns = flatten_metadata.get("duplicated_columns")
                order_of_columns = flatten_metadata.get("order_of_columns")
                path_to_generated_data = (
                    f"{PATH_TO_MODEL_ARTIFACTS}/tmp_store/"
                    f"{slugify(table)}/merged_infer_{slugify(table)}.csv"
                )
                data = self._load_generated_data(path_to_generated_data, table)
                data = self._post_process_generated_data(
                    data,
                    flattening_mapping,
                    duplicated_columns
                )
                destination = self.metadata[table].get("infer_settings", {}).get("destination", "")
                path_to_destination = destination if destination else path_to_generated_data
                self._save_generated_data(data, path_to_destination, order_of_columns, table)
                logger.info("Finish postprocessing of the generated data")

    @staticmethod
    def _save_generated_data(
        generated_data: pd.DataFrame,
        path_to_destination: str,
        order_of_columns: List[str],
        *args
    ):
        """
        Save generated data to the path
        """
        generated_data = generated_data[order_of_columns]
        DataLoader(path_to_destination).save_data(
            generated_data,
            format=get_context().get_config(),
        )
