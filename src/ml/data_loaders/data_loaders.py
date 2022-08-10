import pandas as pd
import pandavro as pdx
import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional


class BaseDataLoader(ABC):
    """
    Abstract class for data loader
    """

    @abstractmethod
    def load_data(self, path: str):
        pass

    @abstractmethod
    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        pass


class CSVLoader(BaseDataLoader):
    """
    Class for loading and saving data in csv format
    """
    
    def load_data(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path, engine="python", **kwargs).iloc[:, :]

    def save_data(self, path: Optional[str], df: pd.DataFrame, **kwargs):
        if df is not None:
            df.to_csv(path, **kwargs)
        

class AvroLoader(BaseDataLoader):
    """
    Class for loading and saving data in avro format
    """

    def load_data(self, path: str, **kwargs) -> pd.DataFrame:
        return pdx.from_avro(path, **kwargs)

    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        if df is not None:
            pdx.to_avro(path, df, **kwargs)


class JSONLoader(BaseDataLoader):
    """
    Class for loading and saving data in json format
    """

    def load_data(self, path: str) -> dict:
        with open(path) as metadata_file:
            metadata = json.load(metadata_file)
        return metadata

    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        raise NotImplementedError("Saving JSON files is not supported")


class DataLoader(BaseDataLoader):
    """
    Base class for loading and saving data either in csv or in avro format
    """

    def __init__(self):
        self.csv_loader = CSVLoader()
        self.avro_loader = AvroLoader()

    def load_data(self, path: str, **kwargs) -> pd.DataFrame:
        path = Path(path)

        if path.suffix == '.avro':
            df = self.avro_loader.load_data(str(path), **kwargs)

        elif path.suffix == '.csv':
            df = self.csv_loader.load_data(str(path), **kwargs)
            df.columns = df.columns.str.replace(':', '')
        else:
            raise NotImplementedError("File format not supported")
        if df.shape[0] < 1:
            raise ValueError("Empty file was provided. Unable to train.")
        return df

    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        path = Path(path)

        if path.suffix == '.avro':
            self.avro_loader.save_data(str(path), df, **kwargs)

        elif path.suffix == '.csv':
            self.csv_loader.save_data(str(path), df, **kwargs)
        
        else:
            raise NotImplementedError("File format not supported")


class MetadataLoader(BaseDataLoader):
    """
    Metadata class for loading and saving metadata in json format
    """

    def __init__(self):
        self.json_loader = JSONLoader()

    def load_data(self, path: str) -> dict:
        path = Path(path)
        if path.suffix == '.json':
            return self.json_loader.load_data(str(path))
        else:
            raise NotImplementedError("File format not supported")

    def save_data(self, path: str, df: pd.DataFrame, **kwargs):
        path = Path(path)
        if path.suffix == '.json':
            self.json_loader.save_data(str(path), df, **kwargs)
        else:
            raise NotImplementedError("File format not supported")
