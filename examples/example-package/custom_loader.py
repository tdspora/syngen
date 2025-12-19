from pathlib import Path
from typing import Optional
import pandas as pd


def get_dataframe(table_name: str, encoding: Optional[str] = "utf-8") -> pd.DataFrame:
    """
    Load a CSV file as a pandas DataFrame from the example-data directory.

    This function demonstrates the usage of a custom loader that can be passed
    to the 'loader' parameter in the training/inference processes.

    Parameters
    ----------
    table_name : str
        The name of the table (CSV file without extension) to load
    encoding : Optional[str], default="utf-8"
        The encoding to use when reading the CSV file

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame from the CSV file

    Raises
    ------
    FileNotFoundError
        If the specified CSV file does not exist in the example-data directory

    Examples
    --------
    >>> df = get_dataframe("housing")
    >>> print(df.head())
    """
    current_dir = Path(__file__).parents[1]
    path_to_example_data = current_dir / "example-data" / f"{table_name}.csv"

    if not path_to_example_data.exists():
        raise FileNotFoundError(
            f"The CSV file '{table_name}.csv' does not exist at: {path_to_example_data}"
        )

    return pd.read_csv(path_to_example_data, encoding=encoding)
