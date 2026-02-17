import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class DataLoaderConfig:
    data_path: str
    target_column: str = "Class"  # Default for creditcard.csv

class DataLoader:
    def __init__(self, config: DataLoaderConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the configured path.
        Handles missing file errors gracefully.
        """
        path = Path(self.config.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found at {self.config.data_path}")
        
        try:
            df = pd.read_csv(path)
            # Basic validation
            if self.config.target_column not in df.columns:
                 # Check if we are in inference mode or maybe the column is named differently
                 # For this capstone, let's just warn or raise if critical
                 print(f"Warning: Target column '{self.config.target_column}' not found in dataset columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")
