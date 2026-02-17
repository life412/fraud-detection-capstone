import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from typing import Tuple, Optional
import pickle
from pathlib import Path

@dataclass
class PreprocessingConfig:
    target_column: str = "Class"
    test_size: float = 0.2
    random_state: int = 42
    scale_features: bool = True
    handle_missing: bool = True

class Preprocessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.scaler = StandardScaler() if config.scale_features else None
        self.imputer = SimpleImputer(strategy='median') if config.handle_missing else None
        self.feature_names: list[str] = []

    def load_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning (duplicates, etc).
        """
        df = df.drop_duplicates()
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data into train and test sets with stratification.
        """
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column {self.config.target_column} not found.")

        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]
        
        # Save feature names
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state, 
            stratify=y
        )
        return X_train, X_test, y_train, y_test

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fits imputer and scaler on training data and transforms it.
        """
        X_processed = X_train.copy()
        
        if self.imputer:
            X_processed = pd.DataFrame(
                self.imputer.fit_transform(X_processed), 
                columns=X_train.columns, 
                index=X_train.index
            )

        if self.scaler:
            X_processed = pd.DataFrame(
                self.scaler.fit_transform(X_processed), 
                columns=X_train.columns, 
                index=X_train.index
            )
            
        return X_processed

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms test data using fitted scaler/imputer.
        """
        X_processed = X_test.copy()
        
        if self.imputer:
            X_processed = pd.DataFrame(
                self.imputer.transform(X_processed), 
                columns=X_test.columns, 
                index=X_test.index
            )

        if self.scaler:
            X_processed = pd.DataFrame(
                self.scaler.transform(X_processed), 
                columns=X_test.columns, 
                index=X_test.index
            )
            
        return X_processed
