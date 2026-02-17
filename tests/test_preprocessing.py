import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import Preprocessor, PreprocessingConfig

@pytest.fixture
def dummy_data():
    """
    Creates a dummy dataframe for testing.
    """
    data = {
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100),
        'Class': np.random.choice([0, 1], size=100)
    }
    return pd.DataFrame(data)

def test_preprocessing_initialization():
    config = PreprocessingConfig()
    preprocessor = Preprocessor(config)
    assert preprocessor.scaler is not None
    assert preprocessor.imputer is not None

def test_split_data(dummy_data):
    config = PreprocessingConfig(test_size=0.2, random_state=42)
    preprocessor = Preprocessor(config)
    
    X = dummy_data.drop(columns=['Class'])
    y = dummy_data['Class']
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(dummy_data)
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

def test_fit_transform(dummy_data):
    config = PreprocessingConfig()
    preprocessor = Preprocessor(config)
    X_train, X_test, y_train, y_test = preprocessor.split_data(dummy_data)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    assert X_train_processed.shape == X_train.shape
    # Check if scaled (mean approx 0)
    assert np.isclose(X_train_processed['Feature1'].mean(), 0, atol=0.1)

def test_missing_target_column(dummy_data):
    config = PreprocessingConfig(target_column="MissingCol")
    preprocessor = Preprocessor(config)
    with pytest.raises(ValueError, match="Target column MissingCol not found"):
         preprocessor.split_data(dummy_data)
