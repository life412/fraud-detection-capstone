import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, average_precision_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

@dataclass
class ModelConfig:
    model_type: str = "xgboost"  # 'logistic', 'random_forest', 'xgboost'
    use_smote: bool = True
    random_state: int = 42
    params: Optional[Dict[str, Any]] = None

class FraudModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._build_model()
    
    def _build_model(self) -> Any:
        params = self.config.params or {}
        
        if self.config.model_type == "logistic":
            return LogisticRegression(
                random_state=self.config.random_state, 
                class_weight='balanced' if not self.config.use_smote else None,
                **params
            )
        elif self.config.model_type == "random_forest":
            return RandomForestClassifier(
                random_state=self.config.random_state, 
                class_weight='balanced' if not self.config.use_smote else None,
                n_jobs=-1,
                **params
            )
        elif self.config.model_type == "xgboost":
             # XGBoost specific handling for imbalance
             scale_pos_weight = params.pop('scale_pos_weight', 1) # Default 1, but user should tune
             return XGBClassifier(
                 random_state=self.config.random_state,
                 n_jobs=-1,
                 scale_pos_weight=scale_pos_weight,
                 **params
             )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the model. If SMOTE is enabled, uses a pipeline.
        """
        if self.config.use_smote:
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=self.config.random_state)),
                ('model', self.model)
            ])
            pipeline.fit(X_train, y_train)
            self.model = pipeline.named_steps['model'] # Extract trained model for later use
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)
        else:
            raise NotImplementedError("Model does not support predict_proba")
