import shap
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path

class Explainability:
    def __init__(self, model, X_train: pd.DataFrame):
        self.model = model
        self.X_train = X_train
        self.explainer = self._create_explainer()

    def _create_explainer(self) -> any:
        # Determine explainer type based on model
        if isinstance(self.model, (XGBClassifier, RandomForestClassifier)):
            return shap.TreeExplainer(self.model) # Note: For some RF versions, might need more care
        elif isinstance(self.model, LogisticRegression):
            return shap.LinearExplainer(self.model, self.X_train, feature_perturbation="interventional")
        else:
             # Fallback to KernelExplainer if model type is generic or wrapped
             # Note: KernelExplainer is slow for large datasets
            return shap.KernelExplainer(self.model.predict, self.X_train.iloc[:100, :]) # Use sample

    def generate_global_explanation(self, X_test: pd.DataFrame, save_path: Optional[str] = None):
        """
        Generates and optionally saves global feature importance (SHAP summary plot).
        """
        # Calculate SHAP values for test set
        # Using a subset for speed if large dataset
        subset = X_test.iloc[:500, :] if len(X_test) > 500 else X_test
        shap_values = self.explainer.shap_values(subset)

        # Plot summary
        plt.figure()
        shap.summary_plot(shap_values, subset, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"SHAP summary plot saved to {save_path}")
        else:
            plt.show()

    def generate_local_explanation(self, instance: pd.Series, save_path: Optional[str] = None):
        """
        Generates explanation for a single prediction.
        """
        # Todo: Implement waterfall plot or similar for single instance
        pass 
