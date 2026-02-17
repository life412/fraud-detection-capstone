import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, f1_score
from pathlib import Path
from typing import Dict, Any

class Evaluator:
    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)
        self.y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else None

    def get_metrics(self) -> Dict[str, float]:
        """
        Calculates key metrics: F1-score, AUC-PR.
        """
        metrics = {}
        
        # F1 Score
        metrics['f1_score'] = f1_score(self.y_test, self.y_pred)
        
        # AUC-PR
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_prob)
        metrics['auc_pr'] = auc(recall, precision)
        
        return metrics

    def save_metrics(self, path: str):
        """
        Saves metrics to JSON file.
        """
        metrics = self.get_metrics()
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {path}")

    def plot_precision_recall_curve(self, path: str = None):
        """
        Plots Precision-Recall Curve.
        """
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_prob)
        auc_score = auc(recall, precision)
        
        plt.figure()
        plt.plot(recall, precision, label=f'AUC-PR = {auc_score:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        
        if path:
            plt.savefig(path)
            print(f"PR Curve saved to {path}")
        else:
            plt.show()
