import pandas as pd
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

from src.data_loader import DataLoader, DataLoaderConfig
from src.preprocessing import Preprocessor, PreprocessingConfig
from src.model import FraudModel, ModelConfig
from src.evaluate import Evaluator
from src.explainability import Explainability

@dataclass
class TrainingPipeline:
    data_path: str = "data/raw/creditcard.csv"
    output_dir: str = "outputs"
    model_type: str = "xgboost"
    use_smote: bool = True

    def run(self):
        # 1. Load Data
        print("Loading data...")
        config = DataLoaderConfig(data_path=self.data_path)
        loader = DataLoader(config)
        try:
            df = loader.load_data()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

        # 2. Preprocessing
        print("Preprocessing data...")
        prep_config = PreprocessingConfig(target_column="Class", scale_features=True)
        preprocessor = Preprocessor(prep_config)
        X_train, X_test, y_train, y_test = preprocessor.split_data(df)
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # 3. Train Model
        print(f"Training {self.model_type} model...")
        model_config = ModelConfig(
            model_type=self.model_type,
            use_smote=self.use_smote,
            random_state=42
        )
        trainer = FraudModel(model_config)
        trainer.train(X_train_processed, y_train)

        # 4. Evaluation
        print("Evaluating model...")
        evaluator = Evaluator(trainer.model, X_test_processed, y_test)
        metrics = evaluator.get_metrics()
        print(f"Metrics: {metrics}")
        
        output_path = Path(self.output_dir)
        output_path.mkdir(exist_ok=True)
        
        metrics_file = output_path / "metrics.json"
        evaluator.save_metrics(str(metrics_file))
        
        pr_curve_file = output_path / "pr_curve.png"
        evaluator.plot_precision_recall_curve(str(pr_curve_file))

        # 5. Explainability
        print("Generating explanations...")
        explainer = Explainability(trainer.model, X_train_processed)
        shap_file = output_path / "shap_summary.png"
        explainer.generate_global_explanation(X_test_processed, str(shap_file))

        print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fraud Detection Model")
    parser.add_argument("--data_path", type=str, default="data/raw/creditcard.csv", help="Path to raw data")
    parser.add_argument("--model_type", type=str, default="xgboost", choices=["xgboost", "random_forest", "logistic"], help="Model algorithm")
    parser.add_argument("--smote", action="store_true", help="Use SMOTE for class imbalance")

    args = parser.parse_args()

    pipeline = TrainingPipeline(
        data_path=args.data_path,
        model_type=args.model_type,
        use_smote=args.smote
    )
    pipeline.run()
