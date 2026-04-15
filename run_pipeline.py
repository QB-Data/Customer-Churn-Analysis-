"""
Run the full churn prediction pipeline end to end.

Usage:
    python run_pipeline.py
"""

from src.data_processing import prepare_data
from src.model import train_model, print_report, evaluate_model
from src.visualize import generate_all_visuals


def main():
    # --- 1. Data ---
    print("Loading and preparing data ...")
    X_train, X_test, y_train, y_test, df = prepare_data()
    feature_names = list(X_train.columns)
    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")
    print()

    # --- 2. Model ---
    print("Training Decision Tree ...")
    model = train_model(X_train, y_train, max_depth=5)
    print()

    # --- 3. Evaluation ---
    metrics = evaluate_model(model, X_test, y_test)
    print_report(model, X_test, y_test)
    print()
    print(f"Quick summary — Accuracy: {metrics['accuracy']:.2%}  |  "
          f"F1 Score: {metrics['f1_score']:.2%}")
    print()

    # --- 4. Visuals ---
    print("Generating charts ...")
    generate_all_visuals(model, X_test, y_test, df, feature_names)
    print("Done.")


if __name__ == "__main__":
    main()
