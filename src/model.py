"""
Train and evaluate a Decision Tree classifier for customer churn prediction.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def train_model(X_train, y_train, max_depth: int = 5) -> DecisionTreeClassifier:
    """Train a Decision Tree with controlled depth to avoid overfitting."""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Return key performance metrics as a dictionary."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    return metrics


def print_report(model, X_test, y_test):
    """Print a full classification report and confusion matrix."""
    y_pred = model.predict(X_test)

    print("Classification Report")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    print("Confusion Matrix")
    print("=" * 50)
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Predicted:  Stayed  Churned")
    print(f"  Stayed      {cm[0][0]:>6}  {cm[0][1]:>7}")
    print(f"  Churned     {cm[1][0]:>6}  {cm[1][1]:>7}")
