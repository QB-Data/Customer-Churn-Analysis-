"""
Load and prepare the Telco Customer Churn dataset for modelling.

Steps
-----
1. Read the CSV
2. Clean up columns (fix types, drop irrelevant fields)
3. Encode categorical features
4. Split into train / test sets
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/telco_churn.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Read raw CSV and return a DataFrame."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and remove columns that won't help the model."""
    df = df.copy()

    # TotalCharges has some blank strings — convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # customerID is just an identifier, not a predictor
    df.drop(columns=["customerID"], inplace=True)

    # Map the target column: Yes -> 1, No -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode every categorical column so the model can use it."""
    df = df.copy()
    label_encoders = {}

    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df


def split_data(df: pd.DataFrame, target: str = "Churn", test_size: float = 0.2):
    """Split into features (X) and target (y), then train/test."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def prepare_data(path: str = DATA_PATH):
    """Full pipeline: load -> clean -> encode -> split."""
    df = load_data(path)
    df = clean_data(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    return X_train, X_test, y_train, y_test, df
