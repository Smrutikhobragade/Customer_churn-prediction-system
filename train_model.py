import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_FILE = "telecom_churn_synthetic_csv_file_.csv"
MODEL_FILE = "logisticregression_model.pkl"
LEGACY_MODEL_FILE = "churn_model.pkl"
METRICS_FILE = "logisticregression_metrics.json"
COMPARISON_FILE = "model_comparison_metrics.csv"

FEATURE_COLUMNS = [
    "Gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "Tenure",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaymentMethod",
]
NUMERICAL_COLUMNS = ["SeniorCitizen", "Tenure", "MonthlyCharges", "TotalCharges"]


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    missing = set(FEATURE_COLUMNS + ["Churn"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in training data: {sorted(missing)}")

    df = df[FEATURE_COLUMNS + ["Churn"]].copy()
    for col in NUMERICAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
    df = df.dropna(subset=["Churn"])

    X = df[FEATURE_COLUMNS]
    y = df["Churn"].astype(int)
    return X, y


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )


def build_pipeline(model):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        "algorithm": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def train_and_save_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    }

    comparison = []
    trained_logistic_pipeline = None
    for name, model in candidates.items():
        pipeline = build_pipeline(model)
        metrics = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        comparison.append(metrics)
        if name == "Logistic Regression":
            trained_logistic_pipeline = pipeline

    comparison_df = pd.DataFrame(comparison).sort_values("accuracy", ascending=False)
    comparison_df.to_csv(COMPARISON_FILE, index=False)

    logistic_metrics = next(row for row in comparison if row["algorithm"] == "Logistic Regression")
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(logistic_metrics, f, indent=2)

    joblib.dump(trained_logistic_pipeline, MODEL_FILE)
    joblib.dump(trained_logistic_pipeline, LEGACY_MODEL_FILE)

    print("\nAlgorithm accuracy check:")
    for _, row in comparison_df.iterrows():
        print(
            f"  {row['algorithm']}: accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"f1={row['f1']:.4f}, roc_auc={row['roc_auc']:.4f}"
        )

    print(f"\nSaved production model: {MODEL_FILE}")
    print(f"Saved legacy alias: {LEGACY_MODEL_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")
    print(f"Saved comparison report: {COMPARISON_FILE}")


if __name__ == "__main__":
    data_path = Path(DATA_FILE)
    if not data_path.exists():
        raise FileNotFoundError(f"Training file not found: {data_path.resolve()}")
    X, y = load_and_preprocess_data(data_path)
    train_and_save_logistic_regression(X, y)


