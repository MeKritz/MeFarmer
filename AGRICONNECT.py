import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    classification_report,
)

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # older scikit-learn versions
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) ** 0.5

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


def generate_synthetic_dataset(n_rows: int = 4000) -> pd.DataFrame:
    crop_types = ["Wheat", "Rice", "Cotton", "Sugarcane", "Maize"]
    time_slots = ["Morning", "Afternoon", "Evening"]
    seasons = ["Kharif", "Rabi", "Zaid"]

    dates = pd.date_range("2023-10-01", periods=n_rows, freq="6h")

    df = pd.DataFrame({
        "date": dates,
        "day_of_week": dates.dayofweek,
        "season": rng.choice(seasons, n_rows),
        "crop_type": rng.choice(crop_types, n_rows),
        "time_of_day_slot": rng.choice(time_slots, n_rows),
        "centre_max_capacity": rng.integers(100, 400, n_rows),
        "staff_count": rng.integers(2, 10, n_rows),
        "registered_farmers_nearby": rng.integers(100, 1000, n_rows),
        "historical_avg_volume_kg": rng.normal(12000, 3000, n_rows).clip(1000),
        "avg_processing_time_min": rng.normal(9, 2, n_rows).clip(3),
    })

   
    df["farmers_scheduled_that_day"] = (
        (df["registered_farmers_nearby"] * rng.uniform(0.05, 0.25, n_rows))
        .astype(int)
        .clip(1, 300)
    )

   
    load_ratio = df["farmers_scheduled_that_day"] / df["staff_count"]
    df["waiting_time_minutes"] = (
        load_ratio * df["avg_processing_time_min"] * 0.6
        + rng.normal(0, 5, n_rows)
    ).clip(2, 240)

    fill_ratio = df["farmers_scheduled_that_day"] / df["centre_max_capacity"]
    df["workload_level"] = pd.cut(
        fill_ratio,
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    return df


NUMERIC_FEATURES = [
    "farmers_scheduled_that_day",
    "centre_max_capacity",
    "staff_count",
    "avg_processing_time_min",
    "registered_farmers_nearby",
    "historical_avg_volume_kg",
    "day_of_week",
]
CATEGORICAL_FEATURES = ["crop_type", "time_of_day_slot", "season"]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def train_waiting_time_model(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["waiting_time_minutes"]

    # Time-based split: train on the earlier 80% of dates, test on the rest.
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    results = {}
    for name, estimator in [
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED)),
    ]:
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        results[name] = {"pipeline": pipe, "mae": mae, "rmse": rmse}
        print(f"[Waiting-time | {name}]  MAE={mae:.2f} min   RMSE={rmse:.2f} min")

    best_name = min(results, key=lambda k: results[k]["mae"])
    best_pipe = results[best_name]["pipeline"]
    print(f"-> Best waiting-time model: {best_name}\n")

    joblib.dump(best_pipe, "waiting_time_model.joblib")
    return best_pipe


def train_workload_model(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["workload_level"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    results = {}
    for name, estimator in [
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)),
    ]:
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        results[name] = {"pipeline": pipe, "acc": acc, "f1": f1}
        print(f"[Workload | {name}]  Accuracy={acc:.3f}   Macro-F1={f1:.3f}")

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_pipe = results[best_name]["pipeline"]
    print(f"-> Best workload model: {best_name}")
    print(classification_report(y_test, best_pipe.predict(X_test)))

    joblib.dump(best_pipe, "workload_model.joblib")
    return best_pipe


def recommend_centres(farmer_location, candidate_centres, waiting_time_model):
    """
    candidate_centres: list of dicts, each describing one procurement centre
        with the same feature columns the waiting_time_model expects, plus
        'distance_km' and 'remaining_capacity'.
    Returns centres ranked best-to-worst by a weighted score.
    """
    rows = pd.DataFrame(candidate_centres)
    predicted_wait = waiting_time_model.predict(rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES])

    # Normalize each factor to 0-1 (lower distance/wait is better; higher
    # remaining capacity is better), then combine with simple weights.
    def norm(series, invert=False):
        s = (series - series.min()) / (series.max() - series.min() + 1e-9)
        return 1 - s if invert else s

    score = (
        0.4 * norm(rows["distance_km"], invert=True)
        + 0.4 * norm(pd.Series(predicted_wait), invert=True)
        + 0.2 * norm(rows["remaining_capacity"])
    )
    rows["predicted_waiting_minutes"] = predicted_wait
    rows["recommendation_score"] = score
    return rows.sort_values("recommendation_score", ascending=False)

if __name__ == "__main__":
    print("Generating synthetic dataset (labelled synthetic, for prototype only)...\n")
    data = generate_synthetic_dataset()

    print("Training waiting-time regression model...")
    wt_model = train_waiting_time_model(data)

    print("Training workload classification model...")
    workload_model = train_workload_model(data)

    print("\nSaved: waiting_time_model.joblib, workload_model.joblib")
    print("These can be loaded in the backend with joblib.load(...) and called")
    print("with .predict(new_data_as_dataframe) to serve live predictions.")