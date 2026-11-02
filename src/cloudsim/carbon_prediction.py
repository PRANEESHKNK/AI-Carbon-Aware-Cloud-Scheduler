
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "carbon_model.pkl"


def load_carbon_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained carbon model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_carbon_data():
    path = DATA_DIR / "carbon_intensity.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Processed carbon dataset not found: {path}"
        )

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def prepare_latest_features(df, provider, region):
    required = [
        "provider",
        "region",
        "timestamp",
        "renewable_percentage",
        "carbon_intensity_gco2_kwh",
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing carbon columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=required)

    region_data = df[
        (df["provider"] == provider) &
        (df["region"] == region)
    ].sort_values("timestamp")

    if len(region_data) < 4:
        raise ValueError(
            f"Not enough historical observations for {provider}/{region}"
        )

    latest = region_data.iloc[-1]
    previous = region_data["carbon_intensity_gco2_kwh"].iloc[-4:-1]

    return pd.DataFrame([{
        "renewable_percentage": latest["renewable_percentage"],
        "hour": latest["timestamp"].hour,
        "day_of_week": latest["timestamp"].dayofweek,
        "month": latest["timestamp"].month,
        "carbon_lag_1": previous.iloc[-1],
        "carbon_lag_2": previous.iloc[-2],
        "carbon_lag_3": previous.iloc[-3],
    }])


def predict_latest_carbon(provider, region):
    df = load_carbon_data()
    model = load_carbon_model()

    features = prepare_latest_features(df, provider, region)
    prediction = model.predict(features)[0]

    return max(0.0, float(prediction))


if __name__ == "__main__":
    carbon_data = load_carbon_data()

    available_regions = (
        carbon_data[["provider", "region"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print("Available provider-region combinations:")
    print(available_regions.head(10).to_string(index=False))
