
import pandas as pd


def select_best_time_window(
    forecast_df,
    time_col="timestamp",
    carbon_col="predicted_carbon_intensity",
    earliest_time=None,
    latest_time=None
):
    required = {time_col, carbon_col}
    missing = required - set(forecast_df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if forecast_df.empty:
        raise ValueError("Forecast data is empty.")

    forecast = forecast_df.copy()
    forecast[time_col] = pd.to_datetime(
        forecast[time_col], errors="coerce", utc=True
    )
    forecast[carbon_col] = pd.to_numeric(
        forecast[carbon_col], errors="coerce"
    )

    forecast = forecast.dropna(subset=[time_col, carbon_col])

    if earliest_time is not None:
        earliest_time = pd.to_datetime(earliest_time, utc=True)
        forecast = forecast[forecast[time_col] >= earliest_time]

    if latest_time is not None:
        latest_time = pd.to_datetime(latest_time, utc=True)
        forecast = forecast[forecast[time_col] <= latest_time]

    if forecast.empty:
        raise ValueError("No forecast available in the requested time window.")

    return forecast.sort_values(
        carbon_col
    ).reset_index(drop=True)


def schedule_workload_by_time(
    workload_df,
    forecast_df,
    arrival_col="Arrival_Time"
):
    if arrival_col not in workload_df.columns:
        raise ValueError(f"Missing column: {arrival_col}")

    workload = workload_df.copy()
    workload[arrival_col] = pd.to_numeric(
        workload[arrival_col], errors="coerce"
    )

    workload = workload.dropna(subset=[arrival_col])

    best_windows = select_best_time_window(forecast_df)

    workload["scheduled_time"] = best_windows.iloc[0]["timestamp"]
    workload["predicted_carbon"] = best_windows.iloc[0][
        "predicted_carbon_intensity"
    ]

    return workload
