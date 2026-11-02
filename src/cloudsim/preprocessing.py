from pathlib import Path
import pandas as pd

def load_carbon_data(file_path):
    df = pd.read_csv(file_path)

    required_columns = [
        "provider",
        "region",
        "timestamp",
        "carbon_intensity_gco2_kwh",
        "renewable_percentage",
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing carbon columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["carbon_intensity_gco2_kwh"] = pd.to_numeric(
        df["carbon_intensity_gco2_kwh"], errors="coerce"
    )
    df["renewable_percentage"] = pd.to_numeric(
        df["renewable_percentage"], errors="coerce"
    )

    df = df.dropna(subset=required_columns)
    df = df.drop_duplicates()
    df = df.sort_values(
        ["provider", "region", "timestamp"]
    ).reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    return df


def load_resource_data(file_path):
    df = pd.read_csv(file_path)

    required_columns = [
        "datacenter_id",
        "provider",
        "region",
        "host_id",
        "pes",
        "mips_per_pe",
        "ram_mb",
        "storage_mb",
        "bandwidth_mbps",
        "idle_power_w",
        "max_power_w",
        "pue",
        "cost_per_hour",
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing resource columns: {missing}")

    numeric_columns = [
        "pes",
        "mips_per_pe",
        "ram_mb",
        "storage_mb",
        "bandwidth_mbps",
        "idle_power_w",
        "max_power_w",
        "pue",
        "cost_per_hour",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_columns)
    df = df.drop_duplicates()
    df = df[
        (df["pes"] > 0)
        & (df["mips_per_pe"] > 0)
        & (df["ram_mb"] > 0)
        & (df["pue"] > 0)
        & (df["idle_power_w"] >= 0)
        & (df["max_power_w"] >= df["idle_power_w"])
        & (df["cost_per_hour"] >= 0)
    ].copy()

    df["total_cpu_mips"] = df["pes"] * df["mips_per_pe"]

    return df.reset_index(drop=True)


def load_workload_data(file_path, chunksize=250_000):
    required_columns = [
        "Job_ID",
        "Task_ID",
        "Arrival_Time",
        "CPU",
        "Memory",
    ]

    chunks = []

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        missing = set(required_columns) - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing workload columns: {missing}")

        for column in ["Arrival_Time", "CPU", "Memory"]:
            chunk[column] = pd.to_numeric(
                chunk[column], errors="coerce"
            )

        chunk = chunk.dropna(subset=required_columns)

        chunk = chunk[
            (chunk["Arrival_Time"] >= 0)
            & (chunk["CPU"] > 0)
            & (chunk["Memory"] > 0)
        ]

        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=required_columns)

    return pd.concat(chunks, ignore_index=True)


def save_processed_data(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")




