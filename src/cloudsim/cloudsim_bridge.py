
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "cloudsim-engine" / "input"
OUTPUT_DIR = BASE_DIR / "cloudsim-engine" / "output"

def export_cloudsim_inputs(workload, resources):
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    workload_columns = [
        "Job_ID",
        "Task_ID",
        "Arrival_Time",
        "CPU",
        "Memory"
    ]

    resource_columns = [
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
        "predicted_carbon"
    ]

    workload_export = workload[workload_columns].copy()
    resources_export = resources[resource_columns].copy()

    workload_export.to_csv(INPUT_DIR / "workload.csv", index=False)
    resources_export.to_csv(INPUT_DIR / "resources.csv", index=False)

    return {
        "workload_records": len(workload_export),
        "cloud_hosts": len(resources_export),
        "input_directory": str(INPUT_DIR)
    }

def load_cloudsim_results():
    result_path = OUTPUT_DIR / "cloudsim_results.csv"

    if not result_path.exists():
        raise FileNotFoundError("CloudSim results not found.")

    return pd.read_csv(result_path)

