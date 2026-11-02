
import pandas as pd
from pathlib import Path


def calculate_carbon_emissions(energy_kwh, carbon_intensity_gco2_kwh):
    if energy_kwh < 0 or carbon_intensity_gco2_kwh < 0:
        raise ValueError("Energy and carbon intensity must be non-negative.")

    return energy_kwh * carbon_intensity_gco2_kwh



def calculate_total_energy(it_energy_kwh, pue=1.0):
    if (pd.Series(it_energy_kwh) < 0).any() or (pd.Series(pue) < 1).any():
        raise ValueError("Energy must be non-negative and PUE must be at least 1.")

    return it_energy_kwh * pue



def calculate_execution_cost(execution_time_seconds, cost_per_hour):
    if execution_time_seconds < 0 or cost_per_hour < 0:
        raise ValueError("Execution time and cost must be non-negative.")

    return (execution_time_seconds / 3600) * cost_per_hour


def compare_schedulers(results):
    required_columns = {
        "scheduler",
        "execution_time",
        "carbon_emissions_g",
        "energy_kwh",
        "execution_cost"
    }

    if results.empty:
        raise ValueError("No evaluation results available.")

    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return results.groupby("scheduler", as_index=False)[
        [
            "execution_time",
            "carbon_emissions_g",
            "energy_kwh",
            "execution_cost"
        ]
    ].mean()


def save_evaluation_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Evaluation results saved to {output_path}")
