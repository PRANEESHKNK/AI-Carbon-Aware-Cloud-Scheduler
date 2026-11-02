
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

results_path = RESULTS_DIR / "scheduler_comparison.csv"

if not results_path.exists():
    raise FileNotFoundError(results_path)

df = pd.read_csv(results_path)

metrics = {
    "execution_time": "Execution Time (seconds)",
    "energy_kwh": "Energy Consumption (kWh)",
    "carbon_emissions_g": "Carbon Emissions (gCO2)",
    "execution_cost": "Execution Cost"
}

for column, title in metrics.items():
    plt.figure(figsize=(8, 5))
    plt.bar(df["scheduler"], df[column])
    plt.title(title)
    plt.ylabel(title)
    plt.xlabel("Scheduler")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"comparison_{column}.png")
    plt.show()

print("All comparison charts saved.")
