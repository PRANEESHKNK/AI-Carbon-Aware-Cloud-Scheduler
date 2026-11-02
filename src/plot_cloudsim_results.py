
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

results_path = RESULTS_DIR / "cloudsim_results.csv"

if not results_path.exists():
    raise FileNotFoundError(
        f"Simulation results not found: {results_path}"
    )

df = pd.read_csv(results_path)

required = {
    "execution_time",
    "energy_kwh",
    "carbon_emissions"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

metrics = [
    ("execution_time", "Execution Time", "Seconds"),
    ("energy_kwh", "Energy Consumption", "kWh"),
    ("carbon_emissions", "Carbon Emissions", "gCO2")
]

for column, title, ylabel in metrics:
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[column])
    plt.title(title)
    plt.xlabel("Task")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{column}.png")
    plt.show()

print("Cloud simulation visualizations saved.")
