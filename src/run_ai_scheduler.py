
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.cloudsim.carbon_prediction import (
    load_carbon_model,
    load_carbon_data,
    prepare_latest_features
)
from src.cloudsim.scheduler import select_best_region
from src.cloudsim.simulation import simulate_workload
from src.cloudsim.evaluation import (
    compare_schedulers,
    save_evaluation_results
)

DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

regions = pd.read_csv(DATA_DIR / "region_config.csv")
carbon_data = load_carbon_data()
model = load_carbon_model()

predictions = []

available_regions = carbon_data[
    ["provider", "region"]
].drop_duplicates()

for _, row in available_regions.iterrows():
    features = prepare_latest_features(
        carbon_data,
        row["provider"],
        row["region"]
    )

    prediction = max(0.0, float(model.predict(features)[0]))

    predictions.append({
        "provider": row["provider"],
        "region": row["region"],
        "predicted_carbon_intensity": prediction
    })

predicted_regions = pd.DataFrame(predictions)

predicted_regions.to_csv(
    RESULTS_DIR / "predicted_regions.csv",
    index=False
)

candidates = regions.merge(
    predicted_regions,
    on=["provider", "region"],
    how="inner"
)

candidates["predicted_carbon"] = candidates[
    "predicted_carbon_intensity"
]

candidates["estimated_energy_kwh"] = (
    candidates["max_power_w"] * candidates["pue"] / 1000
)

candidates["estimated_cost"] = candidates["cost_per_hour"]

ranked = select_best_region(candidates)

ranked.to_csv(
    RESULTS_DIR / "ai_scheduler_results.csv",
    index=False
)

candidates["score"] = 0.0

workload = pd.read_csv(
    DATA_DIR / "workload.csv",
    nrows=10000
)

baseline_results = simulate_workload(
    workload,
    candidates,
    scheduler="baseline"
)

ai_results = simulate_workload(
    workload,
    ranked,
    scheduler="AI Scheduler"
)

all_results = pd.concat(
    [baseline_results, ai_results],
    ignore_index=True
)

all_results.to_csv(
    RESULTS_DIR / "cloudsim_results.csv",
    index=False
)

comparison = compare_schedulers(all_results)

save_evaluation_results(
    comparison,
    RESULTS_DIR / "scheduler_comparison.csv"
)

ranked.head(10).to_csv(
    RESULTS_DIR / "dashboard_predictions.csv",
    index=False
)

print("\nAI scheduler completed")
print("Predicted regions:", len(predicted_regions))
print("Ranked resources:", len(ranked))
print("Workload tasks:", len(workload))

print("\nSCHEDULER COMPARISON")
print(comparison.to_string(index=False))

print("\nResults saved in:", RESULTS_DIR)
