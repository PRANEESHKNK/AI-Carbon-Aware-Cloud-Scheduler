
import pandas as pd


def select_best_region(
    candidates,
    carbon_weight=0.6,
    energy_weight=0.25,
    cost_weight=0.15,
):
    required = {
        "provider",
        "region",
        "predicted_carbon",
        "estimated_energy_kwh",
        "estimated_cost",
    }

    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if candidates.empty:
        raise ValueError("No candidate regions available.")

    if min(carbon_weight, energy_weight, cost_weight) < 0:
        raise ValueError("Weights must be non-negative.")

    if carbon_weight + energy_weight + cost_weight == 0:
        raise ValueError("At least one weight must be positive.")

    candidates = candidates.copy()

    for column in [
        "predicted_carbon",
        "estimated_energy_kwh",
        "estimated_cost",
    ]:
        candidates[column] = pd.to_numeric(
            candidates[column], errors="coerce"
        )

    candidates = candidates.dropna(
        subset=[
            "predicted_carbon",
            "estimated_energy_kwh",
            "estimated_cost",
        ]
    )

    if candidates.empty:
        raise ValueError("No valid candidate estimates.")

    if (
        candidates[
            ["predicted_carbon", "estimated_energy_kwh", "estimated_cost"]
        ] < 0
    ).any().any():
        raise ValueError("Carbon, energy, and cost must be non-negative.")

    weights = [carbon_weight, energy_weight, cost_weight]
    columns = [
        "predicted_carbon",
        "estimated_energy_kwh",
        "estimated_cost",
    ]

    for column, weight in zip(columns, weights):
        minimum = candidates[column].min()
        maximum = candidates[column].max()

        if maximum > minimum:
            candidates[column + "_normalized"] = (
                candidates[column] - minimum
            ) / (maximum - minimum)
        else:
            candidates[column + "_normalized"] = 0.0

    total_weight = sum(weights)

    candidates["score"] = sum(
        weight * candidates[column + "_normalized"]
        for column, weight in zip(columns, weights)
    ) / total_weight

    return candidates.sort_values("score").reset_index(drop=True)
