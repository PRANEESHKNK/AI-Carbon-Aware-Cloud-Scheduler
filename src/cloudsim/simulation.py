
import numpy as np
import pandas as pd


def simulate_workload(
    workload,
    resources,
    scheduler="baseline",
    base_runtime_seconds=60,
    arrival_window_seconds=3600,
    carbon_weight=0.6,
    energy_weight=0.25,
    cost_weight=0.15
):
    if scheduler not in {"baseline", "AI Scheduler"}:
        raise ValueError("Unknown scheduler")

    if workload.empty or resources.empty:
        raise ValueError("Workload and resources cannot be empty")

    if base_runtime_seconds <= 0 or arrival_window_seconds <= 0:
        raise ValueError("Simulation durations must be positive")

    required_tasks = {
        "Job_ID", "Task_ID", "Arrival_Time", "CPU", "Memory"
    }

    required_resources = {
        "provider", "region", "host_id", "total_cpu_mips",
        "ram_mb", "idle_power_w", "max_power_w",
        "pue", "cost_per_hour", "predicted_carbon", "score"
    }

    if required_tasks - set(workload.columns):
        raise ValueError("Missing workload columns")

    if required_resources - set(resources.columns):
        raise ValueError("Missing resource columns")

    tasks = workload.copy().reset_index(drop=True)
    hosts = resources.copy().reset_index(drop=True)

    for column in ["Arrival_Time", "CPU", "Memory"]:
        tasks[column] = pd.to_numeric(tasks[column], errors="coerce")

    numeric_host_columns = [
        "total_cpu_mips", "ram_mb", "idle_power_w",
        "max_power_w", "pue", "cost_per_hour",
        "predicted_carbon", "score"
    ]

    for column in numeric_host_columns:
        hosts[column] = pd.to_numeric(hosts[column], errors="coerce")

    tasks = tasks.dropna(
        subset=["Arrival_Time", "CPU", "Memory"]
    )

    hosts = hosts.dropna(
        subset=numeric_host_columns
    )

    tasks = tasks[
        (tasks["Arrival_Time"] >= 0) &
        (tasks["CPU"] > 0) &
        (tasks["Memory"] > 0)
    ].copy()

    hosts = hosts[
        (hosts["total_cpu_mips"] > 0) &
        (hosts["ram_mb"] > 0) &
        (hosts["max_power_w"] >= hosts["idle_power_w"]) &
        (hosts["pue"] >= 1) &
        (hosts["cost_per_hour"] >= 0) &
        (hosts["predicted_carbon"] >= 0)
    ].copy()

    if tasks.empty or hosts.empty:
        raise ValueError("No valid tasks or resources remain")

    cpu_reference = max(
        float(tasks["CPU"].quantile(0.99)), 1e-9
    )

    memory_reference = max(
        float(tasks["Memory"].quantile(0.99)), 1e-9
    )

    max_host_cpu = hosts["total_cpu_mips"].max()
    max_host_memory = hosts["ram_mb"].max()

    tasks["cpu_demand_normalized"] = (
        tasks["CPU"] / cpu_reference
    ).clip(0, 1)

    tasks["memory_demand_normalized"] = (
        tasks["Memory"] / memory_reference
    ).clip(0, 1)

    hosts["cpu_capacity_normalized"] = (
        hosts["total_cpu_mips"] / max_host_cpu
    )

    hosts["memory_capacity_normalized"] = (
        hosts["ram_mb"] / max_host_memory
    )

    hosts = hosts.reset_index(drop=True)

    tasks = tasks.sort_values(
        "Arrival_Time"
    ).reset_index(drop=True)

    arrival_span = (
        tasks["Arrival_Time"].max() -
        tasks["Arrival_Time"].min()
    )

    if arrival_span > 0:
        tasks["arrival_seconds"] = (
            (tasks["Arrival_Time"] - tasks["Arrival_Time"].min())
            / arrival_span
            * arrival_window_seconds
        )
    else:
        tasks["arrival_seconds"] = 0.0

    host_available = np.zeros(len(hosts))
    records = []

    weights = np.array([
        carbon_weight,
        energy_weight,
        cost_weight
    ], dtype=float)

    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid optimization weights")

    weights = weights / weights.sum()

    for _, task in tasks.iterrows():

        cpu_demand = float(task["cpu_demand_normalized"])
        memory_demand = float(task["memory_demand_normalized"])
        arrival = float(task["arrival_seconds"])

        feasible = hosts[
            (hosts["cpu_capacity_normalized"] >= cpu_demand) &
            (hosts["memory_capacity_normalized"] >= memory_demand)
        ]

        if feasible.empty:
            records.append({
                "Job_ID": task["Job_ID"],
                "Task_ID": task["Task_ID"],
                "scheduler": scheduler,
                "status": "unscheduled",
                "arrival_seconds": arrival
            })
            continue

        feasible_indices = feasible.index.to_numpy()

        cpu_capacity = hosts.loc[
            feasible_indices, "cpu_capacity_normalized"
        ].to_numpy()

        memory_capacity = hosts.loc[
            feasible_indices, "memory_capacity_normalized"
        ].to_numpy()

        cpu_utilization = np.divide(
            cpu_demand,
            cpu_capacity,
            out=np.zeros_like(cpu_capacity),
            where=cpu_capacity > 0
        )

        memory_utilization = np.divide(
            memory_demand,
            memory_capacity,
            out=np.zeros_like(memory_capacity),
            where=memory_capacity > 0
        )

        utilization = np.clip(cpu_utilization, 0, 1)

        duration = (
            base_runtime_seconds *
            np.maximum(cpu_utilization, 0.05)
        )

        start_times = np.maximum(
            arrival,
            host_available[feasible_indices]
        )

        finish_times = start_times + duration

        if scheduler == "baseline":
            chosen_position = np.lexsort((
                feasible_indices,
                start_times
            ))[0]

        else:
            carbon = feasible["predicted_carbon"].to_numpy()
            energy = (
                feasible["max_power_w"].to_numpy() *
                feasible["pue"].to_numpy() *
                duration / 3600000
            )
            cost = (
                feasible["cost_per_hour"].to_numpy() *
                duration / 3600
            )

            def normalize(values):
                low = values.min()
                high = values.max()

                if high == low:
                    return np.zeros(len(values))

                return (values - low) / (high - low)

            resource_score = (
                weights[0] * normalize(carbon) +
                weights[1] * normalize(energy) +
                weights[2] * normalize(cost)
            )

            waiting_time = start_times - arrival

            combined_score = (
                resource_score +
                0.05 * normalize(waiting_time)
            )

            chosen_position = np.argmin(combined_score)

        host_index = feasible_indices[chosen_position]
        host = hosts.loc[host_index]

        task_duration = float(duration[chosen_position])
        task_start = float(start_times[chosen_position])
        task_finish = float(finish_times[chosen_position])
        task_utilization = float(utilization[chosen_position])

        power_w = (
            host["idle_power_w"] +
            task_utilization *
            (host["max_power_w"] - host["idle_power_w"])
        )

        energy_kwh = (
            power_w *
            task_duration *
            host["pue"] / 3600000
        )

        emissions = (
            energy_kwh * host["predicted_carbon"]
        )

        cost = (
            task_duration *
            host["cost_per_hour"] / 3600
        )

        host_available[host_index] = task_finish

        records.append({
            "Job_ID": task["Job_ID"],
            "Task_ID": task["Task_ID"],
            "scheduler": scheduler,
            "status": "scheduled",
            "provider": host["provider"],
            "region": host["region"],
            "host_id": host["host_id"],
            "arrival_seconds": arrival,
            "start_seconds": task_start,
            "finish_seconds": task_finish,
            "execution_time": task_duration,
            "waiting_time": task_start - arrival,
            "cpu_utilization": task_utilization,
            "memory_utilization": float(
                memory_utilization[chosen_position]
            ),
            "energy_kwh": energy_kwh,
            "carbon_emissions_g": emissions,
            "execution_cost": cost,
            "predicted_carbon": host["predicted_carbon"]
        })

    return pd.DataFrame(records)
