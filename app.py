from src.cloudsim.cloudsim_bridge import export_cloudsim_inputs
from pathlib import Path
import sys
from datetime import date

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.cloudsim.carbon_prediction import (
    load_carbon_model,
    load_carbon_data,
    prepare_latest_features
)

from src.cloudsim.scheduler import select_best_region
from src.cloudsim.simulation import simulate_workload

DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="CarbonPilot AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background-color: #F5F8F6;
}
[data-testid="stSidebar"] {
    background-color: #EAF1EC;
}
.main-title {
    font-size: 34px;
    font-weight: 750;
    color: #174B37;
}
.subtitle {
    color: #64756B;
    font-size: 15px;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #DCE8DF;
}
.recommendation {
    background: #E5F4E9;
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #23834D;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_carbon_model()


@st.cache_data
def get_carbon_data():
    return load_carbon_data()


@st.cache_data
def get_regions():
    return pd.read_csv(DATA_DIR / "region_config.csv")


def predict_schedule(model, carbon_data, regions, selected_date, start_hour, end_hour, electricity_kwh, duration_hours, carbon_weight, energy_weight, cost_weight):

    predictions = []

    available = carbon_data[
        ["provider", "region"]
    ].drop_duplicates()

    for _, item in available.iterrows():
        provider = item["provider"]
        region = item["region"]

        try:
            base_features = prepare_latest_features(
                carbon_data,
                provider,
                region
            )
        except ValueError:
            continue

        for hour in range(start_hour, end_hour + 1):
            features = base_features.copy()

            timestamp = pd.Timestamp(selected_date) + pd.Timedelta(hours=hour)

            features["hour"] = timestamp.hour
            features["day_of_week"] = timestamp.dayofweek
            features["month"] = timestamp.month

            predicted_carbon = max(
                0.0,
                float(model.predict(features)[0])
            )

            predictions.append({
                "provider": provider,
                "region": region,
                "timestamp": timestamp,
                "predicted_carbon": predicted_carbon
            })

    if not predictions:
        raise ValueError("No valid region predictions available.")

    prediction_df = pd.DataFrame(predictions)

    candidates = regions.merge(
        prediction_df,
        on=["provider", "region"],
        how="inner"
    )

    candidates["estimated_energy_kwh"] = (
        electricity_kwh * candidates["pue"]
    )

    candidates["estimated_cost"] = (
        duration_hours * candidates["cost_per_hour"]
    )

    ranked = select_best_region(
        candidates,
        carbon_weight=carbon_weight,
        energy_weight=energy_weight,
        cost_weight=cost_weight
    )

    candidates["score"] = 0.0
    return ranked, candidates

def calculate_reduction(baseline, optimized):
    if baseline <= 0:
        return 0.0

    return ((baseline - optimized) / baseline) * 100


st.sidebar.markdown("## 🌱 CarbonPilot AI")
st.sidebar.caption("Carbon-aware cloud scheduling")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Optimize Schedule",
        "About"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Powered by Machine Learning")

if page == "About":
    st.title("About CarbonPilot AI")
    st.write(
        "An AI-powered system that estimates carbon intensity "
        "and recommends cloud regions and scheduling windows "
        "to reduce the environmental impact of cloud workloads."
    )

    st.subheader("Technology")
    st.write(
        "Python · Streamlit · Scikit-learn · Pandas · Plotly"
    )
    st.stop()


st.markdown(
    '<div class="main-title">Carbon-Aware Cloud Scheduler</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Optimize cloud workload placement '
    'and scheduling using machine learning.</div>',
    unsafe_allow_html=True
)

st.write("")

try:
    model = get_model()
    carbon_data = get_carbon_data()
    regions = get_regions()
except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()


if page == "Dashboard":

    st.subheader("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Available Regions", carbon_data[
        ["provider", "region"]
    ].drop_duplicates().shape[0])

    col2.metric("Cloud Resources", len(regions))

    col3.metric("Historical Carbon Records", len(carbon_data))

    st.divider()

    st.subheader("Historical Carbon Intensity")

    history = carbon_data.copy()

    history["timestamp"] = pd.to_datetime(
        history["timestamp"]
    )

    region_options = sorted(
    history[["provider", "region"]]
    .drop_duplicates()
    .apply(
        lambda row: f"{row['provider']} / {row['region']}",
        axis=1
    )
    .tolist()
)

    selected_history_region = st.selectbox(
        "Select region",
        region_options
    )

    provider, region = selected_history_region.split(" / ", 1)

    history = history[
        (history["provider"] == provider) &
        (history["region"] == region)
    ].sort_values("timestamp")

    fig = px.line(
        history,
        x="timestamp",
        y="carbon_intensity_gco2_kwh",
        title=f"Historical Carbon Intensity: {region}",
        labels={
            "timestamp": "Time",
            "carbon_intensity_gco2_kwh": "Carbon Intensity (gCO2/kWh)"
        }
    )

    fig.update_layout(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font_color="#173B2B",
    height=430
)

    st.plotly_chart(fig, width="stretch")

    st.subheader("Renewable Energy Percentage")

    renewable_fig = px.line(
        history,
        x="timestamp",
        y="renewable_percentage",
        title="Renewable Energy Trend"
    )

    renewable_fig.update_layout(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font_color="#173B2B",
    height=350
    )

    st.plotly_chart(
        renewable_fig,
        width="stretch"
    )

    st.info(
        "Historical observations are shown here. "
        "Use Optimize Schedule to generate model-based recommendations."
    )


if page == "Optimize Schedule":

    st.subheader("Configure Your Workload")

    st.write(
        "Enter your workload and current cloud configuration. "
        "You can also upload your workload dataset."
    )

    
    uploaded_file = st.file_uploader(
        "Upload workload CSV (optional)",
        type=["csv"],
        key="workload_csv_upload"
    )

    workload_count = 0
    workload_df = None

    if uploaded_file is not None:
        try:
            workload_df = pd.read_csv(uploaded_file)

            required = {
                "Job_ID",
                "Task_ID",
                "Arrival_Time",
                "CPU",
                "Memory"
            }

            if not required.issubset(workload_df.columns):
                st.error("Invalid workload CSV columns.")
                workload_df = None
            else:
                workload_count = len(workload_df)
                st.success(
                    f"Loaded {workload_count:,} workload records."
                )
                st.dataframe(
                    workload_df.head(5),
                    width="stretch"
                )

        except Exception as error:
            st.error(f"Unable to read workload: {error}")
            workload_df = None


    st.divider()

    st.subheader("Current Cloud Configuration")

    provider_options = sorted(
        regions["provider"].dropna().unique().tolist()
    )

    selected_provider = st.selectbox(
        "Current provider",
        provider_options
    )

    provider_regions = sorted(
        regions.loc[
            regions["provider"] == selected_provider,
            "region"
        ].dropna().unique().tolist()
    )

    selected_region = st.selectbox(
        "Current region",
        provider_regions
    )

    current_history = carbon_data[
        (carbon_data["provider"] == selected_provider) &
        (carbon_data["region"] == selected_region)
    ].sort_values("timestamp")

    if current_history.empty:
        st.error("No carbon history exists for this region.")
        st.stop()

    default_carbon = float(
        current_history.iloc[-1]["carbon_intensity_gco2_kwh"]
    )

    current_resources = regions[
        (regions["provider"] == selected_provider) &
        (regions["region"] == selected_region)
    ]

    default_pue = float(current_resources["pue"].median())

    col1, col2 = st.columns(2)

    with col1:
        current_carbon = st.number_input(
            "Current carbon intensity (gCO2/kWh)",
            min_value=0.01,
            value=max(default_carbon, 0.01),
            step=1.0
        )

        electricity_kwh = st.number_input(
            "IT electricity consumption (kWh)",
            min_value=0.01,
            value=10.0,
            step=1.0
        )

    with col2:
        duration_hours = st.number_input(
            "Estimated workload duration (hours)",
            min_value=0.01,
            value=1.0,
            step=0.25
        )

        current_pue = st.number_input(
            "Current region PUE",
            min_value=1.0,
            value=max(default_pue, 1.0),
            step=0.05
        )

    st.divider()

    st.subheader("Scheduling Preferences")

    col1, col2 = st.columns(2)

    with col1:
        scheduling_date = st.date_input(
            "Scheduling date",
            value=date.today()
        )

        start_hour = st.selectbox(
            "Earliest scheduling hour",
            list(range(24)),
            index=9,
            format_func=lambda x: f"{x:02d}:00"
        )

    with col2:
        end_hour = st.selectbox(
            "Latest scheduling hour",
            list(range(24)),
            index=17,
            format_func=lambda x: f"{x:02d}:00"
        )

    priority = st.selectbox(
        "Optimization priority",
        [
            "Carbon priority",
            "Balanced",
            "Cost priority"
        ]
    )

    if priority == "Carbon priority":
        carbon_weight, energy_weight, cost_weight = 0.75, 0.15, 0.10
    elif priority == "Cost priority":
        carbon_weight, energy_weight, cost_weight = 0.35, 0.15, 0.50
    else:
        carbon_weight, energy_weight, cost_weight = 0.60, 0.25, 0.15

    st.caption(
        f"Optimization weights — Carbon: {carbon_weight:.0%}, "
        f"Energy: {energy_weight:.0%}, Cost: {cost_weight:.0%}"
    )

    optimize = st.button(
        "🌱 Optimize My Cloud Schedule",
        type="primary",
        width="stretch"
    )

    if optimize:

        if end_hour < start_hour:
            st.error("Latest hour must not be earlier than earliest hour.")

        elif scheduling_date < date.today():
            st.error("Please select today or a future date.")

        else:
            with st.spinner(
                "Predicting carbon intensity and evaluating cloud resources..."
            ):

                try:
                    ranked,candidates= predict_schedule(
                        model,
                        carbon_data,
                        regions,
                        scheduling_date,
                        start_hour,
                        end_hour,
                        electricity_kwh,
                        duration_hours,
                        carbon_weight,
                        energy_weight,
                        cost_weight
                    )

                    best = ranked.iloc[0]
                    
                    if workload_df is None:
                        workload_df = pd.read_csv(
                            DATA_DIR / "workload.csv",
                            nrows=10000
                        )
                        st.warning("DEBUG: CloudSim export block reached")

                        st.write("Workload shape:", workload_df.shape)
                        st.write("Ranked resources shape:", ranked.shape)
                        
                        bridge_status = export_cloudsim_inputs(
                            workload=workload_df,
                            resources=ranked
                        )
                        st.write("DEBUG: Export returned:", bridge_status)
                        st.success(
                            f"Exported {bridge_status['workload_records']:,} tasks "
                            f"and {bridge_status['cloud_hosts']:,} cloud hosts for CloudSim Plus."
                        )

                    baseline_results = simulate_workload(
                        workload_df,
                        candidates,
                        scheduler="baseline"
                    )

                    ai_results = simulate_workload(
                        workload_df,
                        ranked,
                        scheduler="AI Scheduler",
                        carbon_weight=carbon_weight,
                        energy_weight=energy_weight,
                        cost_weight=cost_weight
                    )

                    all_simulation_results = pd.concat(
                        [baseline_results, ai_results],
                        ignore_index=True
                    )

                    scheduled_results = all_simulation_results[
                        all_simulation_results["status"] == "scheduled"
                    ]

                    simulation_summary = (
                        scheduled_results.groupby("scheduler")
                        .agg(
                            scheduled_tasks=("Task_ID", "count"),
                            makespan_seconds=("finish_seconds", "max"),
                            average_waiting_seconds=("waiting_time", "mean"),
                            energy_kwh=("energy_kwh", "sum"),
                            carbon_emissions_g=("carbon_emissions_g", "sum"),
                            execution_cost=("execution_cost", "sum")
                        )
                        .reset_index()
                    )

                    simulation_summary["unscheduled_tasks"] = (
                        len(workload_df) - simulation_summary["scheduled_tasks"]
                    )

                    simulation_summary.to_csv(
                        RESULTS_DIR / "workload_simulation_comparison.csv",
                        index=False
                    )

                    all_simulation_results.to_csv(
                        RESULTS_DIR / "workload_task_results.csv",
                        index=False
                    )

                    baseline_metrics = simulation_summary[
                        simulation_summary["scheduler"] == "baseline"
                    ].iloc[0]

                    ai_metrics = simulation_summary[
                        simulation_summary["scheduler"] == "AI Scheduler"
                    ].iloc[0]

                    carbon_reduction = calculate_reduction(
                        baseline_metrics["carbon_emissions_g"],
                        ai_metrics["carbon_emissions_g"]
                    )

                    energy_reduction = calculate_reduction(
                        baseline_metrics["energy_kwh"],
                        ai_metrics["energy_kwh"]
                    )

                    cost_reduction = calculate_reduction(
                        baseline_metrics["execution_cost"],
                        ai_metrics["execution_cost"]
)


                    
                    baseline_energy = baseline_metrics["energy_kwh"]
                    optimized_energy = ai_metrics["energy_kwh"]

                    baseline_emissions = baseline_metrics["carbon_emissions_g"]
                    optimized_emissions = ai_metrics["carbon_emissions_g"]

                    baseline_cost = baseline_metrics["execution_cost"]
                    optimized_cost = ai_metrics["execution_cost"]

                    carbon_reduction = calculate_reduction(
                        baseline_emissions,
                        optimized_emissions
                    )

                    energy_reduction = calculate_reduction(
                        baseline_energy,
                        optimized_energy
                    )

                    cost_reduction = calculate_reduction(
                        baseline_cost,
                        optimized_cost
                    )


                    energy_reduction = calculate_reduction(
                        baseline_energy,
                        optimized_energy
                    )

                    cost_reduction = calculate_reduction(
                        baseline_cost,
                        optimized_cost
                    )

                    st.session_state["ranked"] = ranked
                    st.session_state["best"] = best.to_dict()
                    st.session_state["comparison"] = {
                        "Baseline": {
                            "Carbon Emissions (gCO2)": baseline_emissions,
                            "Energy (kWh)": baseline_energy,
                            "Cost": baseline_cost
                        },
                        "AI Optimized": {
                            "Carbon Emissions (gCO2)": optimized_emissions,
                            "Energy (kWh)": optimized_energy,
                            "Cost": optimized_cost
                        }
                    }

                    st.session_state["reductions"] = {
                        "carbon": carbon_reduction,
                        "energy": energy_reduction,
                        "cost": cost_reduction
                    }

                    st.session_state["simulation_summary"] = simulation_summary
                    st.session_state["task_results"] = all_simulation_results

                except Exception as error:
                    st.error(f"Optimization failed: {error}")

    if "best" in st.session_state:

        best = st.session_state["best"]
        reductions = st.session_state["reductions"]
        comparison = st.session_state["comparison"]
        ranked = st.session_state["ranked"]

        st.divider()
        st.header("AI Optimization Results")

        st.markdown(
            f"""
            <div class="recommendation">
                <h3>🌱 Recommended Cloud Configuration</h3>
                <h2>{best['provider']} · {best['region']}</h2>
                <h3>Recommended time: {pd.Timestamp(best['timestamp']).strftime('%Y-%m-%d %H:%M')}</h3>
                <p>AI predicted carbon intensity:
                <b>{best['predicted_carbon']:.2f} gCO2/kWh</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Carbon Reduction",
            f"{reductions['carbon']:.2f}%",
            delta=f"{reductions['carbon']:.2f}%"
        )

        col2.metric(
            "Energy Reduction",
            f"{reductions['energy']:.2f}%",
            delta=f"{reductions['energy']:.2f}%"
        )

        col3.metric(
            "Cost Reduction",
            f"{reductions['cost']:.2f}%",
            delta=f"{reductions['cost']:.2f}%"
        )

        st.divider()
        
        simulation_summary = st.session_state["simulation_summary"]

        ai_metrics = simulation_summary[
            simulation_summary["scheduler"] == "AI Scheduler"
        ].iloc[0]

        st.subheader("Workload Execution Analysis")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Tasks",
            f"{int(ai_metrics['scheduled_tasks'] + ai_metrics['unscheduled_tasks']):,}"
        )

        col2.metric(
            "Scheduled Tasks",
            f"{int(ai_metrics['scheduled_tasks']):,}"
        )

        col3.metric(
            "Unscheduled Tasks",
            f"{int(ai_metrics['unscheduled_tasks']):,}"
        )

        col4.metric(
            "Execution Makespan",
            f"{ai_metrics['makespan_seconds']:.2f} sec"
        )

        st.subheader("Scheduling Performance")

        st.dataframe(
            simulation_summary,
            width="stretch",
            hide_index=True
        )

        st.metric(
            "Average AI Task Waiting Time",
            f"{ai_metrics['average_waiting_seconds']:.2f} seconds"
        )


        st.subheader("Baseline vs AI Scheduler")

        comparison_df = pd.DataFrame(comparison).T.reset_index()
        comparison_df = comparison_df.rename(
            columns={"index": "Scheduler"}
        )

        metric_choice = st.selectbox(
            "Select comparison metric",
            [
                "Carbon Emissions (gCO2)",
                "Energy (kWh)",
                "Cost"
            ]
        )

        comparison_fig = px.bar(
            comparison_df,
            x="Scheduler",
            y=metric_choice,
            color="Scheduler",
            text_auto=".4f",
            title=f"Baseline vs AI: {metric_choice}"
        )

        comparison_fig.update_layout(
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(
            comparison_fig,
            width="stretch"
        )

        st.subheader("Recommended Region Ranking")

        display_columns = [
            "provider",
            "region",
            "timestamp",
            "predicted_carbon",
            "estimated_energy_kwh",
            "estimated_cost",
            "score"
        ]

        st.dataframe(
            ranked[display_columns].head(20),
            width="stretch",
            hide_index=True
        )

        st.subheader("Carbon Intensity Across Scheduling Times")

        time_summary = (
            ranked.groupby("timestamp", as_index=False)
            ["predicted_carbon"]
            .min()
        )

        time_fig = px.line(
            time_summary,
            x="timestamp",
            y="predicted_carbon",
            markers=True,
            title="Lowest Predicted Carbon Intensity by Hour"
        )

        time_fig.update_layout(template="plotly_white")

        st.plotly_chart(
            time_fig,
            width="stretch"
        )

        st.download_button(
            "Download Optimization Results",
            data=ranked.to_csv(index=False),
            file_name="carbon_scheduler_results.csv",
            mime="text/csv"
        )

        st.success("Optimization results generated successfully.")
