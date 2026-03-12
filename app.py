import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import mean_absolute_error


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(page_title="Manufacturing Optimization Platform", layout="wide")
st.title("🏭 AI-Driven Manufacturing Optimization System")


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_excel("new_batches.xlsx")


@st.cache_data
def load_training_data():
    path = Path("training_data.xlsx")
    if path.exists():
        return pd.read_excel(path)
    return load_data().copy()


def build_feature_frame(df, feature_order):
    frame = df.copy()
    for col in feature_order:
        if col not in frame.columns:
            frame[col] = 0
    return frame[feature_order]


def evaluate_model_variability(model):
    features = list(model.feature_names_in_)

    base = {
        "Temperature_C": 150,
        "Pressure_Bar": 1.0,
        "Humidity_Percent": 50,
        "Motor_Speed_RPM": 25,
        "Compression_Force_kN": 120,
        "Flow_Rate_LPM": 12,
        "Vibration_mm_s": 2,
        "Phase_Compression": 0,
        "Phase_Preparation": 1,
        "Phase_Quality_Testing": 0,
    }

    probe_points = [
        base,
        {**base, "Temperature_C": 120},
        {**base, "Temperature_C": 180},
        {**base, "Compression_Force_kN": 60},
        {**base, "Compression_Force_kN": 190},
        {**base, "Vibration_mm_s": 0},
        {**base, "Vibration_mm_s": 5},
        {**base, "Phase_Preparation": 0, "Phase_Compression": 1},
        {**base, "Phase_Preparation": 0, "Phase_Quality_Testing": 1},
    ]

    probe_df = pd.DataFrame(probe_points)
    probe_df = build_feature_frame(probe_df, features)
    preds = model.predict(probe_df)
    return len(np.unique(np.round(preds, 4))) >= 3


@st.cache_resource
def load_or_train_model(train_df):
    candidate_paths = ["energy_model_v2.pkl", "energy_model.pkl"]

    for model_path in candidate_paths:
        path = Path(model_path)
        if not path.exists():
            continue

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = joblib.load(path)

        has_version_mismatch = any(
            isinstance(w.message, InconsistentVersionWarning)
            for w in captured
        )

        if hasattr(model, "feature_names_in_") and not has_version_mismatch:
            if evaluate_model_variability(model):
                return model, f"Loaded {model_path}"

    feature_drop = ["Power_Consumption_kW", "Batch_ID", "Time_Minutes"]
    X = train_df.drop(columns=feature_drop, errors="ignore")
    y = train_df["Power_Consumption_kW"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    joblib.dump(model, "energy_model_runtime.pkl")
    return model, "Trained fresh model (energy_model_runtime.pkl)"


def infer_phase_from_row(row, phase_column_map):
    for phase_name, phase_col in phase_column_map.items():
        if int(row.get(phase_col, 0)) == 1:
            return phase_name
    return "Preparation"


def normalize_array(values):
    denom = values.max() - values.min()
    if np.isclose(denom, 0):
        return np.zeros_like(values)
    return (values - values.min()) / denom


def compute_phase_alert_thresholds(phase_df):
    # Lightweight rule-based limits derived from phase behavior.
    return {
        "Temperature_C": float(phase_df["Temperature_C"].quantile(0.95)),
        "Pressure_Bar": float(phase_df["Pressure_Bar"].quantile(0.95)),
        "Vibration_mm_s": float(phase_df["Vibration_mm_s"].quantile(0.95)),
    }


# ------------------------------------------------
# LOAD MODEL + GLOBAL METRICS
# ------------------------------------------------

data = load_data()
training_data = load_training_data()
energy_model, model_source = load_or_train_model(training_data)

features = list(energy_model.feature_names_in_)
numeric_features = [
    "Temperature_C",
    "Pressure_Bar",
    "Humidity_Percent",
    "Motor_Speed_RPM",
    "Compression_Force_kN",
    "Flow_Rate_LPM",
    "Vibration_mm_s",
]
phase_column_map = {
    "Preparation": "Phase_Preparation",
    "Compression": "Phase_Compression",
    "Quality_Testing": "Phase_Quality_Testing",
}
param_meta = [
    ("Temperature_C", "Temperature", "C"),
    ("Pressure_Bar", "Pressure", "bar"),
    ("Humidity_Percent", "Humidity", "%"),
    ("Motor_Speed_RPM", "Motor Speed", "RPM"),
    ("Compression_Force_kN", "Compression Force", "kN"),
    ("Flow_Rate_LPM", "Flow Rate", "LPM"),
    ("Vibration_mm_s", "Vibration", "mm/s"),
]

all_X = build_feature_frame(data, features)
all_predictions = energy_model.predict(all_X)
data["Predicted_Energy"] = all_predictions
overall_mae = mean_absolute_error(data["Power_Consumption_kW"], all_predictions)


# ------------------------------------------------
# SYSTEM OVERVIEW
# ------------------------------------------------

st.subheader("System Overview")
st.caption(model_source)

c1, c2, c3 = st.columns(3)
c1.metric("Total Batches", data["Batch_ID"].nunique())
c2.metric("Total Records", len(data))
c3.metric("Prediction MAE", f"{overall_mae:.2f} kW")


# ------------------------------------------------
# BATCH-BY-BATCH MONITORING
# ------------------------------------------------

st.subheader("Batch Monitoring (Industrial Mode)")

batch_ids = sorted(data["Batch_ID"].dropna().unique().tolist())
selected_batch = st.selectbox("Select Batch", batch_ids)

batch_data = data[data["Batch_ID"] == selected_batch].copy()
if "Time_Minutes" in batch_data.columns:
    batch_data = batch_data.sort_values("Time_Minutes")

batch_data = batch_data.reset_index(drop=True)
selected_row = batch_data.iloc[-1]
selected_phase = infer_phase_from_row(selected_row, phase_column_map)

phase_flags = {
    "Phase_Preparation": 1 if selected_phase == "Preparation" else 0,
    "Phase_Compression": 1 if selected_phase == "Compression" else 0,
    "Phase_Quality_Testing": 1 if selected_phase == "Quality_Testing" else 0,
}

current_values = np.array([float(selected_row[col]) for col in numeric_features], dtype=float)

selected_input = pd.DataFrame([
    {
        **{col: float(selected_row[col]) for col in numeric_features},
        **phase_flags,
    }
])
selected_input = build_feature_frame(selected_input, features)
selected_prediction = float(energy_model.predict(selected_input)[0])
actual_energy = float(selected_row["Power_Consumption_kW"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Batch", str(selected_batch))
m2.metric("Phase", selected_phase)
m3.metric("Actual Energy", f"{actual_energy:.2f} kW")
m4.metric("Predicted Energy", f"{selected_prediction:.2f} kW")
if "Time_Minutes" in batch_data.columns:
    st.caption(f"Using latest reading at {float(selected_row['Time_Minutes']):.2f} min")
else:
    st.caption("Using latest available reading")

st.metric("Prediction Error", f"{abs(actual_energy - selected_prediction):.2f} kW")

param_rows = [
    {
        "Parameter": label,
        "Current Value": f"{float(selected_row[col]):.3f} {unit}",
    }
    for col, label, unit in param_meta
]
st.dataframe(pd.DataFrame(param_rows), use_container_width=True)


# ------------------------------------------------
# BATCH ANALYTICS (SELECTED BATCH ONLY)
# ------------------------------------------------

st.subheader("Selected Batch Trend")

if "Time_Minutes" in batch_data.columns and len(batch_data) > 1:
    trend_df = batch_data[["Time_Minutes", "Power_Consumption_kW", "Predicted_Energy"]].copy()
    trend_df = trend_df.rename(
        columns={
            "Power_Consumption_kW": "Actual Energy (kW)",
            "Predicted_Energy": "Predicted Energy (kW)",
        }
    )

    fig = px.line(
        trend_df,
        x="Time_Minutes",
        y=["Actual Energy (kW)", "Predicted Energy (kW)"],
        title="Energy Trend for Selected Batch",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    comp_df = pd.DataFrame(
        {
            "Type": ["Actual Energy", "Predicted Energy"],
            "kW": [actual_energy, selected_prediction],
        }
    )
    fig = px.bar(comp_df, x="Type", y="kW", title="Selected Reading: Actual vs Predicted")
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------
# OPTIMIZATION ENGINE (BATCH-AWARE)
# ------------------------------------------------

st.subheader("Batch-Aware Optimization")
search_radius_pct = st.slider(
    "Local Search Radius Around Current Batch (%)",
    min_value=5,
    max_value=40,
    value=20,
    step=1,
)


class ManufacturingOptimization(Problem):
    def __init__(self, center_point, selected_phase_flags, phase_bounds, radius_ratio):
        self.selected_phase_flags = selected_phase_flags

        lb_global = np.array([phase_bounds[col][0] for col in numeric_features], dtype=float)
        ub_global = np.array([phase_bounds[col][1] for col in numeric_features], dtype=float)

        span = ub_global - lb_global
        local_radius = radius_ratio * span

        local_lb = np.maximum(lb_global, center_point - local_radius)
        local_ub = np.minimum(ub_global, center_point + local_radius)

        fixed_mask = np.isclose(lb_global, ub_global)
        local_lb[fixed_mask] = lb_global[fixed_mask]
        local_ub[fixed_mask] = ub_global[fixed_mask]

        super().__init__(n_var=7, n_obj=4, n_constr=0, xl=local_lb, xu=local_ub)

    def _evaluate(self, X, out, *args, **kwargs):
        results = []

        for row in X:
            row_payload = {
                "Temperature_C": row[0],
                "Pressure_Bar": row[1],
                "Humidity_Percent": row[2],
                "Motor_Speed_RPM": row[3],
                "Compression_Force_kN": row[4],
                "Flow_Rate_LPM": row[5],
                "Vibration_mm_s": row[6],
                **self.selected_phase_flags,
            }

            df = build_feature_frame(pd.DataFrame([row_payload]), features)
            energy = float(energy_model.predict(df)[0])
            carbon = energy * 0.4

            temp = (row[0] - phase_bounds["Temperature_C"][0]) / (
                phase_bounds["Temperature_C"][1] - phase_bounds["Temperature_C"][0] + 1e-9
            )
            press = (row[1] - phase_bounds["Pressure_Bar"][0]) / (
                phase_bounds["Pressure_Bar"][1] - phase_bounds["Pressure_Bar"][0] + 1e-9
            )
            speed = (row[3] - phase_bounds["Motor_Speed_RPM"][0]) / (
                phase_bounds["Motor_Speed_RPM"][1] - phase_bounds["Motor_Speed_RPM"][0] + 1e-9
            )
            force = (row[4] - phase_bounds["Compression_Force_kN"][0]) / (
                phase_bounds["Compression_Force_kN"][1] - phase_bounds["Compression_Force_kN"][0] + 1e-9
            )

            yield_val = 80 + 20 * (0.6 * speed + 0.4 * force) - 0.2 * energy
            quality = 85 + 15 * (0.5 * temp + 0.3 * press) - 0.15 * energy

            results.append([energy, carbon, -yield_val, -quality])

        out["F"] = np.array(results)


phase_train = training_data[training_data[phase_column_map[selected_phase]] == 1].copy()
if phase_train.empty:
    phase_train = training_data.copy()

phase_bounds = {
    col: (float(phase_train[col].min()), float(phase_train[col].max()))
    for col in numeric_features
}
alert_thresholds = compute_phase_alert_thresholds(phase_train)


# ------------------------------------------------
# PROCESS ALERTS
# ------------------------------------------------

st.subheader("Process Alerts")

alerts = []
temperature_now = float(selected_row["Temperature_C"])
pressure_now = float(selected_row["Pressure_Bar"])
vibration_now = float(selected_row["Vibration_mm_s"])
energy_error_kw = abs(actual_energy - selected_prediction)
energy_error_pct = (
    energy_error_kw / max(abs(actual_energy), 1e-6)
) * 100.0

if temperature_now > alert_thresholds["Temperature_C"]:
    alerts.append(
        f"High Temperature: {temperature_now:.2f} C exceeds safe threshold "
        f"{alert_thresholds['Temperature_C']:.2f} C."
    )

if pressure_now > alert_thresholds["Pressure_Bar"]:
    alerts.append(
        f"High Pressure: {pressure_now:.3f} bar exceeds safe threshold "
        f"{alert_thresholds['Pressure_Bar']:.3f} bar."
    )

if vibration_now > alert_thresholds["Vibration_mm_s"]:
    alerts.append(
        f"Excessive Vibration: {vibration_now:.3f} mm/s exceeds safe threshold "
        f"{alert_thresholds['Vibration_mm_s']:.3f} mm/s."
    )

if energy_error_kw > 1.5 and energy_error_pct > 15:
    alerts.append(
        f"Energy Deviation: prediction differs from actual by "
        f"{energy_error_kw:.2f} kW ({energy_error_pct:.1f}%)."
    )

if alerts:
    st.warning("Early warning detected. Review the following alerts:")
    for msg in alerts:
        st.error(msg)
else:
    st.success("Process is operating within safe parameters for the current batch state.")

with st.spinner("Running optimization from latest batch state..."):
    problem = ManufacturingOptimization(
        center_point=current_values,
        selected_phase_flags=phase_flags,
        phase_bounds=phase_bounds,
        radius_ratio=search_radius_pct / 100.0,
    )

    algorithm = NSGA2(pop_size=60)
    result = minimize(problem, algorithm, ("n_gen", 60), seed=1, verbose=False)

energy = result.F[:, 0]
yield_vals = -result.F[:, 2]
quality_vals = -result.F[:, 3]

energy_norm = normalize_array(energy)
yield_norm = normalize_array(yield_vals)
quality_norm = normalize_array(quality_vals)

points = np.vstack([energy_norm, yield_norm, quality_norm]).T
perf_distance = np.linalg.norm(points - np.array([0.5, 0.5, 0.5]), axis=1)

param_distance = np.linalg.norm(result.X - current_values, axis=1)
param_distance_norm = param_distance / (param_distance.max() + 1e-9)

balanced_idx = int(np.argmin(perf_distance + 0.35 * param_distance_norm))

pareto_df = pd.DataFrame({"Energy": energy, "Yield": yield_vals})
fig = px.scatter(pareto_df, x="Energy", y="Yield", title="Pareto Front: Energy vs Yield")

fig.add_scatter(
    x=[energy[balanced_idx]],
    y=[yield_vals[balanced_idx]],
    mode="markers",
    marker=dict(size=14, color="red"),
    name="Balanced Solution",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Recommended Balanced Configuration")

rm1, rm2, rm3 = st.columns(3)
rm1.metric("Energy", f"{energy[balanced_idx]:.2f} kW")
rm2.metric("Yield", f"{yield_vals[balanced_idx]:.2f} %")
rm3.metric("Quality", f"{quality_vals[balanced_idx]:.2f} %")

rec_rows = []
for idx, (col, label, unit) in enumerate(param_meta):
    current_val = float(current_values[idx])
    recommended_val = float(result.X[balanced_idx][idx])
    rec_rows.append(
        {
            "Parameter": label,
            "Current": f"{current_val:.3f} {unit}",
            "Recommended": f"{recommended_val:.3f} {unit}",
            "Delta": f"{(recommended_val - current_val):+.3f} {unit}",
        }
    )

st.dataframe(pd.DataFrame(rec_rows), use_container_width=True)


# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("---")
st.caption("AI-Driven Manufacturing Optimization Prototype")
