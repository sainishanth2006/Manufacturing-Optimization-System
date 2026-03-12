import warnings
from pathlib import Path
import subprocess
import sys

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
# MODEL DOWNLOAD CONFIG
# ------------------------------------------------

MODEL_FILES = {
    "energy_model_v2.pkl": "1ovkaPwuHaqJdRgDR8KazawStJwqIiWnV",
    "energy_model.pkl": "1WETsfnlnvBi-ozd8N3V0HkaEixWBfhUj",
}


def get_gdown():
    """
    Import gdown, installing it into the active Python environment if needed.
    """
    try:
        import gdown as installed_gdown
    except ModuleNotFoundError:
        with st.spinner("Installing missing dependency: gdown"):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown as installed_gdown

    return installed_gdown


def ensure_models_downloaded():
    """
    Download models from Google Drive if not present locally
    """
    gdown = get_gdown()

    for filename, file_id in MODEL_FILES.items():
        path = Path(filename)

        if not path.exists():
            url = f"https://drive.google.com/uc?id={file_id}"
            st.info(f"Downloading {filename} from cloud storage...")
            gdown.download(url, filename, quiet=False)


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


# ------------------------------------------------
# MODEL LOADER
# ------------------------------------------------

@st.cache_resource
def load_or_train_model(train_df):

    candidate_paths = [
        "energy_model_v2.pkl",
        "energy_model.pkl"
    ]

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

    feature_drop = [
        "Power_Consumption_kW",
        "Batch_ID",
        "Time_Minutes"
    ]

    X = train_df.drop(columns=feature_drop, errors="ignore")
    y = train_df["Power_Consumption_kW"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    joblib.dump(model, "energy_model_runtime.pkl")

    return model, "Trained fresh model"


# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------

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

    return {
        "Temperature_C": float(phase_df["Temperature_C"].quantile(0.95)),
        "Pressure_Bar": float(phase_df["Pressure_Bar"].quantile(0.95)),
        "Vibration_mm_s": float(phase_df["Vibration_mm_s"].quantile(0.95)),
    }


def compute_numeric_bounds(df, columns):

    lower = df[columns].quantile(0.05).to_numpy(dtype=float)
    upper = df[columns].quantile(0.95).to_numpy(dtype=float)

    equal_mask = np.isclose(lower, upper)
    upper[equal_mask] = lower[equal_mask] + 1.0

    return lower, upper


class EnergyOptimizationProblem(Problem):

    def __init__(self, model, numeric_cols, feature_order, phase_flags, baseline, lower, upper):

        super().__init__(
            n_var=len(numeric_cols),
            n_obj=2,
            n_constr=0,
            xl=lower,
            xu=upper,
        )
        self.model = model
        self.numeric_cols = numeric_cols
        self.feature_order = feature_order
        self.phase_flags = phase_flags
        self.baseline = np.array(baseline, dtype=float)
        self.scale = np.maximum(upper - lower, 1e-6)

    def _evaluate(self, X, out, *args, **kwargs):

        candidate_df = pd.DataFrame(X, columns=self.numeric_cols)

        for phase_col, phase_value in self.phase_flags.items():
            candidate_df[phase_col] = phase_value

        candidate_df = build_feature_frame(candidate_df, self.feature_order)

        energy = self.model.predict(candidate_df)

        deviation = np.mean(
            np.abs((X - self.baseline) / self.scale),
            axis=1
        )

        out["F"] = np.column_stack([energy, deviation])


def generate_optimization_candidates(
    model,
    train_df,
    numeric_cols,
    feature_order,
    phase_flags,
    baseline,
    current_prediction,
):

    lower, upper = compute_numeric_bounds(train_df, numeric_cols)

    problem = EnergyOptimizationProblem(
        model=model,
        numeric_cols=numeric_cols,
        feature_order=feature_order,
        phase_flags=phase_flags,
        baseline=baseline,
        lower=lower,
        upper=upper,
    )

    result = minimize(
        problem,
        NSGA2(pop_size=40),
        ("n_gen", 20),
        seed=42,
        verbose=False,
    )

    candidates = pd.DataFrame(result.X, columns=numeric_cols)
    candidates["Predicted_Energy"] = result.F[:, 0]
    candidates["Change_Score"] = result.F[:, 1]
    candidates["Energy_Saved"] = float(current_prediction) - candidates["Predicted_Energy"]

    return candidates.sort_values(
        ["Predicted_Energy", "Change_Score"],
        ascending=[True, True]
    ).reset_index(drop=True)


# ------------------------------------------------
# ENSURE MODELS EXIST
# ------------------------------------------------

ensure_models_downloaded()


# ------------------------------------------------
# LOAD MODEL + DATA
# ------------------------------------------------

data = load_data()
training_data = load_training_data()

energy_model, model_source = load_or_train_model(training_data)


# ------------------------------------------------
# FEATURES
# ------------------------------------------------

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


# ------------------------------------------------
# GLOBAL METRICS
# ------------------------------------------------

all_X = build_feature_frame(data, features)

all_predictions = energy_model.predict(all_X)

data["Predicted_Energy"] = all_predictions

overall_mae = mean_absolute_error(
    data["Power_Consumption_kW"],
    all_predictions
)


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
# BATCH MONITORING
# ------------------------------------------------

st.subheader("Batch Monitoring (Industrial Mode)")

batch_ids = sorted(
    data["Batch_ID"].dropna().unique().tolist()
)

selected_batch = st.selectbox("Select Batch", batch_ids)

batch_data = data[data["Batch_ID"] == selected_batch].copy()

if "Time_Minutes" in batch_data.columns:

    batch_data = batch_data.sort_values("Time_Minutes")

batch_data = batch_data.reset_index(drop=True)

selected_row = batch_data.iloc[-1]

selected_phase = infer_phase_from_row(
    selected_row,
    phase_column_map
)

phase_flags = {
    "Phase_Preparation": 1 if selected_phase == "Preparation" else 0,
    "Phase_Compression": 1 if selected_phase == "Compression" else 0,
    "Phase_Quality_Testing": 1 if selected_phase == "Quality_Testing" else 0,
}

current_values = np.array(
    [float(selected_row[col]) for col in numeric_features],
    dtype=float
)

selected_input = pd.DataFrame([
    {
        **{col: float(selected_row[col]) for col in numeric_features},
        **phase_flags,
    }
])

selected_input = build_feature_frame(selected_input, features)

selected_prediction = float(
    energy_model.predict(selected_input)[0]
)

actual_energy = float(
    selected_row["Power_Consumption_kW"]
)


m1, m2, m3, m4 = st.columns(4)

m1.metric("Batch", str(selected_batch))
m2.metric("Phase", selected_phase)
m3.metric("Actual Energy", f"{actual_energy:.2f} kW")
m4.metric("Predicted Energy", f"{selected_prediction:.2f} kW")


st.metric(
    "Prediction Error",
    f"{abs(actual_energy - selected_prediction):.2f} kW"
)


# ------------------------------------------------
# BATCH VISUALS
# ------------------------------------------------

st.subheader("Batch Trend Analysis")

batch_plot = batch_data.copy()
batch_plot["Predicted_Energy"] = energy_model.predict(
    build_feature_frame(batch_plot, features)
)

if "Time_Minutes" in batch_plot.columns:
    x_axis = "Time_Minutes"
else:
    batch_plot["Record_Index"] = np.arange(1, len(batch_plot) + 1)
    x_axis = "Record_Index"

energy_trend = px.line(
    batch_plot,
    x=x_axis,
    y=["Power_Consumption_kW", "Predicted_Energy"],
    labels={
        "value": "Energy (kW)",
        "variable": "Series",
        "Time_Minutes": "Time (Minutes)",
    },
    title=f"Energy Profile for {selected_batch}",
)
st.plotly_chart(energy_trend, use_container_width=True)

parameter_trend = px.line(
    batch_plot,
    x=x_axis,
    y=numeric_features,
    labels={
        "value": "Sensor Reading",
        "variable": "Parameter",
        "Time_Minutes": "Time (Minutes)",
    },
    title="Process Parameters Across the Selected Batch",
)
st.plotly_chart(parameter_trend, use_container_width=True)


# ------------------------------------------------
# ALERTS
# ------------------------------------------------

st.subheader("Operational Alerts")

phase_history = data[
    data[phase_column_map[selected_phase]].astype(bool)
].copy()

if phase_history.empty:
    phase_history = data.copy()

thresholds = compute_phase_alert_thresholds(phase_history)

alert_messages = []

for metric_name, threshold in thresholds.items():
    current_metric_value = float(selected_row[metric_name])
    if current_metric_value >= threshold:
        alert_messages.append(
            f"{metric_name} is elevated at {current_metric_value:.2f} "
            f"(phase threshold {threshold:.2f})."
        )

if alert_messages:
    for message in alert_messages:
        st.warning(message)
else:
    st.success(f"No active threshold breaches detected for the {selected_phase} phase.")


# ------------------------------------------------
# FLEET INSIGHTS
# ------------------------------------------------

st.subheader("Fleet Insights")

phase_labels = np.select(
    [
        data["Phase_Preparation"].astype(bool),
        data["Phase_Compression"].astype(bool),
        data["Phase_Quality_Testing"].astype(bool),
    ],
    [
        "Preparation",
        "Compression",
        "Quality Testing",
    ],
    default="Unknown",
)

phase_summary = (
    data.assign(Phase_Label=phase_labels)
    .groupby("Phase_Label", as_index=False)["Power_Consumption_kW"]
    .mean()
    .sort_values("Power_Consumption_kW", ascending=False)
)

phase_chart = px.bar(
    phase_summary,
    x="Phase_Label",
    y="Power_Consumption_kW",
    color="Phase_Label",
    title="Average Energy Consumption by Phase",
    labels={
        "Phase_Label": "Phase",
        "Power_Consumption_kW": "Average Energy (kW)",
    },
)
st.plotly_chart(phase_chart, use_container_width=True)

batch_summary = (
    data.groupby("Batch_ID", as_index=False)
    .agg(
        Avg_Energy=("Power_Consumption_kW", "mean"),
        Peak_Energy=("Power_Consumption_kW", "max"),
        Avg_Vibration=("Vibration_mm_s", "mean"),
    )
    .sort_values("Avg_Energy", ascending=False)
)

left_col, right_col = st.columns(2)

left_col.dataframe(
    batch_summary.head(10),
    use_container_width=True,
    hide_index=True,
)

feature_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": energy_model.feature_importances_,
}).sort_values("Importance", ascending=False)

importance_chart = px.bar(
    feature_importance_df.head(10),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Drivers of Predicted Energy",
)
right_col.plotly_chart(importance_chart, use_container_width=True)


# ------------------------------------------------
# OPTIMIZATION ADVISOR
# ------------------------------------------------

st.subheader("Optimization Advisor")

with st.expander("Generate operating recommendations", expanded=True):
    if st.button("Run Recommendation Search", use_container_width=True):
        candidates = generate_optimization_candidates(
            model=energy_model,
            train_df=training_data,
            numeric_cols=numeric_features,
            feature_order=features,
            phase_flags=phase_flags,
            baseline=current_values,
            current_prediction=selected_prediction,
        )

        best_candidate = candidates.iloc[0]

        rec1, rec2, rec3 = st.columns(3)
        rec1.metric("Recommended Energy", f"{best_candidate['Predicted_Energy']:.2f} kW")
        rec2.metric("Estimated Savings", f"{best_candidate['Energy_Saved']:.2f} kW")
        rec3.metric("Change Score", f"{best_candidate['Change_Score']:.3f}")

        comparison_df = pd.DataFrame({
            "Parameter": numeric_features,
            "Current": current_values,
            "Recommended": [best_candidate[col] for col in numeric_features],
        })
        comparison_df["Delta"] = comparison_df["Recommended"] - comparison_df["Current"]

        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        pareto_chart = px.scatter(
            candidates,
            x="Change_Score",
            y="Predicted_Energy",
            color="Energy_Saved",
            hover_data=numeric_features,
            title="Recommendation Trade-off: Change Magnitude vs Energy",
            labels={
                "Change_Score": "Operational Change Score",
                "Predicted_Energy": "Predicted Energy (kW)",
            },
        )
        pareto_chart.add_scatter(
            x=[best_candidate["Change_Score"]],
            y=[best_candidate["Predicted_Energy"]],
            mode="markers+text",
            name="Optimal Point",
            text=["Optimal"],
            textposition="top center",
            marker={
                "size": 16,
                "symbol": "star",
                "color": "#d62728",
                "line": {"width": 2, "color": "#ffffff"},
            },
            hovertemplate=(
                "Optimal Point"
                "<br>Operational Change Score: %{x:.3f}"
                "<br>Predicted Energy: %{y:.2f} kW"
                "<extra></extra>"
            ),
        )
        st.plotly_chart(pareto_chart, use_container_width=True)
    else:
        st.caption("Run the search to generate energy-saving parameter recommendations for the selected batch.")


# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("---")
st.caption("AI-Driven Manufacturing Optimization Prototype")
