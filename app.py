import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import gdown

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
    "energy_model_v2.pkl": "https://drive.google.com/file/d/1ovkaPwuHaqJdRgDR8KazawStJwqIiWnV/view?usp=drive_link",
    "energy_model.pkl": "https://drive.google.com/file/d/1WETsfnlnvBi-ozd8N3V0HkaEixWBfhUj/view?usp=sharing"
}


def ensure_models_downloaded():
    """
    Download models from Google Drive if not present locally
    """
    for filename, file_id in MODEL_FILES.items():

        if file_id == "https://drive.google.com/file/d/1ovkaPwuHaqJdRgDR8KazawStJwqIiWnV/view?usp=drive_link":
            continue

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
# FOOTER
# ------------------------------------------------

st.markdown("---")
st.caption("AI-Driven Manufacturing Optimization Prototype")