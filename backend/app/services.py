import warnings
from functools import lru_cache
from pathlib import Path

import gdown
import joblib
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILES = {
    "energy_model_v2.pkl": "1ovkaPwuHaqJdRgDR8KazawStJwqIiWnV",
    "energy_model.pkl": "1WETsfnlnvBi-ozd8N3V0HkaEixWBfhUj",
}

NUMERIC_FEATURES = [
    "Temperature_C",
    "Pressure_Bar",
    "Humidity_Percent",
    "Motor_Speed_RPM",
    "Compression_Force_kN",
    "Flow_Rate_LPM",
    "Vibration_mm_s",
]

PHASE_COLUMN_MAP = {
    "Preparation": "Phase_Preparation",
    "Compression": "Phase_Compression",
    "Quality_Testing": "Phase_Quality_Testing",
}

PARAM_META = [
    {"key": "Temperature_C", "label": "Temperature", "unit": "C"},
    {"key": "Pressure_Bar", "label": "Pressure", "unit": "bar"},
    {"key": "Humidity_Percent", "label": "Humidity", "unit": "%"},
    {"key": "Motor_Speed_RPM", "label": "Motor Speed", "unit": "RPM"},
    {"key": "Compression_Force_kN", "label": "Compression Force", "unit": "kN"},
    {"key": "Flow_Rate_LPM", "label": "Flow Rate", "unit": "LPM"},
    {"key": "Vibration_mm_s", "label": "Vibration", "unit": "mm/s"},
]

ALERT_GUIDANCE = {
    "Temperature_C": "Inspect cooling flow and jacket control before increasing throughput.",
    "Pressure_Bar": "Check valve response and line restriction before the next operating window.",
    "Vibration_mm_s": "Inspect rotating equipment condition and verify alignment before continuing at full load.",
}


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


def infer_phase_from_row(row):
    for phase_name, phase_col in PHASE_COLUMN_MAP.items():
        if int(row.get(phase_col, 0)) == 1:
            return phase_name

    return "Preparation"


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


def classify_adjustment_effort(change_score):
    if change_score <= 0.08:
        return "Low"
    if change_score <= 0.16:
        return "Moderate"
    return "High"


def estimate_yield_uplift_percent(energy_saved, current_prediction):
    if current_prediction <= 0:
        return 0.0

    uplift = (energy_saved / current_prediction) * 55
    return float(np.clip(uplift, 0, 12))


def estimate_quality_confidence_percent(change_score, alert_count):
    confidence = 96 - (change_score * 140) - (alert_count * 4)
    return float(np.clip(confidence, 70, 99))


def ensure_models_downloaded():
    for filename, file_id in MODEL_FILES.items():
        path = PROJECT_ROOT / filename

        if not path.exists():
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(path), quiet=False)


def load_data():
    return pd.read_excel(PROJECT_ROOT / "new_batches.xlsx")


def load_training_data():
    path = PROJECT_ROOT / "training_data.xlsx"
    if path.exists():
        return pd.read_excel(path)
    return load_data().copy()


def load_or_train_model(train_df):
    candidate_paths = ["energy_model_v2.pkl", "energy_model.pkl"]

    for model_name in candidate_paths:
        path = PROJECT_ROOT / model_name
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
                return model, f"Loaded {model_name}"

    feature_drop = [
        "Power_Consumption_kW",
        "Batch_ID",
        "Time_Minutes",
    ]

    X = train_df.drop(columns=feature_drop, errors="ignore")
    y = train_df["Power_Consumption_kW"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    joblib.dump(model, PROJECT_ROOT / "energy_model_runtime.pkl")

    return model, "Trained fresh model"


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
        deviation = np.mean(np.abs((X - self.baseline) / self.scale), axis=1)

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
        ascending=[True, True],
    ).reset_index(drop=True)


class ManufacturingRepository:
    def __init__(self):
        ensure_models_downloaded()
        self.data = load_data()
        self.training_data = load_training_data()
        self.model, self.model_source = load_or_train_model(self.training_data)
        self.features = list(self.model.feature_names_in_)
        self.data = self.data.copy()
        self.data["Predicted_Energy"] = self.model.predict(
            build_feature_frame(self.data, self.features)
        )
        self.overall_mae = mean_absolute_error(
            self.data["Power_Consumption_kW"],
            self.data["Predicted_Energy"],
        )

    def get_latest_batch_id(self):
        batch_series = self.data["Batch_ID"].dropna().astype(str)

        if batch_series.empty:
            return None

        # Treat the last batch appearance in the dataset as the latest available batch.
        return batch_series.iloc[-1]

    def get_batch_ids(self):
        batch_ids = self.data["Batch_ID"].dropna().astype(str).unique().tolist()
        return sorted(batch_ids)

    def get_overview(self):
        return {
            "modelSource": self.model_source,
            "totalBatches": int(self.data["Batch_ID"].nunique()),
            "totalRecords": int(len(self.data)),
            "predictionMae": round(float(self.overall_mae), 3),
            "numericFeatures": NUMERIC_FEATURES,
            "parameterMeta": PARAM_META,
            "latestBatchId": self.get_latest_batch_id(),
        }

    def _get_batch_frame(self, batch_id):
        batch_data = self.data[self.data["Batch_ID"].astype(str) == str(batch_id)].copy()

        if batch_data.empty:
            raise KeyError(f"Unknown batch: {batch_id}")

        if "Time_Minutes" in batch_data.columns:
            batch_data = batch_data.sort_values("Time_Minutes")

        return batch_data.reset_index(drop=True)

    def _build_batch_context(self, batch_id):
        batch_data = self._get_batch_frame(batch_id)
        selected_row = batch_data.iloc[-1]
        selected_phase = infer_phase_from_row(selected_row)
        phase_flags = {
            "Phase_Preparation": 1 if selected_phase == "Preparation" else 0,
            "Phase_Compression": 1 if selected_phase == "Compression" else 0,
            "Phase_Quality_Testing": 1 if selected_phase == "Quality_Testing" else 0,
        }

        current_values = np.array(
            [float(selected_row[col]) for col in NUMERIC_FEATURES],
            dtype=float,
        )

        selected_input = pd.DataFrame([
            {
                **{col: float(selected_row[col]) for col in NUMERIC_FEATURES},
                **phase_flags,
            }
        ])
        selected_input = build_feature_frame(selected_input, self.features)

        selected_prediction = float(self.model.predict(selected_input)[0])
        actual_energy = float(selected_row["Power_Consumption_kW"])

        return {
            "batchData": batch_data,
            "selectedRow": selected_row,
            "selectedPhase": selected_phase,
            "phaseFlags": phase_flags,
            "currentValues": current_values,
            "selectedPrediction": selected_prediction,
            "actualEnergy": actual_energy,
        }

    def get_batch_dashboard(self, batch_id):
        context = self._build_batch_context(batch_id)
        batch_data = context["batchData"].copy()
        batch_data["Predicted_Energy"] = self.model.predict(
            build_feature_frame(batch_data, self.features)
        )

        if "Time_Minutes" in batch_data.columns:
            x_axis = "Time_Minutes"
        else:
            batch_data["Record_Index"] = np.arange(1, len(batch_data) + 1)
            x_axis = "Record_Index"

        trend_points = []
        for _, row in batch_data.iterrows():
            point = {
                "x": float(row[x_axis]),
                "actualEnergy": float(row["Power_Consumption_kW"]),
                "predictedEnergy": float(row["Predicted_Energy"]),
            }
            for feature in NUMERIC_FEATURES:
                point[feature] = float(row[feature])
            trend_points.append(point)

        phase_history = self.data[
            self.data[PHASE_COLUMN_MAP[context["selectedPhase"]]].astype(bool)
        ].copy()
        if phase_history.empty:
            phase_history = self.data.copy()

        thresholds = compute_phase_alert_thresholds(phase_history)
        alerts = []
        for metric_name, threshold in thresholds.items():
            current_value = float(context["selectedRow"][metric_name])
            if current_value >= threshold:
                alerts.append(
                    {
                        "metric": metric_name,
                        "value": round(current_value, 2),
                        "threshold": round(float(threshold), 2),
                        "message": (
                            f"{metric_name} is elevated at {current_value:.2f} "
                            f"(phase threshold {threshold:.2f})."
                        ),
                        "insight": ALERT_GUIDANCE.get(
                            metric_name,
                            "Review the associated process condition and return the signal inside the normal operating band.",
                        ),
                    }
                )

        phase_labels = np.select(
            [
                self.data["Phase_Preparation"].astype(bool),
                self.data["Phase_Compression"].astype(bool),
                self.data["Phase_Quality_Testing"].astype(bool),
            ],
            [
                "Preparation",
                "Compression",
                "Quality Testing",
            ],
            default="Transition / Support",
        )

        phase_summary = (
            self.data.assign(phaseLabel=phase_labels)
            .groupby("phaseLabel", as_index=False)["Power_Consumption_kW"]
            .mean()
            .sort_values("Power_Consumption_kW", ascending=False)
        )

        batch_summary = (
            self.data.groupby("Batch_ID", as_index=False)
            .agg(
                Avg_Energy=("Power_Consumption_kW", "mean"),
                Peak_Energy=("Power_Consumption_kW", "max"),
                Avg_Vibration=("Vibration_mm_s", "mean"),
            )
            .sort_values("Avg_Energy", ascending=False)
            .head(10)
        )

        feature_importance_df = (
            pd.DataFrame(
                {
                    "feature": self.features,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(10)
        )

        current_snapshot = {
            feature: round(float(context["selectedRow"][feature]), 3)
            for feature in NUMERIC_FEATURES
        }

        return {
            "batchId": str(batch_id),
            "xAxisLabel": "Time (Minutes)" if x_axis == "Time_Minutes" else "Record Index",
            "selectedPhase": context["selectedPhase"],
            "metrics": {
                "actualEnergy": round(context["actualEnergy"], 3),
                "predictedEnergy": round(context["selectedPrediction"], 3),
                "predictionError": round(
                    abs(context["actualEnergy"] - context["selectedPrediction"]),
                    3,
                ),
            },
            "currentSnapshot": current_snapshot,
            "trendData": trend_points,
            "alerts": alerts,
            "phaseSummary": [
                {
                    "phaseLabel": row["phaseLabel"],
                    "averageEnergy": round(float(row["Power_Consumption_kW"]), 3),
                }
                for _, row in phase_summary.iterrows()
            ],
            "batchSummary": [
                {
                    "batchId": str(row["Batch_ID"]),
                    "avgEnergy": round(float(row["Avg_Energy"]), 3),
                    "peakEnergy": round(float(row["Peak_Energy"]), 3),
                    "avgVibration": round(float(row["Avg_Vibration"]), 3),
                }
                for _, row in batch_summary.iterrows()
            ],
            "featureImportances": [
                {
                    "feature": row["feature"],
                    "importance": round(float(row["importance"]), 5),
                }
                for _, row in feature_importance_df.iterrows()
            ],
        }

    def get_optimization(self, batch_id):
        context = self._build_batch_context(batch_id)
        phase_history = self.data[
            self.data[PHASE_COLUMN_MAP[context["selectedPhase"]]].astype(bool)
        ].copy()
        if phase_history.empty:
            phase_history = self.data.copy()

        thresholds = compute_phase_alert_thresholds(phase_history)
        active_alert_count = 0
        for metric_name, threshold in thresholds.items():
            if float(context["selectedRow"][metric_name]) >= threshold:
                active_alert_count += 1

        candidates = generate_optimization_candidates(
            model=self.model,
            train_df=self.training_data,
            numeric_cols=NUMERIC_FEATURES,
            feature_order=self.features,
            phase_flags=context["phaseFlags"],
            baseline=context["currentValues"],
            current_prediction=context["selectedPrediction"],
        )

        best_candidate = candidates.iloc[0]
        comparison_rows = []
        lower_bounds, upper_bounds = compute_numeric_bounds(self.training_data, NUMERIC_FEATURES)
        significant_change_count = 0

        for index, feature in enumerate(NUMERIC_FEATURES):
            current_value = float(context["currentValues"][index])
            recommended_value = float(best_candidate[feature])
            normalized_change = abs(recommended_value - current_value) / max(
                float(upper_bounds[index] - lower_bounds[index]),
                1e-6,
            )
            if normalized_change >= 0.08:
                significant_change_count += 1
            comparison_rows.append(
                {
                    "parameter": feature,
                    "label": next(
                        (item["label"] for item in PARAM_META if item["key"] == feature),
                        feature,
                    ),
                    "unit": next(
                        (item["unit"] for item in PARAM_META if item["key"] == feature),
                        "",
                    ),
                    "current": round(current_value, 3),
                    "recommended": round(recommended_value, 3),
                    "delta": round(recommended_value - current_value, 3),
                }
            )

        pareto_points = []
        for _, row in candidates.iterrows():
            point = {
                "changeScore": round(float(row["Change_Score"]), 5),
                "predictedEnergy": round(float(row["Predicted_Energy"]), 5),
                "energySaved": round(float(row["Energy_Saved"]), 5),
            }
            for feature in NUMERIC_FEATURES:
                point[feature] = round(float(row[feature]), 3)
            pareto_points.append(point)

        return {
            "bestCandidate": {
                "predictedEnergy": round(float(best_candidate["Predicted_Energy"]), 3),
                "energySaved": round(float(best_candidate["Energy_Saved"]), 3),
                "changeScore": round(float(best_candidate["Change_Score"]), 5),
                "reductionPercent": round(
                    (float(best_candidate["Energy_Saved"]) / max(context["selectedPrediction"], 1e-6)) * 100,
                    2,
                ),
                "implementationEffort": classify_adjustment_effort(
                    float(best_candidate["Change_Score"])
                ),
                "parametersAdjusted": int(significant_change_count),
                "estimatedYieldUplift": round(
                    estimate_yield_uplift_percent(
                        float(best_candidate["Energy_Saved"]),
                        float(context["selectedPrediction"]),
                    ),
                    2,
                ),
                "qualityConfidence": round(
                    estimate_quality_confidence_percent(
                        float(best_candidate["Change_Score"]),
                        active_alert_count,
                    ),
                    2,
                ),
            },
            "comparison": comparison_rows,
            "paretoPoints": pareto_points,
            "optimalPoint": pareto_points[0],
        }


@lru_cache
def get_repository():
    return ManufacturingRepository()
