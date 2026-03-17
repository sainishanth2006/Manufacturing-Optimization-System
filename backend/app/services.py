import warnings
from functools import lru_cache
from pathlib import Path
import os

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

DEFAULT_RETRAIN_THRESHOLD_PERCENT = 15.0
DEFAULT_CARBON_ALERT_PERCENT = 85.0
CARBON_EMISSION_FACTOR = 0.82

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


def classify_retraining_recommendation(drift_percent, retrain_threshold_percent):
    if retrain_threshold_percent <= 0:
        return "Review"

    if drift_percent >= retrain_threshold_percent:
        return "Recommended"
    if drift_percent >= retrain_threshold_percent * 0.7:
        return "Monitor"
    return "Stable"


def calculate_carbon_emission(energy_value):
    return float(energy_value) * CARBON_EMISSION_FACTOR


def classify_carbon_status(carbon_emission, carbon_limit):
    if carbon_limit <= 0:
        return "Review"
    if carbon_emission >= carbon_limit:
        return "Exceeded"
    if carbon_emission >= carbon_limit * (DEFAULT_CARBON_ALERT_PERCENT / 100):
        return "Monitor"
    return "Stable"


def build_carbon_reduction_suggestions(selected_row, phase_history):
    recommendations = []
    prioritized_features = [
        (
            "Temperature_C",
            "Lower the process temperature toward the phase median of {target:.2f} C to cut avoidable energy draw.",
        ),
        (
            "Motor_Speed_RPM",
            "Trim motor speed toward {target:.2f} RPM where throughput allows to reduce carbon intensity.",
        ),
        (
            "Compression_Force_kN",
            "Reduce compression force closer to {target:.2f} kN while staying inside quality limits.",
        ),
        (
            "Flow_Rate_LPM",
            "Tune flow rate toward {target:.2f} LPM to avoid excess pumping energy.",
        ),
        (
            "Pressure_Bar",
            "Bring line pressure closer to {target:.2f} bar and check for unnecessary over-pressurization.",
        ),
        (
            "Vibration_mm_s",
            "Investigate vibration and restore it toward {target:.2f} mm/s to reduce mechanical losses.",
        ),
    ]

    for feature, template in prioritized_features:
        if feature not in phase_history.columns:
            continue

        target = float(phase_history[feature].median())
        current = float(selected_row[feature])
        tolerance = max(abs(target) * 0.05, 0.1)
        if current > target + tolerance:
            recommendations.append(
                {
                    "feature": feature,
                    "current": round(current, 3),
                    "target": round(target, 3),
                    "message": template.format(target=target),
                }
            )

        if len(recommendations) == 3:
            break

    if not recommendations:
        recommendations.append(
            {
                "feature": "general",
                "current": None,
                "target": None,
                "message": (
                    "Hold the current operating window steady and run the Optimization Advisor to identify the lowest-carbon feasible settings for this batch."
                ),
            }
        )

    return recommendations


def ensure_models_downloaded():
    allow_remote_models = os.getenv("ENABLE_MODEL_DOWNLOAD", "false").lower() == "true"

    if not allow_remote_models:
        return

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


def train_model_from_dataframe(train_df):
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

    return model


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
        self.default_carbon_limit = float(
            self.data["Power_Consumption_kW"]
            .apply(calculate_carbon_emission)
            .quantile(0.9)
        )
        self.last_retrained_batch_id = None
        self.model_monitoring = self._build_model_monitoring(self.get_latest_batch_id())

    def get_latest_batch_id(self):
        batch_series = self.data["Batch_ID"].dropna().astype(str)

        if batch_series.empty:
            return None

        # Treat the last batch appearance in the dataset as the latest available batch.
        return batch_series.iloc[-1]

    def get_batch_ids(self):
        batch_ids = self.data["Batch_ID"].dropna().astype(str).unique().tolist()
        return sorted(batch_ids)

    def _refresh_predictions(self):
        self.features = list(self.model.feature_names_in_)
        self.data = self.data.copy()
        self.data["Predicted_Energy"] = self.model.predict(
            build_feature_frame(self.data, self.features)
        )
        self.overall_mae = mean_absolute_error(
            self.data["Power_Consumption_kW"],
            self.data["Predicted_Energy"],
        )

    def _retrain_model_for_batch(self, batch_id):
        batch_frame = self._get_batch_frame(batch_id)
        combined_training = pd.concat(
            [self.training_data, batch_frame],
            ignore_index=True,
        ).drop_duplicates()
        self.training_data = combined_training.reset_index(drop=True)
        self.model = train_model_from_dataframe(self.training_data)
        self.model_source = f"Auto-retrained using {batch_id}"
        self.last_retrained_batch_id = str(batch_id)
        self._refresh_predictions()

    def _build_model_monitoring(
        self,
        batch_id,
        retrain_threshold_percent=DEFAULT_RETRAIN_THRESHOLD_PERCENT,
        auto_retrain=False,
    ):
        if not batch_id:
            return {
                "batchId": None,
                "batchMae": 0.0,
                "baselineMae": round(float(self.overall_mae), 3),
                "driftPercent": 0.0,
                "retrainThresholdPercent": round(float(retrain_threshold_percent), 2),
                "autoRetrainEnabled": bool(auto_retrain),
                "modelRetrained": False,
                "status": "Stable",
                "insight": "No batch was available for retraining assessment.",
            }

        batch_frame = self._get_batch_frame(batch_id).copy()
        batch_predictions = self.model.predict(
            build_feature_frame(batch_frame, self.features)
        )
        batch_mae = mean_absolute_error(
            batch_frame["Power_Consumption_kW"],
            batch_predictions,
        )
        baseline_mae = float(self.overall_mae)
        drift_percent = (
            ((batch_mae - baseline_mae) / baseline_mae) * 100 if baseline_mae > 0 else 0.0
        )
        should_retrain = (
            auto_retrain
            and drift_percent >= retrain_threshold_percent
            and self.last_retrained_batch_id != str(batch_id)
        )

        if should_retrain:
            self._retrain_model_for_batch(batch_id)
            retrained_monitoring = self._build_model_monitoring(
                batch_id,
                retrain_threshold_percent=retrain_threshold_percent,
                auto_retrain=False,
            )
            retrained_monitoring["modelRetrained"] = True
            retrained_monitoring["autoRetrainEnabled"] = True
            retrained_monitoring["insight"] = (
                "Error drift crossed the configured threshold, so the model was retrained automatically for this session."
            )
            return retrained_monitoring

        status = classify_retraining_recommendation(
            drift_percent,
            retrain_threshold_percent,
        )

        if status == "Recommended":
            insight = (
                "Selected-batch prediction error is above the configured drift threshold. Auto-retraining can be triggered for this batch."
            )
        elif status == "Monitor":
            insight = (
                "Selected-batch performance is drifting upward toward the retraining threshold. Continue monitoring closely."
            )
        else:
            insight = (
                "Selected-batch performance remains within the expected error band, so the current model is still suitable for operational use."
            )

        return {
            "batchId": str(batch_id),
            "batchMae": round(float(batch_mae), 3),
            "baselineMae": round(float(baseline_mae), 3),
            "driftPercent": round(float(drift_percent), 2),
            "retrainThresholdPercent": round(float(retrain_threshold_percent), 2),
            "autoRetrainEnabled": bool(auto_retrain),
            "modelRetrained": False,
            "status": status,
            "insight": insight,
        }

    def get_overview(self):
        return {
            "modelSource": self.model_source,
            "totalBatches": int(self.data["Batch_ID"].nunique()),
            "totalRecords": int(len(self.data)),
            "predictionMae": round(float(self.overall_mae), 3),
            "numericFeatures": NUMERIC_FEATURES,
            "parameterMeta": PARAM_META,
            "latestBatchId": self.get_latest_batch_id(),
            "defaultRetrainThresholdPercent": DEFAULT_RETRAIN_THRESHOLD_PERCENT,
            "defaultCarbonLimitKg": round(float(self.default_carbon_limit), 2),
            "modelMonitoring": self.model_monitoring,
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
        actual_carbon_emission = calculate_carbon_emission(actual_energy)
        predicted_carbon_emission = calculate_carbon_emission(selected_prediction)

        return {
            "batchData": batch_data,
            "selectedRow": selected_row,
            "selectedPhase": selected_phase,
            "phaseFlags": phase_flags,
            "currentValues": current_values,
            "selectedPrediction": selected_prediction,
            "actualEnergy": actual_energy,
            "actualCarbonEmission": actual_carbon_emission,
            "predictedCarbonEmission": predicted_carbon_emission,
        }

    def get_batch_dashboard(
        self,
        batch_id,
        retrain_threshold_percent=DEFAULT_RETRAIN_THRESHOLD_PERCENT,
        auto_retrain=False,
        carbon_limit_kg=None,
    ):
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

        resolved_carbon_limit = (
            float(carbon_limit_kg)
            if carbon_limit_kg is not None
            else float(self.default_carbon_limit)
        )
        carbon_status = classify_carbon_status(
            context["actualCarbonEmission"],
            resolved_carbon_limit,
        )
        carbon_delta = context["actualCarbonEmission"] - resolved_carbon_limit
        carbon_suggestions = build_carbon_reduction_suggestions(
            context["selectedRow"],
            phase_history,
        )

        default_thresholds = compute_phase_alert_thresholds(phase_history)
        thresholds = default_thresholds
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

        if carbon_status == "Exceeded":
            alerts.append(
                {
                    "metric": "Carbon_Emission",
                    "value": round(float(context["actualCarbonEmission"]), 2),
                    "threshold": round(float(resolved_carbon_limit), 2),
                    "message": (
                        f"Carbon footprint is elevated at {context['actualCarbonEmission']:.2f} kg CO2 "
                        f"(limit {resolved_carbon_limit:.2f} kg CO2)."
                    ),
                    "insight": carbon_suggestions[0]["message"],
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
                Total_Energy=("Power_Consumption_kW", "sum"),
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
                "actualCarbonEmission": round(context["actualCarbonEmission"], 3),
                "predictedCarbonEmission": round(context["predictedCarbonEmission"], 3),
                "predictionError": round(
                    abs(context["actualEnergy"] - context["selectedPrediction"]),
                    3,
                ),
            },
            "currentSnapshot": current_snapshot,
            "trendData": trend_points,
            "defaultThresholds": {
                metric_name: round(float(value), 2) for metric_name, value in default_thresholds.items()
            },
            "activeThresholds": {
                metric_name: round(float(value), 2) for metric_name, value in thresholds.items()
            },
            "modelMonitoring": self._build_model_monitoring(
                batch_id,
                retrain_threshold_percent=retrain_threshold_percent,
                auto_retrain=auto_retrain,
            ),
            "carbonMonitoring": {
                "actualCarbonEmission": round(float(context["actualCarbonEmission"]), 3),
                "predictedCarbonEmission": round(float(context["predictedCarbonEmission"]), 3),
                "carbonLimitKg": round(float(resolved_carbon_limit), 3),
                "deltaKg": round(float(carbon_delta), 3),
                "exceeded": bool(carbon_status == "Exceeded"),
                "status": carbon_status,
                "insight": (
                    "Carbon footprint exceeded the configured limit. Apply the recommended process adjustments to bring the batch back inside the carbon guardrail."
                    if carbon_status == "Exceeded"
                    else "Carbon footprint is approaching the configured limit. Consider small operating adjustments before it breaches the guardrail."
                    if carbon_status == "Monitor"
                    else "Carbon footprint remains inside the configured limit for the selected batch."
                ),
                "suggestions": carbon_suggestions,
            },
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
                    "carbonEmission": round(
                        calculate_carbon_emission(float(row["Total_Energy"])),
                        3,
                    ),
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
