"""Train and evaluate uncertainty-aware RUL models for NASA C-MAPSS FD001.

Default backend: fast scikit-learn tree ensembles so the repo is easy to run on
any laptop. Optional XGBoost support can be enabled with ENABLE_XGBOOST=1.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"
TELEMETRY_DIR = ROOT_DIR / "project2_telemetry"

RANDOM_SEED = 42
RUL_CAP = 125
N_ENSEMBLE_MODELS = 3
ALPHA = 0.10
CALIBRATION_FRACTION = 0.20
DROP_COLS = ["unit_id", "cycle_norm", "RUL", "RUL_raw"]
ENABLE_XGBOOST = os.getenv("ENABLE_XGBOOST", "0") == "1"


@dataclass
class EvaluationResult:
    model: str
    evaluation_scope: str
    rmse: float
    mae: float
    interval_coverage: float | None = None
    mean_interval_width: float | None = None


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def coverage(y_true: pd.Series | np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y = np.asarray(y_true)
    return float(np.mean((y >= lower) & (y <= upper)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]


def split_train_calibration_by_unit(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    units = np.array(sorted(train["unit_id"].unique()))
    rng.shuffle(units)
    n_cal = max(1, int(len(units) * CALIBRATION_FRACTION))
    cal_units = set(units[:n_cal])
    return train[~train["unit_id"].isin(cal_units)].copy(), train[train["unit_id"].isin(cal_units)].copy()


def make_tree_model(seed: int):
    """Strong, fast default model used in the bootstrapped ensemble."""
    return ExtraTreesRegressor(
        n_estimators=20,
        max_depth=12,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
    )


def train_bootstrapped_ensemble(X_train: pd.DataFrame, y_train: pd.Series):
    rng = np.random.default_rng(RANDOM_SEED)
    ensemble = []
    print(f"Training bootstrapped tree ensemble ({N_ENSEMBLE_MODELS} base models)...")
    for i in range(N_ENSEMBLE_MODELS):
        indices = rng.choice(len(X_train), size=len(X_train), replace=True)
        model = make_tree_model(RANDOM_SEED + i)
        model.fit(X_train.iloc[indices], y_train.iloc[indices])
        ensemble.append(model)
        print(f" Model {i + 1}/{N_ENSEMBLE_MODELS} trained successfully.")
    return ensemble


def predict_ensemble_uncalibrated(ensemble, X: pd.DataFrame, alpha: float = ALPHA):
    preds = np.column_stack([model.predict(X) for model in ensemble])
    mean_pred = preds.mean(axis=1)
    lower = np.percentile(preds, alpha / 2 * 100, axis=1)
    upper = np.percentile(preds, (1 - alpha / 2) * 100, axis=1)
    return np.clip(mean_pred, 0, RUL_CAP), np.clip(lower, 0, RUL_CAP), np.clip(upper, 0, RUL_CAP)


def conformal_qhat(y_true: pd.Series, y_pred: np.ndarray, alpha: float = ALPHA) -> float:
    scores = np.abs(np.asarray(y_true) - y_pred)
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration split is empty.")
    quantile_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, quantile_level, method="higher"))


def apply_conformal_interval(mean_pred: np.ndarray, qhat: float):
    return np.clip(mean_pred - qhat, 0, RUL_CAP), np.clip(mean_pred + qhat, 0, RUL_CAP)


def final_cycle_view(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("unit_id")["cycle"].idxmax()
    return df.loc[idx].copy().sort_values("unit_id")


def evaluate_predictions(model_name, scope, y_true, y_pred, lower=None, upper=None) -> EvaluationResult:
    result = EvaluationResult(
        model=model_name,
        evaluation_scope=scope,
        rmse=rmse(y_true, y_pred),
        mae=float(mean_absolute_error(y_true, y_pred)),
    )
    if lower is not None and upper is not None:
        result.interval_coverage = coverage(y_true, lower, upper)
        result.mean_interval_width = interval_width(lower, upper)
    return result


def add_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_level"] = np.select(
        [df["lower_RUL"] < 30, df["lower_RUL"] < 60],
        ["High", "Medium"],
        default="Low",
    )
    df["maintenance_recommendation"] = np.select(
        [df["lower_RUL"] < 30, df["lower_RUL"] < 60],
        [
            "Inspect soon: lower confidence bound is inside the critical maintenance window.",
            "Monitor closely: lower confidence bound is approaching the maintenance window.",
        ],
        default="Normal monitoring: confidence interval remains outside the urgent window.",
    )
    return df


def maybe_train_xgboost_baseline(X_fit, y_fit, X_test, y_test, reports: list[EvaluationResult]) -> None:
    if not ENABLE_XGBOOST:
        print("Skipping optional XGBoost baseline. Set ENABLE_XGBOOST=1 to run it.")
        return
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("xgboost is not installed; skipping optional XGBoost baseline.")
        return
    model = XGBRegressor(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_fit, y_fit)
    pred = np.clip(model.predict(X_test), 0, RUL_CAP)
    reports.append(evaluate_predictions("Optional XGBoost baseline", "timeline_all_cycles", y_test, pred))


def save_results(test, mean_pred, calibrated_lower, calibrated_upper, uncal_lower, uncal_upper, reports, feature_importance):
    REPORTS_DIR.mkdir(exist_ok=True)
    TELEMETRY_DIR.mkdir(exist_ok=True)

    results = test[["unit_id", "cycle", "RUL", "RUL_raw"]].copy()
    results["predicted_RUL"] = mean_pred
    results["lower_RUL"] = calibrated_lower
    results["upper_RUL"] = calibrated_upper
    results["uncalibrated_lower_RUL"] = uncal_lower
    results["uncalibrated_upper_RUL"] = uncal_upper
    results["absolute_error"] = np.abs(results["RUL"] - results["predicted_RUL"])
    results["covered_by_interval"] = (results["RUL"] >= results["lower_RUL"]) & (results["RUL"] <= results["upper_RUL"])
    results = add_risk_labels(results)

    results.to_csv(TELEMETRY_DIR / "model_predictions.csv", index=False)
    results.to_csv(REPORTS_DIR / "timeline_predictions.csv", index=False)
    final_cycle_view(results).to_csv(REPORTS_DIR / "final_cycle_predictions.csv", index=False)

    report_df = pd.DataFrame([r.__dict__ for r in reports])
    report_df["interval_coverage_pct"] = report_df["interval_coverage"].mul(100)
    report_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    report_df.to_csv(TELEMETRY_DIR / "model_metrics.csv", index=False)

    feature_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    feature_importance.head(20).to_csv(TELEMETRY_DIR / "feature_importance_top20.csv", index=False)


def print_report(reports: list[EvaluationResult], qhat: float) -> None:
    report_df = pd.DataFrame([r.__dict__ for r in reports])
    printable = report_df.copy()
    printable["rmse"] = printable["rmse"].map(lambda x: f"{x:.2f}")
    printable["mae"] = printable["mae"].map(lambda x: f"{x:.2f}")
    printable["interval_coverage"] = printable["interval_coverage"].map(lambda x: "N/A" if pd.isna(x) else f"{x * 100:.2f}%")
    printable["mean_interval_width"] = printable["mean_interval_width"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")
    print("\n" + "=" * 72)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 72)
    print(printable.to_string(index=False))
    print(f"\nConformal calibration radius qhat: {qhat:.2f} RUL cycles")
    print("\nPredictions exported to project2_telemetry/model_predictions.csv")


def main() -> None:
    train_path = PROCESSED_DIR / "train_processed.parquet"
    test_path = PROCESSED_DIR / "test_processed.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run project1_ml/preprocess.py before training.")

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    features = feature_columns(train)
    train_fit, calibration = split_train_calibration_by_unit(train)

    X_fit, y_fit = train_fit[features], train_fit["RUL"]
    X_cal, y_cal = calibration[features], calibration["RUL"]
    X_test, y_test = test[features], test["RUL"]

    reports: list[EvaluationResult] = []

    print("Training baseline models...")
    linear = LinearRegression().fit(X_fit, y_fit)
    linear_pred = np.clip(linear.predict(X_test), 0, RUL_CAP)
    reports.append(evaluate_predictions("Linear Regression baseline", "timeline_all_cycles", y_test, linear_pred))

    rf = RandomForestRegressor(n_estimators=20, max_depth=10, min_samples_leaf=4, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_fit, y_fit)
    rf_pred = np.clip(rf.predict(X_test), 0, RUL_CAP)
    reports.append(evaluate_predictions("Random Forest baseline", "timeline_all_cycles", y_test, rf_pred))

    maybe_train_xgboost_baseline(X_fit, y_fit, X_test, y_test, reports)

    ensemble = train_bootstrapped_ensemble(X_fit, y_fit)
    cal_mean, _, _ = predict_ensemble_uncalibrated(ensemble, X_cal)
    qhat = conformal_qhat(y_cal, cal_mean)

    mean_preds, uncal_lower, uncal_upper = predict_ensemble_uncalibrated(ensemble, X_test)
    cal_lower, cal_upper = apply_conformal_interval(mean_preds, qhat)

    reports.append(evaluate_predictions("Bootstrapped tree ensemble uncalibrated", "timeline_all_cycles", y_test, mean_preds, uncal_lower, uncal_upper))
    reports.append(evaluate_predictions("Bootstrapped tree ensemble conformal UQ", "timeline_all_cycles", y_test, mean_preds, cal_lower, cal_upper))

    final_test = final_cycle_view(test)
    final_mean, _, _ = predict_ensemble_uncalibrated(ensemble, final_test[features])
    final_lower, final_upper = apply_conformal_interval(final_mean, qhat)
    reports.append(evaluate_predictions("Bootstrapped tree ensemble conformal UQ", "final_cycle_per_engine", final_test["RUL"], final_mean, final_lower, final_upper))

    importances = np.mean([model.feature_importances_ for model in ensemble], axis=0)
    feature_importance = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)

    save_results(test, mean_preds, cal_lower, cal_upper, uncal_lower, uncal_upper, reports, feature_importance)
    print_report(reports, qhat)


if __name__ == "__main__":
    main()
