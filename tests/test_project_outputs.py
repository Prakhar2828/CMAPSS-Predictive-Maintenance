from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_processed_rul_is_capped():
    train_path = ROOT / "project1_ml" / "processed" / "train_processed.parquet"
    test_path = ROOT / "project1_ml" / "processed" / "test_processed.parquet"
    assert train_path.exists(), "Run project1_ml/preprocess.py first."
    assert test_path.exists(), "Run project1_ml/preprocess.py first."

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    assert train["RUL"].max() <= 125
    assert test["RUL"].max() <= 125
    assert train["RUL"].min() >= 0
    assert test["RUL"].min() >= 0


def test_prediction_bridge_has_uncertainty_columns():
    pred_path = ROOT / "project2_telemetry" / "model_predictions.csv"
    assert pred_path.exists(), "Run project1_ml/train_uq.py first."
    preds = pd.read_csv(pred_path)

    required = {
        "unit_id",
        "cycle",
        "RUL",
        "predicted_RUL",
        "lower_RUL",
        "upper_RUL",
        "risk_level",
        "maintenance_recommendation",
    }
    assert required.issubset(preds.columns)
    assert (preds["lower_RUL"] <= preds["predicted_RUL"]).all()
    assert (preds["predicted_RUL"] <= preds["upper_RUL"]).all()


def test_telemetry_pipeline_output_has_anomaly_columns():
    telemetry_path = ROOT / "project2_telemetry" / "cleaned_telemetry.csv"
    assert telemetry_path.exists(), "Run project2_telemetry/pipeline.py first."
    telemetry = pd.read_csv(telemetry_path)
    assert {"anomaly_flag", "anomaly_score", "anomaly_reason", "cycle_norm"}.issubset(telemetry.columns)
    assert set(telemetry["anomaly_flag"].unique()).issubset({0, 1})


def test_metrics_file_contains_final_cycle_evaluation():
    metrics_path = ROOT / "project2_telemetry" / "model_metrics.csv"
    assert metrics_path.exists(), "Run project1_ml/train_uq.py first."
    metrics = pd.read_csv(metrics_path)
    assert "final_cycle_per_engine" in set(metrics["evaluation_scope"])
    assert "timeline_all_cycles" in set(metrics["evaluation_scope"])
