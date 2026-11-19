"""Export the browser replay from the same prediction and telemetry files as Streamlit."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project2_telemetry"


def main():
    predictions = pd.read_csv(SOURCE / "model_predictions.csv")
    telemetry = pd.read_csv(SOURCE / "cleaned_telemetry.csv")
    keys = ["unit_id", "cycle"]
    merged = predictions.merge(telemetry, on=keys, validate="one_to_one", how="outer", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Prediction and telemetry cycles differ")
    metrics = pd.read_csv(SOURCE / "model_metrics.csv")
    final = metrics.loc[metrics.evaluation_scope == "final_cycle_per_engine"].iloc[0]
    columns = ["cycle", "predicted_RUL", "lower_RUL", "upper_RUL", "RUL", "risk_level",
               "anomaly_reason", "s2", "s4", "s11"]
    engines = {str(unit): group.sort_values("cycle")[columns].round(4).values.tolist()
               for unit, group in merged.groupby("unit_id")}
    payload = {"columns": columns, "engines": engines, "mae": final.mae,
               "coverage": final.interval_coverage_pct}
    (ROOT / "docs/fleet.json").write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
