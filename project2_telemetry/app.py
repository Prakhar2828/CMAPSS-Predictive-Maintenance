from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="C-MAPSS Predictive Maintenance Hub", layout="wide")


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    telemetry = pd.read_csv(APP_DIR / "cleaned_telemetry.csv")
    predictions = pd.read_csv(APP_DIR / "model_predictions.csv")

    metrics_path = APP_DIR / "model_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    fi_path = APP_DIR / "feature_importance_top20.csv"
    feature_importance = pd.read_csv(fi_path) if fi_path.exists() else pd.DataFrame()
    return telemetry, predictions, metrics, feature_importance


def latest_fleet_table(telemetry: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    latest_idx = predictions.groupby("unit_id")["cycle"].idxmax()
    latest_preds = predictions.loc[latest_idx].copy()
    anomaly_counts = telemetry.groupby("unit_id")["anomaly_flag"].sum().rename("anomaly_count")
    latest_preds = latest_preds.merge(anomaly_counts, on="unit_id", how="left")
    latest_preds["anomaly_count"] = latest_preds["anomaly_count"].fillna(0).astype(int)
    priority_map = {"High": 0, "Medium": 1, "Low": 2}
    latest_preds["risk_priority"] = latest_preds["risk_level"].map(priority_map).fillna(3)
    return latest_preds.sort_values(["risk_priority", "lower_RUL", "predicted_RUL"])


def risk_badge(risk: str) -> str:
    if risk == "High":
        return "🔴 High risk"
    if risk == "Medium":
        return "🟠 Medium risk"
    return "🟢 Low risk"


def build_rul_figure(unit_preds: pd.DataFrame, unit_anoms: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=unit_preds["cycle"],
            y=unit_preds["upper_RUL"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=unit_preds["cycle"],
            y=unit_preds["lower_RUL"],
            mode="lines",
            fill="tonexty",
            name="Calibrated 90% interval",
            line=dict(width=0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=unit_preds["cycle"],
            y=unit_preds["RUL"],
            mode="lines",
            name="Actual capped RUL",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=unit_preds["cycle"],
            y=unit_preds["predicted_RUL"],
            mode="lines",
            name="Predicted RUL",
        )
    )

    anomaly_cycles = unit_anoms.loc[unit_anoms["anomaly_flag"] == 1, "cycle"]
    if not anomaly_cycles.empty:
        anomaly_points = unit_preds[unit_preds["cycle"].isin(anomaly_cycles)]
        fig.add_trace(
            go.Scatter(
                x=anomaly_points["cycle"],
                y=anomaly_points["predicted_RUL"],
                mode="markers",
                name="Anomaly marker",
                marker=dict(size=8, symbol="x"),
            )
        )

    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_title="Operating cycle",
        yaxis_title="Remaining Useful Life (cycles)",
        hovermode="x unified",
    )
    return fig


def build_sensor_figure(unit_telemetry: pd.DataFrame, sensor: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=unit_telemetry["cycle"], y=unit_telemetry[sensor], mode="lines", name=sensor))
    anomaly_rows = unit_telemetry[unit_telemetry["anomaly_flag"] == 1]
    if not anomaly_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=anomaly_rows["cycle"],
                y=anomaly_rows[sensor],
                mode="markers",
                name="Anomaly",
                marker=dict(size=8, symbol="x"),
                text=anomaly_rows["anomaly_reason"],
            )
        )
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_title="Operating cycle",
        yaxis_title=sensor,
        hovermode="x unified",
    )
    return fig


st.title("🛰️ C-MAPSS Predictive Maintenance Operations Hub")
st.caption(
    "End-to-end NASA FD001 turbofan RUL prediction with calibrated uncertainty, anomaly monitoring, and fleet-level risk triage."
)

try:
    telemetry_df, preds_df, metrics_df, feature_importance_df = load_dashboard_data()

    fleet_df = latest_fleet_table(telemetry_df, preds_df)
    unit_list = fleet_df["unit_id"].astype(int).tolist()

    st.sidebar.header("Fleet Navigation")
    selected_unit = st.sidebar.selectbox("Select engine unit", unit_list)
    selected_sensor = st.sidebar.selectbox("Sensor stream", options=[f"s{i}" for i in range(1, 22)], index=3)

    unit_telemetry = telemetry_df[telemetry_df["unit_id"] == selected_unit]
    unit_preds = preds_df[preds_df["unit_id"] == selected_unit]
    latest = unit_preds.loc[unit_preds["cycle"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current cycle", f"{int(latest['cycle'])}")
    c2.metric("Predicted RUL", f"{latest['predicted_RUL']:.1f}")
    c3.metric("Lower confidence bound", f"{latest['lower_RUL']:.1f}")
    c4.metric("Risk status", risk_badge(str(latest["risk_level"])))

    st.info(str(latest["maintenance_recommendation"]))

    st.subheader("Fleet Risk Overview")
    overview_cols = [
        "unit_id",
        "cycle",
        "predicted_RUL",
        "lower_RUL",
        "upper_RUL",
        "risk_level",
        "anomaly_count",
    ]
    st.dataframe(
        fleet_df[overview_cols].rename(
            columns={
                "unit_id": "Engine",
                "cycle": "Latest cycle",
                "predicted_RUL": "Predicted RUL",
                "lower_RUL": "Lower bound",
                "upper_RUL": "Upper bound",
                "risk_level": "Risk",
                "anomaly_count": "Anomaly rows",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("RUL Forecast with Calibrated Uncertainty")
        st.plotly_chart(build_rul_figure(unit_preds, unit_telemetry), use_container_width=True)
    with chart_col2:
        st.subheader("Sensor Telemetry and Anomaly Markers")
        st.plotly_chart(build_sensor_figure(unit_telemetry, selected_sensor), use_container_width=True)

    st.subheader("Model Evaluation")
    if not metrics_df.empty:
        display_metrics = metrics_df.copy()
        for col in ["rmse", "mae", "mean_interval_width"]:
            if col in display_metrics:
                display_metrics[col] = display_metrics[col].round(2)
        if "interval_coverage_pct" in display_metrics:
            display_metrics["interval_coverage_pct"] = display_metrics["interval_coverage_pct"].round(2)
        st.dataframe(display_metrics, use_container_width=True, hide_index=True)
    else:
        st.warning("No model_metrics.csv found. Run project1_ml/train_uq.py first.")

    if not feature_importance_df.empty:
        st.subheader("Top XGBoost Feature Importances")
        st.bar_chart(feature_importance_df.set_index("feature")["importance"])

    with st.expander("Latest telemetry rows for selected engine"):
        st.dataframe(unit_telemetry.tail(20), use_container_width=True, hide_index=True)

except FileNotFoundError:
    st.error("Dashboard inputs are missing.")
    st.code(
        "cd project1_ml && python preprocess.py && python train_uq.py\n"
        "cd ../project2_telemetry && python pipeline.py && streamlit run app.py",
        language="bash",
    )
