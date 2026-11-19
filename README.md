# NASA C-MAPSS Predictive Maintenance Platform

**[Open the interactive fleet monitor](https://prakhar2828.github.io/CMAPSS-Predictive-Maintenance/)**

Select a test engine, replay its observed cycles, and compare predicted remaining life with the NASA test label and calibrated interval. Filter by inspection priority or switch sensors to investigate flagged readings. The demo runs in the browser using a snapshot of the offline pipeline. No account, API key, or installation is required.

An end-to-end predictive maintenance project for turbofan engines using the NASA C-MAPSS FD001 dataset. The system predicts **Remaining Useful Life (RUL)**, quantifies uncertainty around each prediction, flags telemetry anomalies, and exposes the results through a Streamlit operations dashboard.

This project is designed as a portfolio-quality ML systems project, not just a notebook. It separates the heavy model-training workflow from the lightweight operational telemetry/dashboard workflow.

---

## What this project does

Aircraft engines produce streams of sensor data over time. In the real world, maintenance teams want to know when an engine is starting to degrade before it fails. This project simulates that workflow using NASA's C-MAPSS turbofan degradation data.

The project answers four practical questions:

1. **How much useful life does each engine probably have left?**
2. **How uncertain is that prediction?**
3. **Which engines should be prioritized for inspection?**
4. **Are any sensor streams showing unusual operational behavior?**

The output is a dashboard that shows fleet-level risk, engine-level RUL forecasts, calibrated uncertainty bounds, and anomaly markers.

---

## Dataset

This project uses the **FD001** subset of NASA C-MAPSS.

FD001 is a simulated turbofan engine degradation dataset with:

- Multiple engine units.
- Cycle-by-cycle operating data.
- Three operating settings.
- Twenty-one sensor readings.
- One operating condition.
- One fault mode.
- A test truth file containing RUL at the final observed cycle for each test engine.

Each row represents one engine at one operating cycle. The model learns how sensor patterns change as the engine approaches failure.

---

## Key concept: Remaining Useful Life

**RUL** means **Remaining Useful Life**.

If an engine fails at cycle 200, then:

```text
Cycle 1   -> RUL 199
Cycle 50  -> RUL 150
Cycle 175 -> RUL 25
Cycle 200 -> RUL 0
```

The model tries to predict that RUL value from the engine's sensor readings.

This project uses a standard **piecewise RUL cap** of `125`. That means very large RUL values are clipped to 125 because very healthy engines are often hard to distinguish from each other, and the most important maintenance decisions happen closer to failure.

Example:

```text
Raw RUL:    240
Capped RUL: 125
```

The same cap is applied to both training and test labels. This is important because evaluating a model trained on capped targets against uncapped test targets creates misleadingly bad metrics.

---

## Architecture

The project uses a split ML-system architecture:

### Cold Path: `project1_ml/`

The cold path handles heavier offline work:

- Raw data ingestion.
- RUL label creation.
- Feature engineering.
- Model training.
- Baseline evaluation.
- Bootstrapped tree-ensemble training, with optional XGBoost baseline support.
- Conformal uncertainty calibration.
- Exporting predictions and reports.

This is called the cold path because it does not need to run every second. It is the model-building side of the system.

### Hot Path: `project2_telemetry/`

The hot path handles operational monitoring:

- Raw telemetry ingestion.
- Schema validation.
- Lightweight anomaly rules.
- Dashboard-ready telemetry cache.
- Streamlit visualization.

This is called the hot path because it represents what an operations dashboard would consume in a live monitoring setup.

See `assets/architecture.md` for a Mermaid architecture diagram.

---

## Repository structure

```text
CMAPSS-Predictive-Maintenance/
├── README.md
├── requirements.txt
├── Makefile
├── .gitignore
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   └── ...
├── project1_ml/
│   ├── preprocess.py
│   ├── train_uq.py
│   ├── processed/
│   │   ├── train_processed.parquet
│   │   └── test_processed.parquet
│   └── reports/
│       ├── model_comparison.csv
│       ├── timeline_predictions.csv
│       ├── final_cycle_predictions.csv
│       └── feature_importance.csv
├── project2_telemetry/
│   ├── pipeline.py
│   ├── app.py
│   ├── cleaned_telemetry.csv
│   ├── model_predictions.csv
│   ├── model_metrics.csv
│   └── feature_importance_top20.csv
├── tests/
│   └── test_project_outputs.py
├── assets/
│   └── architecture.md
└── .github/workflows/ci.yml
```

---

## Main files

### `project1_ml/preprocess.py`

This script prepares the raw NASA files for machine learning.

It does the following:

- Loads `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`.
- Creates RUL labels for training engines.
- Extrapolates RUL labels across the test trajectories.
- Applies the same 125-cycle RUL cap to train and test data.
- Adds degradation features:
  - rolling means,
  - rolling standard deviations,
  - one-cycle deltas,
  - elapsed cycles, available at prediction time.
- Drops low-information raw sensors for FD001.
- Saves optimized parquet files.

Generated outputs:

```text
project1_ml/processed/train_processed.parquet
project1_ml/processed/test_processed.parquet
```

### `project1_ml/train_uq.py`

This script trains and evaluates the models.

It includes:

- Linear Regression baseline.
- Random Forest baseline.
- Optional XGBoost baseline.
- Bootstrapped tree ensemble.
- Unit-level calibration split to avoid engine-row leakage.
- Conformal uncertainty calibration.
- Timeline evaluation across all cycles.
- Final-cycle evaluation across one latest-cycle row per test engine.
- Feature importance export.
- Dashboard prediction export.

Generated outputs:

```text
project1_ml/reports/model_comparison.csv
project1_ml/reports/timeline_predictions.csv
project1_ml/reports/final_cycle_predictions.csv
project1_ml/reports/feature_importance.csv
project2_telemetry/model_predictions.csv
project2_telemetry/model_metrics.csv
project2_telemetry/feature_importance_top20.csv
```

### `project2_telemetry/pipeline.py`

This script represents the operational telemetry pipeline.

It does the following:

- Loads raw test telemetry.
- Validates required schema columns.
- Flags anomalies using sensor z-scores and sudden-shift rules.
- Adds an anomaly score, anomaly reason, and normalized cycle value.
- Saves the dashboard-ready telemetry file.

Generated output:

```text
project2_telemetry/cleaned_telemetry.csv
```

### `project2_telemetry/app.py`

This is the Streamlit dashboard.

It shows:

- Fleet risk overview.
- Engine-level risk status.
- Predicted RUL.
- Lower and upper calibrated uncertainty bounds.
- Maintenance recommendation text.
- RUL chart with uncertainty envelope.
- Sensor telemetry chart with anomaly markers.
- Model comparison metrics.
- Top feature importances.

---

## Setup

Recommended Python version: **Python 3.11 or 3.12**.

Python 3.14 may work locally, but Python 3.11/3.12 is safer for sharing, GitHub CI, and deployment because ML packages usually support those versions earlier and more consistently.

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

On macOS Apple Silicon, if optional XGBoost complains about OpenMP, install `libomp`:

```bash
brew install libomp
```

Optional: run an XGBoost baseline too:

```bash
ENABLE_XGBOOST=1 make train
```

The default model backend is a fast scikit-learn tree ensemble so the project is easier to run on laptops and in GitHub CI.

---

## How to run

### Option 1: Use the Makefile

From the project root:

```bash
make preprocess
make train
make telemetry
make app
```

Or run the full pipeline:

```bash
make run-all
```

### Option 2: Run manually

From the project root:

```bash
cd project1_ml
python preprocess.py
python train_uq.py

cd ../project2_telemetry
python pipeline.py
streamlit run app.py
```

Streamlit should print a local URL similar to:

```text
http://localhost:8501
```

Open that in your browser.

---

## How to test

After generating the outputs, run:

```bash
make test
```

or:

```bash
pytest -q
```

The tests check that:

- Train and test RUL are capped at 125.
- Prediction outputs contain calibrated uncertainty columns.
- Telemetry output contains anomaly columns.
- Metrics include both timeline and final-cycle evaluation.

---

## How to understand the results

### RMSE

RMSE means **Root Mean Squared Error**.

It measures the model's prediction error in RUL cycles. Lower is better. RMSE penalizes large errors more strongly than small errors.

### MAE

MAE means **Mean Absolute Error**.

It is the average absolute prediction error in RUL cycles. If MAE is 12, the model is off by about 12 cycles on average.

### Prediction interval coverage

The model does not only predict one RUL value. It also predicts a range:

```text
Predicted RUL: 80
Lower bound:   62
Upper bound:   98
```

The project uses calibrated 90% prediction intervals. Coverage measures how often the true RUL falls inside those intervals.

A good uncertainty model should have coverage close to its stated level, while still keeping intervals reasonably narrow.

### Mean interval width

This measures how wide the uncertainty bands are on average.

There is a tradeoff:

- Very narrow intervals are more precise but may miss the truth often.
- Very wide intervals cover more truth values but are less useful operationally.

The conformal interval gives a more honest uncertainty band than using raw ensemble spread alone.

### Timeline evaluation

Timeline evaluation scores the model across every visible test cycle. This is useful for dashboard behavior because the dashboard shows a full engine trajectory.

### Final-cycle evaluation

Final-cycle evaluation scores only the latest available cycle for each test engine. This is closer to the standard C-MAPSS prediction task because the test truth file gives RUL at the end of each test trajectory.

Both views are useful, so the project reports both.

---

## Dashboard interpretation

The dashboard is designed like a small maintenance operations hub.

### Fleet Risk Overview

This table sorts engines by risk. The main columns are:

- Engine ID.
- Latest observed cycle.
- Predicted RUL.
- Lower confidence bound.
- Upper confidence bound.
- Risk level.
- Anomaly count.

The lower confidence bound is important. If the model predicts 50 cycles left but the lower bound is 20, that engine may need attention soon.

### Risk levels

Risk levels are based on the lower RUL bound:

```text
High risk   -> lower_RUL < 30
Medium risk -> lower_RUL < 60
Low risk    -> lower_RUL >= 60
```

This is intentionally conservative. Maintenance teams usually care about downside risk, not just average predictions.

### RUL uncertainty chart

The RUL chart shows:

- Actual capped RUL.
- Predicted RUL.
- Calibrated lower and upper uncertainty bounds.
- Anomaly markers where applicable.

The shaded uncertainty band is the model saying:

> I think the engine is around this RUL value, but here is the realistic range I want you to consider.

### Sensor telemetry chart

The sensor chart shows one selected sensor over time and marks anomaly rows. This helps connect model-level predictions back to raw telemetry behavior.

---

## What changed from the original version

This updated version improves the project in several important ways:

1. Fixed the test RUL capping bug.
2. Added stronger degradation features.
3. Added Linear Regression, Random Forest, and single optional XGBoost baselines.
4. Added unit-level calibration splitting to avoid leakage.
5. Added conformal calibrated uncertainty intervals.
6. Added final-cycle benchmark evaluation.
7. Added fleet risk levels and maintenance recommendations.
8. Upgraded the Streamlit dashboard with Plotly charts.
9. Added feature importance outputs.
10. Added tests, `requirements.txt`, `.gitignore`, `Makefile`, and GitHub Actions CI.

---

## Limitations

This is still a simulated dataset project, not a live aircraft monitoring system.

Important limitations:

- FD001 has one operating condition and one fault mode.
- The anomaly engine is rule-based and intentionally lightweight.
- The dashboard simulates an operational workflow from local files.
- The model does not yet support FD002, FD003, and FD004 through a command-line option.
- The uncertainty intervals are calibrated for this specific split and dataset setup.

---

## Suggested next improvements

Good future upgrades:

1. Add support for FD002, FD003, and FD004.
2. Add LSTM/GRU/Temporal CNN baselines.
3. Add SHAP explanations for model interpretability.
4. Add critical-window metrics for engines with RUL below 30 or 50.
5. Add Docker support.
6. Add a model registry folder or MLflow experiment tracking.

---

## Portfolio summary

This project demonstrates:

- Machine learning for predictive maintenance.
- Time-series feature engineering.
- RUL target construction.
- Tree-ensemble modeling, with optional XGBoost support.
- Bootstrapped ensembles.
- Conformal uncertainty quantification.
- Model evaluation and baseline comparison.
- Lightweight telemetry data engineering.
- Streamlit dashboarding.
- Testing and repo hygiene.

## Refresh the public demo

After installing the project requirements, run `make demo` to rebuild the FD001 predictions, telemetry and `docs/fleet.json`. Commit the refreshed snapshot. GitHub Pages serves the `docs/` directory on `main` using the free public-repository hosting option. To preview locally, run `python -m http.server --directory docs 8000` and visit `http://localhost:8000`. The Streamlit interface remains available with `make app`.
