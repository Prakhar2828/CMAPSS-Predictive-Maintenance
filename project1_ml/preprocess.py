"""Preprocess NASA C-MAPSS FD001 data for RUL prediction.

This script turns the raw fixed-width/text FD001 files into model-ready parquet
files. It applies the same piecewise-linear RUL cap to both train and test data
so the model is evaluated against the same target definition it was trained on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = Path(__file__).resolve().parent / "processed"
DATASET_ID = "FD001"
RUL_CAP = 125
ROLLING_WINDOW = 5

COLUMNS = ["unit_id", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]

# Sensors commonly treated as low-variance/non-informative for FD001. We keep
# them in the raw file but drop them from the ML feature space to reduce noise.
LOW_INFORMATION_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]


def load_raw_cmapss_file(data_path: Path) -> pd.DataFrame:
    """Load a raw C-MAPSS file using the standard FD00x column layout."""
    if not data_path.exists():
        raise FileNotFoundError(f"Missing raw data file: {data_path}")
    return pd.read_csv(data_path, sep=r"\s+", header=None, names=COLUMNS)


def add_train_rul(df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """Add piecewise-capped RUL labels for training trajectories."""
    df = df.copy()
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL_raw"] = max_cycles - df["cycle"]
    df["RUL"] = df["RUL_raw"].clip(upper=rul_cap)
    return df


def add_test_rul(df: pd.DataFrame, rul_path: Path, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """Add piecewise-capped RUL labels for test trajectories.

    The NASA test truth file gives the true RUL at the final observed cycle for
    each engine. We extrapolate that value backward over the visible trajectory,
    then apply the same cap used for training. This avoids the common evaluation
    bug where a model trained on capped targets is judged against uncapped test
    labels.
    """
    if not rul_path.exists():
        raise FileNotFoundError(f"Missing test RUL truth file: {rul_path}")

    df = df.copy()
    rul_truths = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["final_cycle_true_rul"])
    rul_truths.index += 1  # unit_id is 1-based in FD001

    max_cycles = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL_raw"] = df.apply(
        lambda row: rul_truths.loc[int(row["unit_id"]), "final_cycle_true_rul"]
        + (max_cycles.loc[row.name] - row["cycle"]),
        axis=1,
    )
    df["RUL"] = df["RUL_raw"].clip(upper=rul_cap)
    return df


def add_degradation_features(df: pd.DataFrame, window_size: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Create trailing sensor statistics and cycle-to-cycle changes per engine."""
    df = df.copy().sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    feature_sensors = [sensor for sensor in SENSOR_COLS if sensor not in LOW_INFORMATION_SENSORS]

    for col in feature_sensors:
        grouped = df.groupby("unit_id")[col]
        df[f"{col}_roll_mean_{window_size}"] = grouped.transform(
            lambda x: x.rolling(window_size, min_periods=1).mean()
        )
        df[f"{col}_roll_std_{window_size}"] = grouped.transform(
            lambda x: x.rolling(window_size, min_periods=1).std().fillna(0)
        )
        df[f"{col}_delta_1"] = grouped.diff().fillna(0)

    return df


def build_processed_dataset(dataset_id: str = DATASET_ID) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build processed train/test parquet files for one C-MAPSS subset."""
    train_raw = load_raw_cmapss_file(DATA_DIR / f"train_{dataset_id}.txt")
    test_raw = load_raw_cmapss_file(DATA_DIR / f"test_{dataset_id}.txt")

    train = add_train_rul(train_raw)
    test = add_test_rul(test_raw, DATA_DIR / f"RUL_{dataset_id}.txt")

    train = add_degradation_features(train)
    test = add_degradation_features(test)

    # Drop low-information raw sensor columns after feature creation. Their
    # rolling features were intentionally never created above.
    train = train.drop(columns=LOW_INFORMATION_SENSORS)
    test = test.drop(columns=LOW_INFORMATION_SENSORS)

    return train, test


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Processing NASA C-MAPSS {DATASET_ID} training and test data...")
    train_df, test_df = build_processed_dataset(DATASET_ID)

    train_path = OUT_DIR / "train_processed.parquet"
    test_path = OUT_DIR / "test_processed.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print("Preprocessing successfully completed!")
    print(f" Train rows: {len(train_df):,} | max capped RUL: {train_df['RUL'].max():.0f}")
    print(f" Test rows:  {len(test_df):,} | max capped RUL: {test_df['RUL'].max():.0f}")
    print(f" Saved: {train_path}")
    print(f" Saved: {test_path}")


if __name__ == "__main__":
    main()
