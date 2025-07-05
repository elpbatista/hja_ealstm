import pandas as pd
from pathlib import Path

MET_FILE = Path("data/meteorology/openmeteo_hja_daily.parquet")
FLOW_FILE = Path("data/streamflow/HF004_daily_streamflow_entity2.parquet")
OUTDIR = Path("data/processed")
OUTDIR.mkdir(parents=True, exist_ok=True)

X_OUT = OUTDIR / "lstm_inputs.parquet"
Y_OUT = OUTDIR / "lstm_targets.parquet"

def load_and_align():
    # Load meteorological inputs (x[t])
    df_x = pd.read_parquet(MET_FILE)
    df_x = df_x.sort_index()
    print(f"Loaded meteorology: {df_x.shape}")

    # Load streamflow (y[t]) and extract mean_q_area as target
    df_y = pd.read_parquet(FLOW_FILE)
    df_y = df_y.sort_index()
    print("Streamflow columns:", df_y.columns)
    df_y = df_y[["mean_q_area"]].rename(columns={"mean_q_area": "discharge_mm_day"})
    print(f"Loaded streamflow: {df_y.shape}")

    # Inner join to align both time series
    df = df_x.join(df_y, how="inner")
    print(f"Aligned dataset: {df.shape}")

    # Drop any remaining NA rows (if any)
    df = df.dropna()

    # Split into inputs (x) and output (y)
    df_x = df.drop(columns=["discharge_mm_day"])
    df_y = df[["discharge_mm_day"]]

    # Save to disk
    df_x.to_parquet(X_OUT)
    df_y.to_parquet(Y_OUT)
    print(f"Saved inputs to {X_OUT}")
    print(f"Saved targets to {Y_OUT}")

if __name__ == "__main__":
    load_and_align()
