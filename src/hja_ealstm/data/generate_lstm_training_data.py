import pandas as pd
import numpy as np
from pathlib import Path

SEQUENCE_LENGTH = 270

X_FILE = Path("data/processed/lstm_inputs.parquet")
Y_FILE = Path("data/processed/lstm_targets.parquet")

OUTDIR = Path("data/processed")
X_SEQ_OUT = OUTDIR / "x_sequences.npy"
Y_SEQ_OUT = OUTDIR / "y_targets.npy"

def create_lstm_sequences():
    df_x = pd.read_parquet(X_FILE)
    df_y = pd.read_parquet(Y_FILE)

    print(f"Loaded x: {df_x.shape}, y: {df_y.shape}")

    x = df_x.values
    y = df_y.values.squeeze()

    n_samples = x.shape[0] - SEQUENCE_LENGTH
    x_seq = np.zeros((n_samples, SEQUENCE_LENGTH, x.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        x_seq[i] = x[i : i + SEQUENCE_LENGTH]
        y_seq[i] = y[i + SEQUENCE_LENGTH]

    np.save(X_SEQ_OUT, x_seq)
    np.save(Y_SEQ_OUT, y_seq)

    print(f"Saved X sequences to {X_SEQ_OUT} with shape {x_seq.shape}")
    print(f"Saved Y targets to  {Y_SEQ_OUT} with shape {y_seq.shape}")

if __name__ == "__main__":
    create_lstm_sequences()