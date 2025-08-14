# features/basic.py
import numpy as np
import pandas as pd

def make_features(time: np.ndarray, signal: np.ndarray, win: int = 50) -> pd.DataFrame:
    """Simple causal features for a 1D signal."""
    df = pd.DataFrame({"t": time, "x": signal})
    # first derivative (approx)
    df["dx_dt"] = np.gradient(df["x"], df["t"])
    # rolling stats (causal)
    df["roll_mean"] = df["x"].rolling(win, min_periods=1).mean()
    df["roll_std"]  = df["x"].rolling(win, min_periods=1).std().fillna(0.0)
    # normalized deviation from recent mean
    df["zscore"] = (df["x"] - df["roll_mean"]) / (df["roll_std"] + 1e-6)
    return df
