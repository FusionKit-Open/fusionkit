# adapters/csv_adapter.py
from pathlib import Path
import pandas as pd

REQUIRED_COLS = ["time"]  # rest are flexible

def load_csv_timeseries(path: str) -> pd.DataFrame:
    """
    Load a CSV with columns:
      time (seconds), and any number of signal columns (e.g., Ip, Dalpha, Mirnov, Prad...).
      Optional column: event_time (single scalar repeated or a single value row) or event flag.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(p)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
    # if there's a single event_time cell elsewhere, propagate it
    if "event_time" in df.columns and df["event_time"].notna().any():
        ev = df["event_time"].dropna().iloc[0]
        df["event_time"] = ev
    return df
