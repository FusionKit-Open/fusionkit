# eval/labeler.py
import numpy as np
import pandas as pd

def make_lead_labels(time: np.ndarray, event_time: float, lead: float = 0.2):
    """
    Binary label y=1 if t in [event_time - lead, event_time), else 0.
    """
    if np.isnan(event_time):
        return np.zeros_like(time, dtype=int)
    return ((time >= (event_time - lead)) & (time < event_time)).astype(int)

def align_signals(df: pd.DataFrame, keep_cols=None) -> pd.DataFrame:
    """
    Clean + keep selected columns; drop rows with NaNs at the end.
    """
    if keep_cols is None:
        keep_cols = [c for c in df.columns if c != "event_time"]
    out = df[["time"] + keep_cols].copy()
    out = out.dropna().reset_index(drop=True)
    return out
