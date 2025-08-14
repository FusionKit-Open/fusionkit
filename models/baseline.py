# models/baseline.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def synth_series(n: int = 2000, t_end: float = 1.0, t_disrupt: float = 0.8):
    """Create a toy 'pre-disruption' pattern: variance & slope grow before t_disrupt."""
    t = np.linspace(0, t_end, n)
    base = 0.2*np.sin(2*np.pi*5*t)
    noise = 0.05*np.random.randn(n)

    # pre-disruption ramp (nonstationary)
    ramp = np.zeros_like(t)
    ramp[t >= (t_disrupt - 0.2)] = np.linspace(0, 0.6, (t >= (t_disrupt - 0.2)).sum())
    x = base + noise + ramp

    # sharp drop at disruption
    x[t >= t_disrupt] -= 1.5

    # labels: 1 if we are within a lead window before disruption, else 0
    lead = 0.15  # 150 ms
    y = (t >= (t_disrupt - lead)) & (t < t_disrupt)
    y = y.astype(int)
    return t, x, y, t_disrupt

def train_logreg(features: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    X = features[["x", "dx_dt", "roll_mean", "roll_std", "zscore"]].values
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    return clf

def predict_proba(clf: LogisticRegression, features: pd.DataFrame) -> np.ndarray:
    X = features[["x", "dx_dt", "roll_mean", "roll_std", "zscore"]].values
    return clf.predict_proba(X)[:, 1]
