# eval/metrics.py
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, brier_score_loss

def evaluate(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Return a small dict of useful metrics for imbalanced, time-critical prediction.
    """
    ap = average_precision_score(y_true, y_prob)
    bs = brier_score_loss(y_true, y_prob)
    # lead-time proxy: time between first threshold crossing and first positive label
    return {"AP": ap, "brier": bs}

def threshold_at_precision(y_true: np.ndarray, y_prob: np.ndarray, target_p=0.8):
    p, r, th = precision_recall_curve(y_true, y_prob)
    # choose the highest threshold that achieves >= target precision
    mask = p[:-1] >= target_p
    if not mask.any():
        return 0.5
    return float(th[mask][-1])
