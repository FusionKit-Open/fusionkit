# eval/realdata_demo.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adapters.csv_adapter import load_csv_timeseries
from features.basic import make_features
from models.baseline import train_logreg, predict_proba
from eval.labeler import make_lead_labels, align_signals
from eval.metrics import evaluate, threshold_at_precision

def main(csv_path: str = ".data/shot001.csv", lead: float = 0.2):
    df = load_csv_timeseries(csv_path)
    if "event_time" in df.columns:
        event_time = float(df["event_time"].iloc[0]) if df["event_time"].notna().any() else np.nan
    else:
        event_time = np.nan  # treat as a non-disruptive shot unless provided

    # choose a few signals to start (change these names to match your CSV)
    candidate = [c for c in df.columns if c not in ("time", "event_time")]
    # keep up to 4 signals for a simple baseline
    keep = candidate[:4] if candidate else []
    data = align_signals(df, keep_cols=keep)

    t = data["time"].to_numpy()
    # Simple aggregate: start with one representative signal (or combine later)
    x = data[keep[0]].to_numpy() if keep else data["time"].to_numpy()*0.0

    feats = make_features(t, x, win=50)

    # labels for evaluation
    y = make_lead_labels(t, event_time, lead=lead)

    clf = train_logreg(feats, y)
    risk = predict_proba(clf, feats)

    # metrics
    m = evaluate(y, risk)
    th = threshold_at_precision(y, risk, target_p=0.8)

    # plot
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    ax1.plot(t, x, lw=1.1)
    if not np.isnan(event_time):
        ax1.axvline(event_time, color="k", ls="--", alpha=0.6, label="event")
        ax1.legend(loc="upper left")
    ax1.set_ylabel("signal")

    ax2.plot(t, risk, lw=1.1)
    ax2.axhline(th, color="k", ls="--", alpha=0.4, label=f"thr@P≥0.8")
    ax2.set_ylim(0,1)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("risk")

    out = Path(".data/out_real.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved plot -> {out}")
    print("Metrics:", m)

if __name__ == "__main__":
    main()
