# eval/demo_replay.py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from adapters.h5_adapter import write_demo_h5, load_h5_timeseries
from features.basic import make_features
from models.baseline import synth_series, train_logreg, predict_proba

def main():
    outdir = Path(".data")
    outdir.mkdir(exist_ok=True)

    # 1) quick HDF5 sanity check (creates a tiny demo file and reloads it)
    write_demo_h5(outdir / "demo.h5")
    loaded = load_h5_timeseries(outdir / "demo.h5", ["time", "signal"])
    assert loaded["time"] is not None and loaded["signal"] is not None

    # 2) synthetic 'pre-disruption' training data
    t, x, y, tD = synth_series()
    feats = make_features(t, x)

    # 3) train a tiny baseline
    clf = train_logreg(feats, y)

    # 4) get risk over time
    risk = predict_proba(clf, feats)

    # 5) plot signal (top) and risk (bottom)
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(t, x, lw=1.2)
    ax1.axvline(tD, color="k", ls="--", alpha=0.6, label="disruption")
    ax1.set_ylabel("signal")
    ax1.legend(loc="upper left")

    ax2.plot(t, risk, lw=1.2)
    ax2.axhline(0.5, color="k", ls="--", alpha=0.4)
    ax2.set_ylabel("risk")
    ax2.set_xlabel("time (s)")
    ax2.set_ylim(0, 1)

    out_png = outdir / "out_demo.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved demo plot to {out_png}")

if __name__ == "__main__":
    main()
