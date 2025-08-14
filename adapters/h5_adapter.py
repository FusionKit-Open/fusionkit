# adapters/h5_adapter.py
from pathlib import Path
import h5py
import numpy as np

def load_h5_timeseries(path: str, keys: list[str]) -> dict[str, np.ndarray]:
    """Load 1D timeseries arrays from an HDF5 file."""
    out = {}
    with h5py.File(path, "r") as f:
        for k in keys:
            out[k] = np.array(f[k]) if k in f else None
    return out

def write_demo_h5(path: str, n: int = 2000):
    """Create a tiny demo HDF5 file with time and a sinusoidal signal."""
    t = np.linspace(0, 1, n, dtype=float)
    sig = np.sin(2*np.pi*6*t)
    p = Path(path)
    p.parent.mkdir(exist_ok=True, parents=True)
    with h5py.File(p, "w") as f:
        f.create_dataset("time", data=t)
        f.create_dataset("signal", data=sig)

if __name__ == "__main__":
    demo = ".data/demo.h5"
    write_demo_h5(demo)
    loaded = load_h5_timeseries(demo, ["time", "signal"])
    print("Loaded keys:", [k for k, v in loaded.items() if v is not None])
