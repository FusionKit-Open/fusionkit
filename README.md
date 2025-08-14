# FusionKit (v0.1) — Open Early-Warning & Advisory for Tokamaks

**Goal:** Predict disruptions/ELMs ahead of time from standard diagnostics and suggest **safe, ranked actuator moves** (with uncertainty).  
**Inputs:** timeseries diagnostics + actuator logs. **Outputs:** risk probability within 50–1000 ms and advisory options.

## Modules (Phase 1)
- `adapters/` — IMAS/OMAS, MDSplus, HDF5 loaders
- `features/` — rolling stats, derivatives, spectra, cross-lags
- `models/` — baselines (LogReg/XGBoost), temporal nets (CNN/GRU)
- `advisory/` — safety-guarded ranked suggestions
- `sim/` — hooks for BOUT++/Hermes-3 (edge), DREAM (runaway)
- `eval/` — metrics & replay demo

## Quick Start
```bash
# python 3.11 recommended
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
