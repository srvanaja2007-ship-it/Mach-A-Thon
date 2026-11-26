# LEO Satellite Mission Control

Streamlit dashboard for the existing Eco_Act, UrbanPulse, and DisasterRisk
projects. It combines the original application pipelines with an orbital
visibility model, a priority downlink queue, and a ground-station view.

## Run

From the `Mach-A-Thon` directory:

```powershell
..\.venv\Scripts\streamlit.exe run LEO_Sim\app.py
```

The verified environment is Python 3.14 with NumPy 2.5.5. Use the project
virtual environment so UrbanPulse caches written with NumPy 2 remain readable.

## Data and status rules

- Eco_Act runs its existing VIIRS image analysis on files in
  `Eco_Act/Dataset/monthly_10yr`.
- UrbanPulse uses its existing multi-layer pipeline. The onboard adapter
  refuses inference if any required source layer is missing or synthetic.
- DisasterRisk's current preset connector is demo data. The onboard adapter
  reports that blocker and refuses to emit a risk payload.
- The orbit and radio link are simulated. Input sizes and JSON sizes are
  measured from the actual file/serialization when eligible inference runs.

See [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) for the latest verified
measurements and current source blockers.
