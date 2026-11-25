# LEO Satellite Onboard AI Processing & Intelligent Data Transmission

This repository extends the existing **Eco_Act** project to simulate a Low‑Earth‑Orbit (LEO) satellite that processes VIIRS nighttime imagery onboard and transmits only compact economic insights.

## Project Structure
```
LEO_Sim/
├─ eco_act_adapter/      # Thin wrappers around Eco_Act modules
├─ onboard_processor/   # Runs Eco_Act analysis on a given image
├─ image_acquisition/   # Streamlit UI for image upload / selection
├─ orbit/               # Simple Keplerian orbit utilities
├─ satellite_simulation/ # Earth visualisation & satellite animation
├─ insight_extraction/  # Convert raw metrics into JSON payload
├─ payload_generator/   # Serialise payload & measure size
├─ communication_simulator/ # Transmission time & progress simulation
├─ ground_station/      # Dashboard for received payload & stats
├─ dashboard/           # Streamlit multi‑page app entry point
├─ reports/             # Auto‑generated comparison charts
├─ tests/               # Unit / integration tests
└─ requirements.txt
```

## Setup
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

## Run the demo
```bash
streamlit run dashboard/app.py
```

## Assumptions
* LEO altitude: **500 km** (≈ 92 min orbital period).
* Communication bandwidth: **2 Mbps** (default, configurable in UI).
* Ground station location: **India (lat 20.59 N, lon 78.96 E)**.

---
*The original Eco_Act code remains untouched; all new functionality lives under `LEO_Sim/`.*