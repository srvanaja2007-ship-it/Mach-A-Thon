# 🛰️ UrbanPulse AI — Orbit-to-Economy Migration & Sprawl Intelligence

**Aeromania '26 Mach-A-Thon** | Track A: Migration & Urbanization | Boeing Judges

---

## What It Does

UrbanPulse AI fuses **4 real satellite data layers** into an executive-grade geospatial intelligence platform that tells Urban Development Authorities **exactly where migrants will settle next year** — and how to lock in transit & power corridors *before* encroachment makes it 3.4× more expensive.

| Data Layer | Source | Signal |
|---|---|---|
| Sentinel-2 L2A (Optical) | Planetary Computer STAC | ΔNDBI concrete growth, NDVI vegetation loss |
| Sentinel-1 GRD (Radar) | Planetary Computer STAC | VV/VH backscatter → building height proxy |
| VIIRS Nighttime Lights | Planetary Computer STAC | Electricity gap = informal settlement detector |
| OpenStreetMap Road Graph | Overpass / osmnx | Road distance + water/lake mask |

---

## Project Structure

```
migration & urbanization/
├── app.py               ← Streamlit Aerospace Command Center dashboard
├── data_pipeline.py     ← Multi-sensor satellite & OSM fetcher (250 m grid)
├── ml_engine.py         ← 4-Zone AI classifier + 2027–28 sprawl forecaster
├── planner_engine.py    ← AI corridor auto-router + ₹ ROI simulator
├── requirements.txt
├── README.md
└── cache/               ← Pre-processed results for 4 Karnataka preset hubs
```

---

## Karnataka Preset Hubs (Pre-Cached, Loads Instantly)

| Hub | Coordinates | Authority |
|---|---|---|
| Bengaluru — Devanahalli (North IT & Airport) | 13.20°N, 77.70°E | BDA |
| Tumakuru / Tumkur — Vasanthanarasapura (NH-48) | 13.34°N, 77.10°E | TUDA |
| Mysuru / Mysore — Hinkal (Ring Road & IT) | 12.31°N, 76.62°E | MUDA |
| Mangaluru / Mangalore — Baikampady (Port Belt) | 12.95°N, 74.82°E | NMMC |

---

## Setup & Run

### 1. Install dependencies
```powershell
cd "migration & urbanization"
pip install -r requirements.txt
```

### 2. Build the preset cache (fetches real satellite data)
```powershell
python data_pipeline.py
```
> This pre-caches all 4 Karnataka hubs. Takes ~5–10 min on first run (live satellite fetch). Subsequent loads are instant from `cache/`.

### 3. Launch the dashboard
```powershell
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## AI Models

### 4-Zone Gradient Boosting Classifier
- Features: ΔNDBI · NDVI · SAR VV/VH · VIIRS DNB · road distance · water mask
- 5-Fold Stratified CV Accuracy printed to console on each run
- Physics-informed labels from Karnataka satellite statistics literature

### 2027–28 Spatial RF Regressor (Migration Gravity Forecaster)
- Trains on 2021→2026 growth drivers (road pull, neighbour density, NLG attractor)
- Outputs per-cell sprawl risk probability (0–1) heatmap

### AI Corridor Auto-Router (NetworkX Dijkstra)
- Routes cheapest bus-road + power-line paths from NLG settlements to OSM roads
- High cost penalties around water bodies and green buffers
- Outputs GeoJSON LineStrings + ₹ Crore cost estimates

---

## Who Pays & Economic Value

Urban Development Authorities (BDA, TUDA, MUDA, NMMC) and Municipal Corporations pay for UrbanPulse AI to:
1. **Identify** Night-Light Gap settlements *before* they become entrenched slums
2. **Lock in** transit & power corridors early at ₹ 3.3 Cr / km
3. **Avoid** demolition + retrofit 3 years later at **3.4× the cost**

The interactive "Act Now vs. Wait 3 Years" slider quantifies **Net Municipal Savings in ₹ Crore and $ Million** in real-time.

---

## Team — Track A (5 Modules)
🛰 UrbanPulse AI (Migration & Urbanization) · 🌊 FloodSense · 🌾 AgroSat · 🏗 InfraScope · 📡 TrafficOrbit

*Aeromania '26 Mach-A-Thon · "Orbit to Economy: AI in Space" · Boeing Division*

