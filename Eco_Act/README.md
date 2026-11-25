# Orbit to Economy: Spatio-Temporal AI for Tumkur Economic Intelligence
**Track A: Read the Planet (Satellite VIIRS Night Lights)**  
*Aeromania '26 Mach-A-Thon (in collaboration with Boeing)*

---

## 🛰️ Project Overview
This AI platform analyzes satellite night-light imagery (NASA/NOAA VIIRS) taken at regular intervals across **Tumkur District, Karnataka** (including the Tumkur City Core, Vasanthanarasapura KIAADB Industrial Mega-Park, Antharasanahalli, and the NH-48 Bengaluru-Tumakuru expressway corridor).

By observing radiant flux dynamics, spatial sprawl, and industrial clustering, the system:
1. **Nowcasts Missing Economic Activity:** Estimates Gross District Domestic Product (GDDP proxy in ₹ Crores) and wealth indices where official ground surveys are lagged or missing.
2. **Predicts the Future:** Spatio-temporal AI models project economic growth, industrial corridor expansion vectors, and synthesize **predicted future 2D satellite night-light maps** for upcoming years (e.g. 2025–2028).
3. **Pinpoints Opportunity Zones:** Identifies high-growth emerging hubs for strategic capital deployment.

---

## 🚀 How to Run

1. **Drop your images (Optional):**
   * Place your historical or interval satellite images (`.png`, `.jpg`, `.tif`, `.tiff`, `.bmp`) directly inside `Eco_Act/Dataset/`.
   * Files can be named by year or timestamp (e.g., `tumkur_2018.png`, `2020.tif`, `2024.jpg`).
   * *Note: If `Dataset/` is empty, the system automatically uses the calibrated Tumkur VIIRS reference sequence (2018–2024).*

2. **Execute the Pipeline:**
   ```bash
   python Eco_Act/main.py
   ```
   *Optional parameters:*
   ```bash
   python Eco_Act/main.py --horizon 4     # Predict 4 years into the future
   ```

3. **View the Interactive Web Dashboard:**
   * Double-click `Eco_Act/outputs/tumkur_ai_dashboard.html` to open the interactive dashboard in any web browser.
   * Features a scrubbable time slider, satellite map viewer, GDDP curve, and investment opportunity table.

---

## 📊 Outputs Generated
All results are saved in `Eco_Act/outputs/`:
* `tumkur_ai_dashboard.html`: Self-contained interactive web demo for judges.
* `tumkur_timeline_progression.png`: Side-by-side progression from historical satellite images to future AI predictions.
* `tumkur_metrics_forecast.png`: 4-panel analytical plots (Sum of Lights, Estimated GDDP, Urban Footprint %, Economic Index).
* `tumkur_future_predictions.csv`: Tabular predictions with 95% confidence intervals.
* `tumkur_historical_features.csv`: Extracted spatial & radiant flux time-series metrics.

---

## 💼 Pitch Cheat Sheet for Boeing Judges
* **The Problem:** Sub-national GDP and census data have a 3–5 year lag; rural & informal economies lack official paperwork.
* **The AI Innovation:** Deep spatial feature extraction + Spatio-temporal diffusion predicting both macro metrics (GDDP) and microscopic 2D future raster maps.
* **Who Pays:**
  1. **Banks & Fintechs / NBFCs:** Underwrite loans and deploy ATMs in unbanked, high-growth corridors.
  2. **Infrastructure & Logistics Firms:** Optimize warehouse siting along NH-48 and Vasanthanarasapura.
  3. **Government & Power Grids:** Plan grid capacity and substation placement 3 years in advance.
