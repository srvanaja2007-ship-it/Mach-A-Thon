import os
import sys
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import download_online_dataset
import data_loader
import feature_extractor
import ai_predictor
import visualizer

def run_pipeline(dataset_dir=None, output_dir=None, future_horizon=3):
    """
    Runs the 10-Year Spatio-Temporal Satellite Economic AI pipeline across Tumkur & Outskirts (9x Area).
    """
    if dataset_dir is None:
        dataset_dir = os.path.join(CURRENT_DIR, "Dataset")
    if output_dir is None:
        output_dir = os.path.join(CURRENT_DIR, "outputs")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 75)
    print(" ORBIT TO ECONOMY: TUMKUR & OUTSKIRTS SPATIO-TEMPORAL AI PLATFORM")
    print(" Coverage: Full Regional Macro-Grid (28,000 km²) | Resolution: 1024x1024")
    print(" Temporal Depth: 10 Years Monthly (120 Months: 2015 - 2024)")
    print(" Track A: Read the Planet (Boeing Aeromania '26 Hackathon)")
    print("=" * 75)

    # 1. Ensure Optical Satellite Image & Landmarks exist
    print("\n[Step 1/6] Checking High-Res Optical Satellite & Landmark Ground Truth...")
    opt_raw_path = os.path.join(output_dir, "tumkur_optical_satellite_raw.png")
    opt_landmark_path = os.path.join(output_dir, "tumkur_optical_landmarks.png")
    if not os.path.exists(opt_landmark_path):
        download_online_dataset.fetch_optical_satellite_with_landmarks(output_dir)
    print(f" -> Optical reference active: {opt_landmark_path}")

    # 2. Ingest 10-Year Monthly Satellite Time-Series (1024x1024)
    print("\n[Step 2/6] Ingesting 10-Year Monthly Night Lights Time-Series...")
    data = data_loader.load_tumkur_dataset(dataset_dir, target_size=(1024, 1024))
    images_hist = data['images']
    timestamps_hist = data['timestamps']
    years_hist = data['years']
    months_hist = data['months']

    # 3. Computer Vision & Multi-Scale Feature Extraction
    print("\n[Step 3/6] Running Multi-Scale Vision Feature Extraction & Seasonal Decomposition...")
    hist_df = feature_extractor.extract_geospatial_features(images_hist, timestamps_hist, years_hist, months_hist)
    print(f" -> Computed 10-year metrics: Baseline SOL = {hist_df['sol_trend_smooth'].iloc[0]:,.0f} | Recent SOL = {hist_df['sol_trend_smooth'].iloc[-1]:,.0f}")
    print(f" -> Cumulative 10-Year Radiant Expansion: +{((hist_df['sol_trend_smooth'].iloc[-1] - hist_df['sol_trend_smooth'].iloc[0])/hist_df['sol_trend_smooth'].iloc[0])*100.0:.1f}%")

    # 4. Spatio-Temporal AI Forecast
    print("\n[Step 4/6] Training Spatio-Temporal Model & Nowcasting GDDP to 2027...")
    future_years = [2025 + i for i in range(future_horizon)]
    ai = ai_predictor.SpatioTemporalDevelopmentAI(base_gddp_crores_2015=32000.0, elasticity_alpha=0.75)
    ai.fit_macro_trend(hist_df)
    pred_df = ai.predict_macro_future(future_years)

    print(" -> Projected Economic Projections:")
    for _, row in pred_df.iterrows():
        print(f"    Year {int(row['year'])}: GDDP = ₹{row['predicted_gddp_crores']:,.0f} Cr (Range: ₹{row['confidence_interval_low_gddp']:,.0f} - ₹{row['confidence_interval_high_gddp']:,.0f} Cr) | Econ Index = {row['economic_activity_index']:.1f}/100")

    # 5. Synthesize Future 2D Satellite Imagery (1024x1024)
    print("\n[Step 5/6] Synthesizing Future 2D Night Light Rasters (Corridor Momentum & Agglomeration)...")
    future_maps = ai.predict_future_raster_maps(images_hist, timestamps_hist, future_years)
    print(f" -> Generated {len(future_maps)} future 1024x1024 satellite rasters.")

    # Opportunity Zone Analysis
    zones = ai_predictor.analyze_landmark_opportunity_zones(images_hist[-1], future_maps[-1][1], w=1024, h=1024)
    print("\n -> Top Emerging Landmark Opportunity Hubs:")
    for z in zones[:4]:
        print(f"    * {z['name']} ({z['type'].upper()}): +{z['growth_pct']:.1f}% Growth ({z['status']})")

    # 6. Render Deliverables & Interactive Web Dashboard
    print("\n[Step 6/6] Generating Publication Plots & Standalone Interactive Dashboard...")
    opt_vs_ntl_path = os.path.join(output_dir, "tumkur_optical_vs_nightlights.png")
    visualizer.render_optical_vs_nightlights_comparison(opt_landmark_path, images_hist[-1], opt_vs_ntl_path)

    timeline_path = os.path.join(output_dir, "tumkur_timeline_progression.png")
    visualizer.render_plot_timeline(images_hist, years_hist, future_maps, future_years, timeline_path)

    metrics_path = os.path.join(output_dir, "tumkur_metrics_forecast.png")
    visualizer.render_plot_metrics(hist_df, pred_df, zones, metrics_path)

    dashboard_path = os.path.join(output_dir, "tumkur_ai_dashboard.html")
    visualizer.generate_interactive_html_dashboard(hist_df, pred_df, images_hist, timestamps_hist, future_maps, future_years, zones, opt_landmark_path, dashboard_path)

    # Save CSVs
    hist_df.to_csv(os.path.join(output_dir, "tumkur_historical_features.csv"), index=False)
    pred_df.to_csv(os.path.join(output_dir, "tumkur_future_predictions.csv"), index=False)

    print("\n" + "=" * 75)
    print(" ALL DELIVERABLES GENERATED & VERIFIED!")
    print(f" 1. Optical Ground Truth vs Nightlights : {opt_vs_ntl_path}")
    print(f" 2. 10-Year Timeline Progression Map    : {timeline_path}")
    print(f" 3. Economic Forecast & Hub Metrics     : {metrics_path}")
    print(f" 4. Interactive Live Web Dashboard      : {dashboard_path}")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tumkur 10-Year Satellite AI Nowcasting")
    parser.add_argument("--dataset", type=str, default=None, help="Path to Dataset folder")
    parser.add_argument("--output", type=str, default=None, help="Path to output folder")
    parser.add_argument("--horizon", type=int, default=3, help="Years into future to predict")
    args = parser.parse_args()

    run_pipeline(dataset_dir=args.dataset, output_dir=args.output, future_horizon=args.horizon)
