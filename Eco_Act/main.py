import os
import sys
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import data_loader
import feature_extractor
import ai_predictor
import visualizer


def _find_optical_image(dataset_dir, output_dir, region_slug):
    """Looks for an optical reference image in dataset_dir or output_dir."""
    candidates = [
        os.path.join(dataset_dir, f"{region_slug}_optical_landmarks.png"),
        os.path.join(dataset_dir, "optical.png"),
        os.path.join(output_dir, f"{region_slug}_optical_landmarks.png"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    if "tumkur" in region_slug:
        import download_online_dataset
        opt_landmark_path = os.path.join(output_dir, "tumkur_optical_landmarks.png")
        if not os.path.exists(opt_landmark_path):
            download_online_dataset.fetch_optical_satellite_with_landmarks(output_dir)
        return opt_landmark_path
    return None


def run_pipeline(dataset_dir=None, output_dir=None, future_horizon=3, region_name=None, base_gddp=32000.0, target_size=None):
    """
    Runs the Spatio-Temporal Satellite Economic AI pipeline on any night-light dataset.
    Automatically adapts to the dataset folder, filenames, resolution, and time span.
    """
    if dataset_dir is None:
        dataset_dir = os.path.join(CURRENT_DIR, "Dataset")
    if output_dir is None:
        output_dir = os.path.join(CURRENT_DIR, "outputs")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Ingest Dataset & Auto-detect Region Name, Resolution, and Landmarks
    print("\n[Step 1/5] Ingesting Satellite Night-Light Dataset...")
    data = data_loader.load_dataset(dataset_dir, target_size=target_size, region_name=region_name)
    region_display = data['region_name']
    region_slug = data['region_slug']
    res_h, res_w = data['resolution']
    landmark_info = data['landmarks']
    images_hist = data['images']
    timestamps_hist = data['timestamps']
    years_hist = data['years']
    months_hist = data['months']

    print("=" * 75)
    print(f" ORBIT TO ECONOMY: {region_display.upper()} SPATIO-TEMPORAL AI PLATFORM")
    print(f" Resolution: {res_w}x{res_h} | Frames: {len(images_hist)} ({years_hist[0]}-{months_hist[0]:02d} to {years_hist[-1]}-{months_hist[-1]:02d})")
    print(" Track A: Read the Planet (Boeing Aeromania '26 Hackathon)")
    print("=" * 75)

    opt_landmark_path = _find_optical_image(dataset_dir, output_dir, region_slug)

    # 2. Computer Vision & Multi-Scale Feature Extraction
    print("\n[Step 2/5] Running Multi-Scale Vision Feature Extraction...")
    hist_df = feature_extractor.extract_geospatial_features(
        images_hist, timestamps_hist, years_hist, months_hist, landmark_info=landmark_info
    )
    base_sol = hist_df['sol_trend_smooth'].iloc[0]
    recent_sol = hist_df['sol_trend_smooth'].iloc[-1]
    pct_exp = ((recent_sol - base_sol) / max(1.0, base_sol)) * 100.0
    print(f" -> Baseline SOL = {base_sol:,.0f} | Recent SOL = {recent_sol:,.0f} ({pct_exp:+.1f}%)")

    # 3. Spatio-Temporal AI Forecast
    last_year = int(years_hist[-1])
    future_years = [last_year + 1 + i for i in range(future_horizon)]
    print(f"\n[Step 3/5] Training Spatio-Temporal Model & Nowcasting GDDP to {future_years[-1]}...")
    ai = ai_predictor.SpatioTemporalDevelopmentAI(base_gddp_crores=base_gddp, elasticity_alpha=0.75)
    ai.fit_macro_trend(hist_df)
    pred_df = ai.predict_macro_future(future_years)

    for _, row in pred_df.iterrows():
        print(f"    Year {int(row['year'])}: GDDP = ₹{row['predicted_gddp_crores']:,.0f} Cr (Range: ₹{row['confidence_interval_low_gddp']:,.0f} - ₹{row['confidence_interval_high_gddp']:,.0f} Cr) | Econ Index = {row['economic_activity_index']:.1f}/100")

    # 4. Synthesize Future 2D Satellite Imagery & Discover Opportunity Zones
    print(f"\n[Step 4/5] Synthesizing Future 2D Night Light Rasters ({res_w}x{res_h})...")
    future_maps = ai.predict_future_raster_maps(images_hist, timestamps_hist, future_years)
    zones = ai_predictor.analyze_opportunity_zones(images_hist[-1], future_maps[-1][1], landmark_info=landmark_info, w=res_w, h=res_h)
    print("\n -> Top Emerging Opportunity Hubs:")
    for z in zones[:4]:
        print(f"    * {z['name']} ({z['type'].upper()}): {z['growth_pct']:+.1f}% Growth ({z['status']})")

    # 5. Render Deliverables, Interactive Dashboard & Tabular Predictions
    print("\n[Step 5/5] Generating Plots, Interactive Dashboard & Tabular Predictions...")
    opt_vs_ntl_path = os.path.join(output_dir, f"{region_slug}_optical_vs_nightlights.png")
    if opt_landmark_path:
        visualizer.render_optical_vs_nightlights_comparison(
            opt_landmark_path, images_hist[-1], opt_vs_ntl_path, region_name=region_display, landmark_info=landmark_info
        )

    timeline_path = os.path.join(output_dir, f"{region_slug}_timeline_progression.png")
    visualizer.render_plot_timeline(
        images_hist, years_hist, future_maps, future_years, timeline_path, region_name=region_display, landmark_info=landmark_info
    )

    metrics_path = os.path.join(output_dir, f"{region_slug}_metrics_forecast.png")
    visualizer.render_plot_metrics(hist_df, pred_df, zones, metrics_path, region_name=region_display, base_gddp=base_gddp)

    dashboard_path = os.path.join(output_dir, f"{region_slug}_ai_dashboard.html")
    visualizer.generate_interactive_html_dashboard(
        hist_df, pred_df, images_hist, timestamps_hist, future_maps, future_years, zones, opt_landmark_path, dashboard_path,
        region_name=region_display, base_gddp=base_gddp
    )

    table_path = os.path.join(output_dir, f"{region_slug}_predictions_table.html")
    visualizer.generate_predictions_table_html(hist_df, pred_df, zones, table_path, region_name=region_display)

    hist_csv = os.path.join(output_dir, f"{region_slug}_historical_features.csv")
    pred_csv = os.path.join(output_dir, f"{region_slug}_future_predictions.csv")
    hist_df.to_csv(hist_csv, index=False)
    pred_df.to_csv(pred_csv, index=False)

    print("\n" + "=" * 75)
    print(f" ALL DELIVERABLES GENERATED FOR '{region_display.upper()}'!")
    if opt_landmark_path:
        print(f" 1. Optical vs Nightlights : {opt_vs_ntl_path}")
    print(f" 2. Timeline Progression   : {timeline_path}")
    print(f" 3. Economic Forecast Plot : {metrics_path}")
    print(f" 4. Interactive Dashboard  : {dashboard_path}")
    print(f" 5. Predictions & Table    : {table_path}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Satellite Night-Light Economic AI")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset folder (auto-detects name & resolution)")
    parser.add_argument("--output", type=str, default=None, help="Path to output folder")
    parser.add_argument("--name", type=str, default=None, help="Optional override for region/city name")
    parser.add_argument("--horizon", type=int, default=3, help="Years into future to predict")
    parser.add_argument("--base-gddp", type=float, default=32000.0, help="Baseline economic output (₹ Crores)")
    parser.add_argument("--size", type=int, default=None, help="Optional square resize dimension (e.g. 512 or 1024)")
    args = parser.parse_args()

    t_size = (args.size, args.size) if args.size else None
    run_pipeline(
        dataset_dir=args.dataset,
        output_dir=args.output,
        future_horizon=args.horizon,
        region_name=args.name,
        base_gddp=args.base_gddp,
        target_size=t_size
    )
