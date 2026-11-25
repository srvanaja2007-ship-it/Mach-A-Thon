"""
Head-to-Head GPS Model Accuracy Evaluation Script.
Evaluates existing trained Random Forest and XGBoost models on the same unseen
GSDC test dataset/trajectory using WGS84 geodesic distance, audits for data leakage,
saves CSV results, and generates a short comparison report.
"""
import sys
import logging
from typing import Optional
import pandas as pd

from src.config import (
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    FEATURE_COLUMNS,
    MODELS_DIR,
    REPORTS_DIR,
    MAX_PHYSICAL_CORRECTION_METERS,
)
from src.feature_engineering import extract_features_dataset
from src.model import GpsErrorCorrectionModel
from src.evaluation import compare_models_on_test_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _check_required_artifacts() -> list[str]:
    """Verifies that all required test data and trained model artifacts exist."""
    missing = []
    if not TEST_DATA_FILE.exists():
        missing.append(f"Test dataset missing: {TEST_DATA_FILE}")
    for prefix in ("random_forest", "xgboost"):
        for suffix in ("metadata.json", "model_N.joblib", "model_E.joblib"):
            p = MODELS_DIR / f"{prefix}_{suffix}"
            if not p.exists():
                missing.append(f"Trained model artifact missing: {p}")
    return missing


def run_comparison_evaluation(trip_id: Optional[str] = None):
    """
    Loads existing trained Random Forest and XGBoost models, evaluates both on the
    exact same unseen test trajectory/dataset, saves CSV & Markdown reports.
    """
    missing = _check_required_artifacts()
    if missing:
        logger.error("Cannot run evaluation because required artifacts are missing:")
        for item in missing:
            logger.error(f"  - {item}")
        sys.exit(1)

    test_raw = pd.read_parquet(TEST_DATA_FILE)
    train_raw = pd.read_parquet(TRAIN_DATA_FILE) if TRAIN_DATA_FILE.exists() else None

    if trip_id is not None:
        if trip_id not in test_raw["tripId"].values:
            logger.error(
                f"Trip '{trip_id}' not found in test set. "
                f"Available test trips: {test_raw['tripId'].unique().tolist()}"
            )
            sys.exit(1)
        test_raw = test_raw[test_raw["tripId"] == trip_id].copy()

    # Check required ground-truth and observed columns
    req_raw_cols = [
        "tripId",
        "UnixTimeMillis",
        "LatitudeDegrees_obs",
        "LongitudeDegrees_obs",
        "LatitudeDegrees_gt",
        "LongitudeDegrees_gt",
    ]
    missing_cols = [c for c in req_raw_cols if c not in test_raw.columns]
    if missing_cols:
        logger.error(f"Missing required ground-truth or observed columns in test data: {missing_cols}")
        sys.exit(1)

    # Extract causal features (ground truth is never passed as a feature)
    test_feat = extract_features_dataset(test_raw, include_target=True)

    rf_model = GpsErrorCorrectionModel.load(load_dir=MODELS_DIR, prefix="random_forest")
    xgb_model = GpsErrorCorrectionModel.load(load_dir=MODELS_DIR, prefix="xgboost")

    results = compare_models_on_test_data(
        test_feat_df=test_feat,
        rf_model=rf_model,
        xgb_model=xgb_model,
        feature_columns=FEATURE_COLUMNS,
        max_correction_meters=MAX_PHYSICAL_CORRECTION_METERS,
        train_df=train_raw,
    )

    # Print formatted console summary
    print("\n" + "=" * 78)
    print("REAL-WORLD GPS MODEL ACCURACY EVALUATION (WGS84 GEODESIC DISTANCE)")
    if trip_id:
        print(f"Evaluated Unseen Trajectory : {trip_id}")
    print(f"Number of Test Observations : {results['n_observations']:,}")
    print(f"Number of Test Trajectories : {results['n_trajectories']}")
    print(
        f"Data Leakage Audit          : "
        f"{'PASSED' if results['leakage_audit']['passed'] else 'FAILED'} "
        f"(Overlap trips: {len(results['leakage_audit']['overlapping_trips'])}, "
        f"Leaked features: {len(results['leakage_audit']['leaked_features'])})"
    )
    print("-" * 78)
    print(results["comparison_table"].to_string(index=False))
    print("-" * 78)
    print(f"Raw GPS RMSE                : {results['raw_metrics']['rmse']:.4f} m")
    print(
        f"Random Forest Corrected RMSE: {results['rf_metrics']['rmse']:.4f} m "
        f"({results['rf_rmse_improvement_pct']:+.2f}%)"
    )
    print(
        f"XGBoost Corrected RMSE      : {results['xgb_metrics']['rmse']:.4f} m "
        f"({results['xgb_rmse_improvement_pct']:+.2f}%)"
    )
    print(f"Better Performing Model     : {results['better_model']}")
    print("=" * 78 + "\n")

    # Save CSV results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "model_comparison_results.csv"
    comp_export = results["comparison_table"].copy()
    comp_export.insert(0, "Test Observations", results["n_observations"])
    comp_export.insert(1, "Test Trajectories", results["n_trajectories"])
    comp_export.to_csv(csv_path, index=False)

    per_trip_csv = REPORTS_DIR / "per_trajectory_comparison.csv"
    if not results["per_trip_table"].empty:
        results["per_trip_table"].to_csv(per_trip_csv, index=False)

    # Save short Markdown comparison report
    report_md_path = REPORTS_DIR / "model_comparison_report.md"
    leak = results["leakage_audit"]
    report_lines = [
        "# GPS/GNSS Model Accuracy Evaluation Report",
        "",
        "## Dataset & Leakage Audit",
        f"- **Test GPS Observations**: {results['n_observations']:,}",
        f"- **Test Trajectories**: {results['n_trajectories']}",
        f"- **Distance Metric**: WGS84 Ellipsoidal Geodesic Distance (Vincenty inverse formula, metres)",
        f"- **Data Leakage Check**: {'PASSED' if leak['passed'] else 'FAILED'} "
        f"({len(leak['overlapping_trips'])} overlapping train/test trips, "
        f"{len(leak['leaked_features'])} ground-truth features in model inputs)",
        "",
        "## Model Accuracy Comparison",
        "",
        "| Model | MAE (m) | RMSE (m) | Median Error (m) | 95th Percentile Error (m) | RMSE Improvement (%) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in results["comparison_table"].iterrows():
        report_lines.append(
            f"| {row['Model']} | {row['MAE (m)']:.4f} | {row['RMSE (m)']:.4f} | "
            f"{row['Median Error (m)']:.4f} | {row['95th Percentile Error (m)']:.4f} | "
            f"{row['RMSE Improvement (%)']:+.2f}% |"
        )
    report_lines += [
        "",
        "## Summary",
        f"- **Raw GPS RMSE**: `{results['raw_metrics']['rmse']:.4f} m`",
        f"- **Random Forest Corrected RMSE**: `{results['rf_metrics']['rmse']:.4f} m` (`{results['rf_rmse_improvement_pct']:+.2f}%`)",
        f"- **XGBoost Corrected RMSE**: `{results['xgb_metrics']['rmse']:.4f} m` (`{results['xgb_rmse_improvement_pct']:+.2f}%`)",
        f"- **Better Performing Model**: **{results['better_model']}**",
        "",
    ]
    if not results["per_trip_table"].empty:
        report_lines += [
            "## Per-Trajectory RMSE Breakdown",
            "",
            "| Trajectory (`tripId`) | Obs | Raw RMSE (m) | RF RMSE (m) | XGB RMSE (m) | RF Imp (%) | XGB Imp (%) | Winner |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
        ]
        for _, r in results["per_trip_table"].iterrows():
            report_lines.append(
                f"| `{r['tripId']}` | {r['observations']} | {r['raw_rmse_m']:.4f} | "
                f"{r['rf_rmse_m']:.4f} | {r['xgb_rmse_m']:.4f} | "
                f"{r['rf_rmse_imp_pct']:+.2f}% | {r['xgb_rmse_imp_pct']:+.2f}% | {r['better_model']} |"
            )
        report_lines.append("")

    report_md_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"Saved CSV comparison results to {csv_path}")
    if not results["per_trip_table"].empty:
        logger.info(f"Saved per-trajectory CSV to {per_trip_csv}")
    logger.info(f"Saved comparison report to {report_md_path}")

    return results


if __name__ == "__main__":
    arg_trip = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] not in ("both", "all") else None
    run_comparison_evaluation(trip_id=arg_trip)
