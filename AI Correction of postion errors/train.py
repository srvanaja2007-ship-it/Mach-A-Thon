"""
Model Training and Cross-Trajectory Benchmarking Script.
Trains both XGBoost and Random Forest on GSDC training data and evaluates on unseen test trips.
"""
import json
import logging
import pandas as pd

from src.config import (
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    FEATURE_COLUMNS,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.data_loader import prepare_and_cache_dataset
from src.feature_engineering import extract_features_dataset
from src.model import GpsErrorCorrectionModel
from src.evaluation import evaluate_positioning_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline(max_trips: int = 30):
    logger.info("=" * 70)
    logger.info("AI-BASED GPS/GNSS POSITION ERROR CORRECTION - TRAINING PIPELINE")
    logger.info("=" * 70)
    
    # 1. Load or prepare train/test data
    if not TRAIN_DATA_FILE.exists() or not TEST_DATA_FILE.exists():
        logger.info("Processed datasets not found. Extracting from raw GSDC data...")
        train_raw, test_raw = prepare_and_cache_dataset(max_trips=max_trips)
    else:
        logger.info(f"Loading cached datasets from {TRAIN_DATA_FILE}...")
        train_raw = pd.read_parquet(TRAIN_DATA_FILE)
        test_raw = pd.read_parquet(TEST_DATA_FILE)
        
    logger.info(f"Train Raw: {len(train_raw):,} rows across {train_raw['tripId'].nunique()} trips.")
    logger.info(f"Test Raw:  {len(test_raw):,} rows across {test_raw['tripId'].nunique()} trips.")
    
    # 2. Causal Feature Engineering
    logger.info("Performing causal feature extraction on training trips...")
    train_feat = extract_features_dataset(train_raw, include_target=True)
    logger.info("Performing causal feature extraction on test trips...")
    test_feat = extract_features_dataset(test_raw, include_target=True)
    
    X_train = train_feat[FEATURE_COLUMNS]
    y_train_N = train_feat["target_err_N"].values
    y_train_E = train_feat["target_err_E"].values
    
    X_test = test_feat[FEATURE_COLUMNS]
    lat_obs_test = test_feat["LatitudeDegrees_obs"].values
    lon_obs_test = test_feat["LongitudeDegrees_obs"].values
    lat_gt_test = test_feat["LatitudeDegrees_gt"].values
    lon_gt_test = test_feat["LongitudeDegrees_gt"].values
    
    # 3. Train Models: XGBoost & Random Forest
    results = {}
    
    for model_name in ["xgboost", "random_forest"]:
        logger.info("-" * 60)
        logger.info(f"Training {model_name.upper()} Model...")
        model = GpsErrorCorrectionModel(model_type=model_name)
        model.fit(X_train, y_train_N, y_train_E)
        
        # Save model
        model.save(save_dir=MODELS_DIR, prefix=model_name)
        
        # Predict & Correct on UNSEEN Test Trips
        logger.info(f"Evaluating {model_name.upper()} on {test_feat['tripId'].nunique()} unseen test trips...")
        lat_corr, lon_corr, pred_N, pred_E = model.correct_coordinates(
            lat_obs_test, lon_obs_test, X_test
        )
        
        eval_result = evaluate_positioning_accuracy(
            lat_obs=lat_obs_test,
            lon_obs=lon_obs_test,
            lat_gt=lat_gt_test,
            lon_gt=lon_gt_test,
            lat_corr=lat_corr,
            lon_corr=lon_corr,
        )
        
        feat_imp = model.get_feature_importance()
        
        results[model_name] = {
            "eval_result": eval_result,
            "feature_importance": feat_imp,
            "lat_corr": lat_corr,
            "lon_corr": lon_corr,
        }
        
        print(f"\n==================== {model_name.upper()} PERFORMANCE ====================")
        print(eval_result["summary_table"].to_string(index=False))
        print(f"Feature Importance (Top 5):")
        print(feat_imp.head(5).to_string(index=False))
        print("====================================================================\n")

    # 4. Save Comprehensive Benchmark Report
    report_data = {
        "dataset": "Google Smartphone Decimeter Challenge (GSDC)",
        "train_trips_count": int(train_raw["tripId"].nunique()),
        "test_trips_count": int(test_raw["tripId"].nunique()),
        "train_samples": int(len(train_feat)),
        "test_samples": int(len(test_feat)),
        "models": {},
    }
    
    for name, res in results.items():
        ev = res["eval_result"]
        report_data["models"][name] = {
            "raw_metrics": ev["raw_metrics"],
            "corr_metrics": ev["corr_metrics"],
            "improvement_rmse_percent": ev["improvement_rmse"],
            "improvement_mean_percent": ev["improvement_mean"],
            "improvement_median_percent": ev["improvement_median"],
            "improvement_p95_percent": ev["improvement_p95"],
        }
        
    report_file = REPORTS_DIR / "evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Saved evaluation report to {report_file}.")
    
    # Save test predictions for visualization & fast app inspection
    test_eval_df = test_feat.copy()
    test_eval_df["lat_corr_xgb"] = results["xgboost"]["lat_corr"]
    test_eval_df["lon_corr_xgb"] = results["xgboost"]["lon_corr"]
    test_eval_df["lat_corr_rf"] = results["random_forest"]["lat_corr"]
    test_eval_df["lon_corr_rf"] = results["random_forest"]["lon_corr"]
    test_eval_df["error_raw_m"] = results["xgboost"]["eval_result"]["raw_errors"]
    test_eval_df["error_xgb_m"] = results["xgboost"]["eval_result"]["corr_errors"]
    test_eval_df["error_rf_m"] = results["random_forest"]["eval_result"]["corr_errors"]
    
    eval_cache_path = REPORTS_DIR / "test_evaluation_results.parquet"
    test_eval_df.to_parquet(eval_cache_path, index=False)
    logger.info(f"Saved test evaluation predictions to {eval_cache_path}.")
    
    return results


if __name__ == "__main__":
    run_training_pipeline()
