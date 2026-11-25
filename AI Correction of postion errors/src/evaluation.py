"""
Evaluation metrics module for GPS/GNSS positioning error correction.
Computes WGS84 ellipsoidal geodesic distance errors, MAE, RMSE, median,
95th percentile error, percentage improvements, and head-to-head model comparison.
"""
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from src.config import EARTH_RADIUS_METERS, WGS84_A, WGS84_B


def haversine_distance_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Computes great-circle distance between two pairs of points on Earth in meters using Haversine formula.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def geodesic_distance_wgs84_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Computes geodesic surface distance in metres on the WGS84 ellipsoid using
    Vincenty's inverse formula, with Haversine fallback for non-convergent pairs.
    """
    lat1_arr = np.asarray(lat1, dtype=np.float64)
    lon1_arr = np.asarray(lon1, dtype=np.float64)
    lat2_arr = np.asarray(lat2, dtype=np.float64)
    lon2_arr = np.asarray(lon2, dtype=np.float64)

    f = (WGS84_A - WGS84_B) / WGS84_A
    phi1 = np.radians(lat1_arr)
    phi2 = np.radians(lat2_arr)
    L = np.radians(lon2_arr - lon1_arr)

    U1 = np.arctan((1.0 - f) * np.tan(phi1))
    U2 = np.arctan((1.0 - f) * np.tan(phi2))
    sinU1, cosU1 = np.sin(U1), np.cos(U1)
    sinU2, cosU2 = np.sin(U2), np.cos(U2)

    same_point = (lat1_arr == lat2_arr) & (lon1_arr == lon2_arr)
    lam = L.copy()
    sin_sigma = np.zeros_like(lam)
    cos_sigma = np.ones_like(lam)
    sigma = np.zeros_like(lam)
    cos_sq_alpha = np.ones_like(lam)
    cos_2sigma_m = np.zeros_like(lam)

    active = ~same_point
    for _ in range(max_iter):
        if not np.any(active):
            break
        sin_lam = np.sin(lam[active])
        cos_lam = np.cos(lam[active])

        s1 = cosU2[active] * sin_lam
        s2 = cosU1[active] * sinU2[active] - sinU1[active] * cosU2[active] * cos_lam
        ss = np.sqrt(s1 * s1 + s2 * s2)
        cs = sinU1[active] * sinU2[active] + cosU1[active] * cosU2[active] * cos_lam
        sig = np.arctan2(ss, cs)

        with np.errstate(divide="ignore", invalid="ignore"):
            sin_alpha = np.where(ss > 0, cosU1[active] * cosU2[active] * sin_lam / ss, 0.0)
        csq_alpha = 1.0 - sin_alpha * sin_alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            c2sm = np.where(csq_alpha > 0, cs - 2.0 * sinU1[active] * sinU2[active] / csq_alpha, 0.0)

        C = (f / 16.0) * csq_alpha * (4.0 + f * (4.0 - 3.0 * csq_alpha))
        lam_prev = lam[active]
        lam_new = L[active] + (1.0 - C) * f * sin_alpha * (
            sig + C * ss * (c2sm + C * cs * (-1.0 + 2.0 * c2sm * c2sm))
        )

        sin_sigma[active] = ss
        cos_sigma[active] = cs
        sigma[active] = sig
        cos_sq_alpha[active] = csq_alpha
        cos_2sigma_m[active] = c2sm

        converged = np.abs(lam_new - lam_prev) <= tol
        active_indices = np.where(active)[0]
        lam[active_indices] = lam_new
        active[active_indices[converged]] = False

    u_sq = cos_sq_alpha * (WGS84_A ** 2 - WGS84_B ** 2) / (WGS84_B ** 2)
    A = 1.0 + (u_sq / 16384.0) * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)))
    B = (u_sq / 1024.0) * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)))
    delta_sigma = B * sin_sigma * (
        cos_2sigma_m
        + (B / 4.0)
        * (
            cos_sigma * (-1.0 + 2.0 * cos_2sigma_m ** 2)
            - (B / 6.0) * cos_2sigma_m * (-3.0 + 4.0 * sin_sigma ** 2) * (-3.0 + 4.0 * cos_2sigma_m ** 2)
        )
    )
    distances = WGS84_B * A * (sigma - delta_sigma)
    distances[same_point] = 0.0

    # Fallback to Haversine for any non-converged or NaN entries
    invalid = active | np.isnan(distances)
    if np.any(invalid):
        distances[invalid] = haversine_distance_vectorized(
            lat1_arr[invalid], lon1_arr[invalid], lat2_arr[invalid], lon2_arr[invalid]
        )

    return distances


def compute_metrics(errors_meters: np.ndarray) -> Dict[str, float]:
    """
    Computes key statistical metrics for an array of metric distance errors.
    """
    rmse = float(np.sqrt(np.mean(errors_meters ** 2)))
    mae = float(np.mean(errors_meters))
    median = float(np.median(errors_meters))
    p75 = float(np.percentile(errors_meters, 75))
    p95 = float(np.percentile(errors_meters, 95))
    max_err = float(np.max(errors_meters))
    min_err = float(np.min(errors_meters))
    std_err = float(np.std(errors_meters))

    return {
        "mae": mae,
        "rmse": rmse,
        "mean_error": mae,
        "median_error": median,
        "p75_error": p75,
        "p95_error": p95,
        "max_error": max_err,
        "min_error": min_err,
        "std_error": std_err,
    }


def evaluate_positioning_accuracy(
    lat_obs: np.ndarray,
    lon_obs: np.ndarray,
    lat_gt: np.ndarray,
    lon_gt: np.ndarray,
    lat_corr: np.ndarray,
    lon_corr: np.ndarray,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation comparing original vs AI-corrected positioning accuracy
    using WGS84 geodesic distances.
    """
    raw_errors = geodesic_distance_wgs84_vectorized(lat_obs, lon_obs, lat_gt, lon_gt)
    corr_errors = geodesic_distance_wgs84_vectorized(lat_corr, lon_corr, lat_gt, lon_gt)

    raw_metrics = compute_metrics(raw_errors)
    corr_metrics = compute_metrics(corr_errors)

    improvement_rmse = (raw_metrics["rmse"] - corr_metrics["rmse"]) / raw_metrics["rmse"] * 100.0
    improvement_mean = (raw_metrics["mean_error"] - corr_metrics["mean_error"]) / raw_metrics["mean_error"] * 100.0
    improvement_median = (raw_metrics["median_error"] - corr_metrics["median_error"]) / raw_metrics["median_error"] * 100.0
    improvement_p95 = (raw_metrics["p95_error"] - corr_metrics["p95_error"]) / raw_metrics["p95_error"] * 100.0

    summary_table = pd.DataFrame([
        {
            "Metric": "MAE (m)",
            "Original (Raw)": f"{raw_metrics['mae']:.3f} m",
            "AI-Corrected": f"{corr_metrics['mae']:.3f} m",
            "Absolute Reduction": f"{raw_metrics['mae'] - corr_metrics['mae']:.3f} m",
            "Improvement (%)": f"{improvement_mean:+.2f}%",
        },
        {
            "Metric": "RMSE (m)",
            "Original (Raw)": f"{raw_metrics['rmse']:.3f} m",
            "AI-Corrected": f"{corr_metrics['rmse']:.3f} m",
            "Absolute Reduction": f"{raw_metrics['rmse'] - corr_metrics['rmse']:.3f} m",
            "Improvement (%)": f"{improvement_rmse:+.2f}%",
        },
        {
            "Metric": "Median Error (m)",
            "Original (Raw)": f"{raw_metrics['median_error']:.3f} m",
            "AI-Corrected": f"{corr_metrics['median_error']:.3f} m",
            "Absolute Reduction": f"{raw_metrics['median_error'] - corr_metrics['median_error']:.3f} m",
            "Improvement (%)": f"{improvement_median:+.2f}%",
        },
        {
            "Metric": "95th Percentile (m)",
            "Original (Raw)": f"{raw_metrics['p95_error']:.3f} m",
            "AI-Corrected": f"{corr_metrics['p95_error']:.3f} m",
            "Absolute Reduction": f"{raw_metrics['p95_error'] - corr_metrics['p95_error']:.3f} m",
            "Improvement (%)": f"{improvement_p95:+.2f}%",
        },
    ])

    return {
        "raw_metrics": raw_metrics,
        "corr_metrics": corr_metrics,
        "improvement_rmse": improvement_rmse,
        "improvement_mean": improvement_mean,
        "improvement_median": improvement_median,
        "improvement_p95": improvement_p95,
        "summary_table": summary_table,
        "raw_errors": raw_errors,
        "corr_errors": corr_errors,
    }


def check_data_leakage(
    feature_columns: List[str],
    test_df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Audits model features and train/test splits for data leakage.
    """
    suspicious_tokens = ("gt", "true", "target", "ground_truth", "reference")
    leaked_features = [
        col for col in feature_columns
        if any(tok in col.lower() for tok in suspicious_tokens)
    ]

    overlapping_trips: List[str] = []
    if train_df is not None and "tripId" in train_df.columns and "tripId" in test_df.columns:
        overlapping_trips = sorted(
            set(train_df["tripId"].unique()).intersection(set(test_df["tripId"].unique()))
        )

    passed = (len(leaked_features) == 0) and (len(overlapping_trips) == 0)
    return {
        "passed": passed,
        "leaked_features": leaked_features,
        "overlapping_trips": overlapping_trips,
        "feature_count": len(feature_columns),
    }


def compare_models_on_test_data(
    test_feat_df: pd.DataFrame,
    rf_model: Any,
    xgb_model: Any,
    feature_columns: List[str],
    max_correction_meters: Optional[float] = 5.0,
    train_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Evaluates Raw GPS, Random Forest, and XGBoost on the exact same unseen test dataset/trajectory.
    """
    required_cols = [
        "LatitudeDegrees_obs",
        "LongitudeDegrees_obs",
        "LatitudeDegrees_gt",
        "LongitudeDegrees_gt",
    ] + list(feature_columns)
    missing_cols = [c for c in required_cols if c not in test_feat_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for evaluation: {missing_cols}")

    if test_feat_df[required_cols].isna().any().any():
        nan_cols = test_feat_df[required_cols].columns[test_feat_df[required_cols].isna().any()].tolist()
        raise ValueError(f"NaN values found in required columns: {nan_cols}")

    # Both regressors must receive the same feature vector in the same order.
    # Ground-truth columns are intentionally excluded from this inference frame.
    rf_features = list(rf_model.feature_columns)
    xgb_features = list(xgb_model.feature_columns)
    requested_features = list(feature_columns)
    if rf_features != requested_features or xgb_features != requested_features:
        raise ValueError(
            "Random Forest and XGBoost do not share the requested inference feature schema. "
            f"RF={rf_features}, XGBoost={xgb_features}, requested={requested_features}"
        )

    leakage_audit = check_data_leakage(
        feature_columns=rf_model.feature_columns,
        test_df=test_feat_df,
        train_df=train_df,
    )
    xgb_leakage = check_data_leakage(
        feature_columns=xgb_model.feature_columns,
        test_df=test_feat_df,
        train_df=train_df,
    )
    leakage_audit["passed"] = leakage_audit["passed"] and xgb_leakage["passed"]
    leakage_audit["leaked_features"] = sorted(
        set(leakage_audit["leaked_features"]) | set(xgb_leakage["leaked_features"])
    )

    lat_obs = test_feat_df["LatitudeDegrees_obs"].to_numpy(dtype=np.float64)
    lon_obs = test_feat_df["LongitudeDegrees_obs"].to_numpy(dtype=np.float64)
    lat_gt = test_feat_df["LatitudeDegrees_gt"].to_numpy(dtype=np.float64)
    lon_gt = test_feat_df["LongitudeDegrees_gt"].to_numpy(dtype=np.float64)
    X_test = test_feat_df.loc[:, feature_columns]

    lat_rf, lon_rf, _, _ = rf_model.correct_coordinates(
        lat_obs, lon_obs, X_test, max_correction_meters=max_correction_meters
    )
    lat_xgb, lon_xgb, _, _ = xgb_model.correct_coordinates(
        lat_obs, lon_obs, X_test, max_correction_meters=max_correction_meters
    )

    raw_errors = geodesic_distance_wgs84_vectorized(lat_obs, lon_obs, lat_gt, lon_gt)
    rf_errors = geodesic_distance_wgs84_vectorized(lat_rf, lon_rf, lat_gt, lon_gt)
    xgb_errors = geodesic_distance_wgs84_vectorized(lat_xgb, lon_xgb, lat_gt, lon_gt)

    raw_metrics = compute_metrics(raw_errors)
    rf_metrics = compute_metrics(rf_errors)
    xgb_metrics = compute_metrics(xgb_errors)

    rf_rmse_imp = (raw_metrics["rmse"] - rf_metrics["rmse"]) / raw_metrics["rmse"] * 100.0
    xgb_rmse_imp = (raw_metrics["rmse"] - xgb_metrics["rmse"]) / raw_metrics["rmse"] * 100.0

    better_model = "Random Forest" if rf_metrics["rmse"] < xgb_metrics["rmse"] else "XGBoost"
    if np.isclose(rf_metrics["rmse"], xgb_metrics["rmse"]):
        better_model = "Tie"

    comparison_df = pd.DataFrame([
        {
            "Model": "Raw GPS (Observed)",
            "MAE (m)": round(raw_metrics["mae"], 4),
            "RMSE (m)": round(raw_metrics["rmse"], 4),
            "Median Error (m)": round(raw_metrics["median_error"], 4),
            "95th Percentile Error (m)": round(raw_metrics["p95_error"], 4),
            "RMSE Improvement (%)": 0.0,
        },
        {
            "Model": "Random Forest",
            "MAE (m)": round(rf_metrics["mae"], 4),
            "RMSE (m)": round(rf_metrics["rmse"], 4),
            "Median Error (m)": round(rf_metrics["median_error"], 4),
            "95th Percentile Error (m)": round(rf_metrics["p95_error"], 4),
            "RMSE Improvement (%)": round(rf_rmse_imp, 2),
        },
        {
            "Model": "XGBoost",
            "MAE (m)": round(xgb_metrics["mae"], 4),
            "RMSE (m)": round(xgb_metrics["rmse"], 4),
            "Median Error (m)": round(xgb_metrics["median_error"], 4),
            "95th Percentile Error (m)": round(xgb_metrics["p95_error"], 4),
            "RMSE Improvement (%)": round(xgb_rmse_imp, 2),
        },
    ])

    # Per-trajectory breakdown
    per_trip_rows = []
    if "tripId" in test_feat_df.columns:
        for trip_id, idx in test_feat_df.groupby("tripId", sort=False).groups.items():
            pos = test_feat_df.index.get_indexer(idx)
            r_m = compute_metrics(raw_errors[pos])
            rf_m = compute_metrics(rf_errors[pos])
            xgb_m = compute_metrics(xgb_errors[pos])
            rf_imp = (r_m["rmse"] - rf_m["rmse"]) / r_m["rmse"] * 100.0
            xgb_imp = (r_m["rmse"] - xgb_m["rmse"]) / r_m["rmse"] * 100.0
            per_trip_rows.append({
                "tripId": trip_id,
                "observations": len(pos),
                "raw_rmse_m": round(r_m["rmse"], 4),
                "rf_rmse_m": round(rf_m["rmse"], 4),
                "xgb_rmse_m": round(xgb_m["rmse"], 4),
                "rf_rmse_imp_pct": round(rf_imp, 2),
                "xgb_rmse_imp_pct": round(xgb_imp, 2),
                "better_model": "Random Forest" if rf_m["rmse"] < xgb_m["rmse"] else "XGBoost",
            })
    per_trip_df = pd.DataFrame(per_trip_rows)

    n_observations = int(len(test_feat_df))
    n_trajectories = int(test_feat_df["tripId"].nunique()) if "tripId" in test_feat_df.columns else 1

    return {
        "evaluation_split": "Held-out GSDC test data",
        "inference_feature_columns": requested_features,
        "n_observations": n_observations,
        "n_trajectories": n_trajectories,
        "raw_metrics": raw_metrics,
        "rf_metrics": rf_metrics,
        "xgb_metrics": xgb_metrics,
        "rf_rmse_improvement_pct": rf_rmse_imp,
        "xgb_rmse_improvement_pct": xgb_rmse_imp,
        "better_model": better_model,
        "comparison_table": comparison_df,
        "per_trip_table": per_trip_df,
        "leakage_audit": leakage_audit,
        "raw_errors": raw_errors,
        "rf_errors": rf_errors,
        "xgb_errors": xgb_errors,
        "lat_rf": lat_rf,
        "lon_rf": lon_rf,
        "lat_xgb": lat_xgb,
        "lon_xgb": lon_xgb,
    }
