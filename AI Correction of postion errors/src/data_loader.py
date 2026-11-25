"""
Data loader and preprocessing module for Google Smartphone Decimeter Challenge (GSDC) dataset.
Ensures strict train/test isolation by trajectory/trip to prevent data leakage.
"""
import logging
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

from src.config import (
    GT_RAW_FILE,
    OBS_RAW_FILE,
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    TRAIN_DATA_CSV,
    TEST_DATA_CSV,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_gsdc_data(
    gt_file: Path = GT_RAW_FILE,
    obs_file: Path = OBS_RAW_FILE,
    max_trips: Optional[int] = None,
) -> pd.DataFrame:
    """
    Loads raw ground truth and baseline observed coordinates and merges them on tripId & UnixTimeMillis.
    
    Args:
        gt_file: Path to ground_truths_train.csv
        obs_file: Path to baseline_locations_train.csv
        max_trips: Optional limit on the number of trips to load (for faster training)
        
    Returns:
        Merged DataFrame containing observed and ground truth coordinates.
    """
    logger.info(f"Loading Ground Truth from {gt_file}...")
    gt_df = pd.read_csv(gt_file)
    logger.info(f"Loading Baseline Observed Coordinates from {obs_file}...")
    obs_df = pd.read_csv(obs_file)

    # Standardize column naming
    gt_cols = {c: c.strip() for c in gt_df.columns}
    obs_cols = {c: c.strip() for c in obs_df.columns}
    gt_df = gt_df.rename(columns=gt_cols)
    obs_df = obs_df.rename(columns=obs_cols)

    # Merge on tripId and UnixTimeMillis
    logger.info("Merging observed and ground truth datasets on tripId and UnixTimeMillis...")
    merged = pd.merge(
        obs_df,
        gt_df,
        on=["tripId", "UnixTimeMillis"],
        suffixes=("_obs", "_gt"),
    )
    logger.info(f"Merged total records: {len(merged):,} across {merged['tripId'].nunique()} unique trips.")

    # Filter to top trips if max_trips specified
    if max_trips is not None and max_trips > 0:
        top_trips = merged["tripId"].value_counts().index[:max_trips].tolist()
        merged = merged[merged["tripId"].isin(top_trips)].copy()
        logger.info(f"Filtered to top {max_trips} trips ({len(merged):,} records).")

    # Sort strictly by tripId and UnixTimeMillis for causal processing
    merged = merged.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)
    merged["is_simulated"] = False
    return merged


def split_train_test_trips(
    df: pd.DataFrame,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Splits dataset into train and test sets strictly by tripId to prevent any temporal or spatial data leakage.
    
    Args:
        df: Merged dataframe.
        test_ratio: Fraction of trips to hold out for testing (default 0.25).
        seed: Random seed for trip selection reproducibility.
        
    Returns:
        train_df, test_df, train_trips, test_trips
    """
    unique_trips = df["tripId"].unique()
    np.random.seed(seed)
    shuffled_trips = np.random.permutation(unique_trips)
    
    n_test = max(1, int(len(unique_trips) * test_ratio))
    test_trips = list(shuffled_trips[:n_test])
    train_trips = list(shuffled_trips[n_test:])
    
    train_df = df[df["tripId"].isin(train_trips)].copy().reset_index(drop=True)
    test_df = df[df["tripId"].isin(test_trips)].copy().reset_index(drop=True)
    
    logger.info(f"Trip Split: {len(train_trips)} train trips ({len(train_df):,} epochs), "
                f"{len(test_trips)} test trips ({len(test_df):,} epochs).")
    return train_df, test_df, train_trips, test_trips


def prepare_and_cache_dataset(
    max_trips: int = 30,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw data, splits by trip, and caches processed CSV and Parquet files for fast reuse.
    """
    merged = load_raw_gsdc_data(max_trips=max_trips)
    train_df, test_df, train_trips, test_trips = split_train_test_trips(
        merged, test_ratio=test_ratio, seed=seed
    )
    
    # Save to disk
    train_df.to_parquet(TRAIN_DATA_FILE, index=False)
    test_df.to_parquet(TEST_DATA_FILE, index=False)
    train_df.to_csv(TRAIN_DATA_CSV, index=False)
    test_df.to_csv(TEST_DATA_CSV, index=False)
    logger.info(f"Cached train data to {TRAIN_DATA_FILE} and test data to {TEST_DATA_FILE}.")
    return train_df, test_df


def get_available_trips(df: pd.DataFrame) -> List[str]:
    """Returns sorted list of unique trip IDs."""
    return sorted(df["tripId"].unique().tolist())


def get_trip_data(df: pd.DataFrame, trip_id: str) -> pd.DataFrame:
    """Extracts a single trip's data sorted chronologically."""
    trip = df[df["tripId"] == trip_id].sort_values("UnixTimeMillis").reset_index(drop=True)
    return trip


if __name__ == "__main__":
    prepare_and_cache_dataset(max_trips=30)
