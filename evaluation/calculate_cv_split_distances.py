"""Calculate nearest train/validation distances for the stored CV splits.

The script uses the cv_splits.json from a completed CV run and the cube
coordinates from train_test_split.csv. Distances are great-circle distances
between cube centroids, calculated with the Haversine formula.

Example
-------
python -m evaluation.paper_figures.calculate_cv_split_distances \
    --run-dir /net/home/sloeblein/ARCEME-Vegetation-Recovery/model/wand_db_logs/CV_Training_SGEDConvLSTM_RGBI_2026-08-03_20-20-39 \
    --metadata-csv data_processing/data/train_test_split.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.paper_figures.plot_cv_split_maps import (
    load_metadata,
    load_split_membership,
)


EARTH_RADIUS_KM = 6_371.0088


def haversine_distance_km(
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> np.ndarray:
    """Return the pairwise great-circle distance matrix in kilometres."""
    source_latitude = np.radians(source_latitude)[:, None]
    source_longitude = np.radians(source_longitude)[:, None]
    target_latitude = np.radians(target_latitude)[None, :]
    target_longitude = np.radians(target_longitude)[None, :]

    delta_latitude = target_latitude - source_latitude
    delta_longitude = target_longitude - source_longitude
    haversine_term = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(source_latitude)
        * np.cos(target_latitude)
        * np.sin(delta_longitude / 2.0) ** 2
    )
    central_angle = 2.0 * np.arcsin(np.sqrt(np.clip(haversine_term, 0.0, 1.0)))
    return EARTH_RADIUS_KM * central_angle


def calculate_fold_distances(
    fold: int, membership: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    """Find the nearest training cube for every validation cube in one fold."""
    frame = membership[membership["fold"] == fold]
    train = frame[frame["subset"] == "train"].reset_index(drop=True)
    validation = frame[frame["subset"] == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError(
            f"Fold {fold} must contain both training and validation cubes."
        )

    distances = haversine_distance_km(
        validation["latitude"].to_numpy(),
        validation["longitude"].to_numpy(),
        train["latitude"].to_numpy(),
        train["longitude"].to_numpy(),
    )
    nearest_train_indices = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(validation)), nearest_train_indices]

    nearest_rows = pd.DataFrame(
        {
            "fold": fold,
            "validation_cube_id": validation["DisNo."],
            "validation_latitude": validation["latitude"],
            "validation_longitude": validation["longitude"],
            "validation_climate_class": validation["koppen_geiger"],
            "nearest_train_cube_id": train.loc[
                nearest_train_indices, "DisNo."
            ].to_numpy(),
            "nearest_train_latitude": train.loc[
                nearest_train_indices, "latitude"
            ].to_numpy(),
            "nearest_train_longitude": train.loc[
                nearest_train_indices, "longitude"
            ].to_numpy(),
            "nearest_train_climate_class": train.loc[
                nearest_train_indices, "koppen_geiger"
            ].to_numpy(),
            "nearest_train_distance_km": nearest_distances,
        }
    )

    minimum_row = nearest_rows.loc[nearest_rows["nearest_train_distance_km"].idxmin()]
    summary = {
        "fold": fold,
        "n_train_cubes": len(train),
        "n_validation_cubes": len(validation),
        "minimum_train_validation_distance_km": minimum_row[
            "nearest_train_distance_km"
        ],
        "minimum_distance_validation_cube": minimum_row["validation_cube_id"],
        "minimum_distance_train_cube": minimum_row["nearest_train_cube_id"],
        "median_nearest_train_distance_km": nearest_rows[
            "nearest_train_distance_km"
        ].median(),
        "mean_nearest_train_distance_km": nearest_rows[
            "nearest_train_distance_km"
        ].mean(),
        "maximum_nearest_train_distance_km": nearest_rows[
            "nearest_train_distance_km"
        ].max(),
    }
    return summary, nearest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="CV run directory containing cv_splits.json.",
    )
    parser.add_argument(
        "--metadata-csv",
        default="data_processing/data/train_test_split.csv",
        help="Metadata CSV with cube IDs and centroid coordinates.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <run-dir>/results/paper_figures/cv_distances.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    split_path = run_dir / "cv_splits.json"
    if not split_path.exists():
        raise FileNotFoundError(f"CV split file not found: {split_path}")

    metadata_path = Path(args.metadata_csv).expanduser().resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / "results" / "paper_figures" / "cv_distances"
    )
    membership = load_split_membership(split_path, load_metadata(metadata_path))

    summaries = []
    nearest_rows = []
    for fold in sorted(membership["fold"].unique()):
        summary, fold_nearest_rows = calculate_fold_distances(fold, membership)
        summaries.append(summary)
        nearest_rows.append(fold_nearest_rows)

    summary_frame = pd.DataFrame(summaries).sort_values("fold")
    nearest_frame = pd.concat(nearest_rows, ignore_index=True).sort_values(
        ["fold", "nearest_train_distance_km"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "cv_minimum_train_validation_distances.csv"
    nearest_path = output_dir / "cv_validation_nearest_train_distances.csv"
    summary_frame.to_csv(summary_path, index=False, float_format="%.3f")
    nearest_frame.to_csv(nearest_path, index=False, float_format="%.3f")

    print(f"Saved fold summary: {summary_path}")
    print(f"Saved validation-cube distances: {nearest_path}")
    print(
        summary_frame.to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )


if __name__ == "__main__":
    main()
