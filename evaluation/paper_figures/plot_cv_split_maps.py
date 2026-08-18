"""Plot the spatial distribution of the train and validation cubes per CV fold.

The script uses the cv_splits.json saved with a completed CV run. It therefore
visualises the exact cube assignments used during training rather than creating
new splits from the metadata table.

Example
-------
python -m evaluation.paper_figures.plot_cv_split_maps \
    --run-dir /net/home/sloeblein/ARCEME-Vegetation-Recovery/model/wand_db_logs/CV_Training_SGEDConvLSTM_RGBI_2026-08-03_20-20-39 \
    --metadata-csv data_processing/data/train_test_split.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


FOLD_PATTERN = re.compile(r"fold_(\d+)$")


def cube_id_from_path(path: str) -> str:
    """Extract an ARCEME cube identifier from a Zarr path."""
    name = Path(path).name
    if name.endswith("_postprocessed.zarr"):
        return name.removesuffix("_postprocessed.zarr")
    if name.endswith(".zarr"):
        return name.removesuffix(".zarr")
    raise ValueError(f"Expected a .zarr cube path, received: {path}")


def load_metadata(path: Path) -> pd.DataFrame:
    """Read the metadata required to locate the development cubes."""
    metadata = pd.read_csv(path)
    required = {"DisNo.", "latitude", "longitude", "koppen_geiger"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(
            f"Metadata file is missing required columns: {sorted(missing)}"
        )

    metadata = metadata.copy()
    metadata["DisNo."] = metadata["DisNo."].astype(str)
    metadata["latitude"] = pd.to_numeric(metadata["latitude"], errors="coerce")
    metadata["longitude"] = pd.to_numeric(metadata["longitude"], errors="coerce")
    metadata = metadata.dropna(subset=["latitude", "longitude"])

    if metadata["DisNo."].duplicated().any():
        duplicates = metadata.loc[
            metadata["DisNo."].duplicated(keep=False), "DisNo."
        ].tolist()
        raise ValueError(f"Duplicate cube identifiers in metadata: {duplicates[:5]}")
    return metadata


def load_split_membership(split_path: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    """Build one train/validation membership table for each stored fold."""
    with split_path.open("r", encoding="utf-8") as handle:
        splits = json.load(handle)

    fold_keys = sorted(
        (key for key in splits if FOLD_PATTERN.fullmatch(key)),
        key=lambda key: int(FOLD_PATTERN.fullmatch(key).group(1)),
    )
    if not fold_keys:
        raise ValueError(f"No fold_# entries found in {split_path}")

    metadata_by_id = metadata.set_index("DisNo.", verify_integrity=True)
    rows: list[pd.DataFrame] = []

    for fold_key in fold_keys:
        fold = int(FOLD_PATTERN.fullmatch(fold_key).group(1))
        split = splits[fold_key]
        train_ids = [cube_id_from_path(path) for path in split["train_files"]]
        val_ids = [cube_id_from_path(path) for path in split["val_files"]]

        overlap = sorted(set(train_ids).intersection(val_ids))
        if overlap:
            raise ValueError(f"Fold {fold} contains train/validation overlap: {overlap[:5]}")
        if len(train_ids) != int(split["num_train"]):
            raise ValueError(f"Stored train count does not match train files in fold {fold}")
        if len(val_ids) != int(split["num_val"]):
            raise ValueError(f"Stored validation count does not match validation files in fold {fold}")

        for subset, cube_ids in (("train", train_ids), ("validation", val_ids)):
            missing = sorted(set(cube_ids).difference(metadata_by_id.index))
            if missing:
                raise ValueError(
                    f"{len(missing)} {subset} cubes in fold {fold} are absent from metadata; "
                    f"first: {missing[0]}"
                )
            subset_frame = metadata_by_id.loc[cube_ids].reset_index().copy()
            subset_frame["fold"] = fold
            subset_frame["subset"] = subset
            rows.append(subset_frame)

    membership = pd.concat(rows, ignore_index=True)
    for fold, frame in membership.groupby("fold"):
        train_classes = set(frame.loc[frame["subset"] == "train", "koppen_geiger"])
        val_classes = set(frame.loc[frame["subset"] == "validation", "koppen_geiger"])
        shared_classes = sorted(train_classes.intersection(val_classes))
        if shared_classes:
            raise ValueError(
                f"Fold {fold} is not climate-class grouped. Shared classes: "
                f"{shared_classes}"
            )
    return membership


def _cartopy_modules():
    """Import Cartopy only when a map is actually requested."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as error:
        raise ImportError(
            "This figure requires Cartopy. Install it in the HPC environment with "
            "'conda install -c conda-forge cartopy'."
        ) from error
    return ccrs, cfeature


def plot_fold_maps(membership: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Create one global map panel for every cross-validation fold."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ccrs, cfeature = _cartopy_modules()
    folds = sorted(membership["fold"].unique())
    figure, axes = plt.subplots(
        1,
        len(folds),
        figsize=(6.4 * len(folds), 5.2),
        subplot_kw={"projection": ccrs.Robinson()},
        layout="constrained",
    )
    if len(folds) == 1:
        axes = [axes]

    train_colour = "#4C78A8"
    validation_colour = "#D1495B"
    geographic_crs = ccrs.PlateCarree()

    for axis, fold in zip(axes, folds):
        frame = membership[membership["fold"] == fold]
        train = frame[frame["subset"] == "train"]
        validation = frame[frame["subset"] == "validation"]

        axis.set_global()
        axis.add_feature(cfeature.LAND, facecolor="#F2F2F2")
        axis.add_feature(cfeature.OCEAN, facecolor="#FFFFFF")
        axis.coastlines(linewidth=0.45, color="#666666")
        axis.gridlines(linewidth=0.25, color="#B8B8B8", alpha=0.6)

        axis.scatter(
            train["longitude"],
            train["latitude"],
            transform=geographic_crs,
            s=22,
            c=train_colour,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )
        axis.scatter(
            validation["longitude"],
            validation["latitude"],
            transform=geographic_crs,
            s=35,
            c=validation_colour,
            edgecolors="#4A1020",
            linewidths=0.45,
            zorder=4,
        )
        axis.set_title(
            f"Fold {fold + 1}\nTrain: n={len(train)} | Validation: n={len(validation)}",
            fontsize=11,
            fontweight="bold",
        )

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=train_colour,
            markeredgecolor="white", markersize=7, label="Training cubes"
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=validation_colour,
            markeredgecolor="#4A1020", markersize=7, label="Validation cubes"
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    figure.suptitle(
        "Climate-class-grouped cross-validation splits",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def summarise_folds(membership: pd.DataFrame) -> pd.DataFrame:
    """Return cube counts and withheld climate classes for each fold."""
    rows = []
    for fold, frame in membership.groupby("fold"):
        train = frame[frame["subset"] == "train"]
        validation = frame[frame["subset"] == "validation"]
        rows.append(
            {
                "fold": fold,
                "n_train_cubes": len(train),
                "n_validation_cubes": len(validation),
                "n_train_climate_classes": train["koppen_geiger"].nunique(),
                "n_validation_climate_classes": validation["koppen_geiger"].nunique(),
                "validation_climate_classes": "; ".join(
                    sorted(validation["koppen_geiger"].unique())
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("fold")


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
        help="Metadata CSV with cube IDs, coordinates and Köppen-Geiger classes.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <run-dir>/results/paper_figures/cv_splits.",
    )
    parser.add_argument("--dpi", type=int, default=300)
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
        else run_dir / "results" / "paper_figures" / "cv_splits"
    )
    membership = load_split_membership(split_path, load_metadata(metadata_path))
    summary = summarise_folds(membership)

    output_dir.mkdir(parents=True, exist_ok=True)
    membership.to_csv(output_dir / "cv_split_membership.csv", index=False)
    summary.to_csv(output_dir / "cv_split_summary.csv", index=False)
    plot_fold_maps(membership, output_dir / "cv_train_validation_maps.png", args.dpi)

    print(f"Saved split map: {output_dir / 'cv_train_validation_maps.png'}")
    print(f"Saved split summary: {output_dir / 'cv_split_summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
