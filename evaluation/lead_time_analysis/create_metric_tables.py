"""Create cube, forecast-step and landcover tables from saved predictions."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.lead_time_analysis.metrics_starter import (
    load_cube,
    cube_table,
    macro_summary,
)


# Paths and dataset selection.
RUN_DIR = (
    REPO / "model/wand_db_logs/FinalRefit_SGEDConvLSTM_RGBI_kNDVI_2026-10-07_19-31-23"
)
SOURCE_DIR = Path(
    "/net/projects/arceme/vegetation_recovery_prediction/data/final_new/test"
)
TABLE_DIR = REPO / "evaluation/analysis_data"

DATASETS = [
    {
        "configuration": "SGEDConvLSTM RGB+NIR",
        "split": "holdout",
        "fold": -1,
        "evaluation_dir": RUN_DIR,
        "config_path": RUN_DIR / "config_used.yaml",
        "source_dir": SOURCE_DIR,
        "expected_cubes": 26,
    }
]

# Optional: a CV run with exported validation predictions for every fold.
CV_RUN = None
# CV_RUN = REPO / "model/wand_db_logs/CV_Training_SGEDConvLSTM_RGBI_2026-08-03_20-20-39"

if CV_RUN is not None:
    for fold in range(3):
        DATASETS.append(
            {
                "configuration": "SGEDConvLSTM RGB+NIR",
                "split": "cv",
                "fold": fold,
                "evaluation_dir": CV_RUN / f"results/oof_predictions/fold_{fold}",
                "config_path": CV_RUN / "config_used.yaml",
                "source_dir": SOURCE_DIR.parent / "train",
                "expected_cubes": [55, 48, 43][fold],
            }
        )

# NSE/NNSE eligibility for individual steps and landcover subsets.
SLICE_RULES = {
    "min_count": 1000,
    "min_coverage": 0.15,
    "min_variance": 1e-6,
}


def main():
    tables = []

    for dataset in DATASETS:
        evaluation_dir = dataset["evaluation_dir"]
        with dataset["config_path"].open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        # Full-cube eligibility comes from the original run.
        validation = config["training"]["validation"]
        cube_rules = {
            "min_count": int(validation["min_valid_target_count"]),
            "min_coverage": float(validation["min_valid_target_coverage"]),
            "min_variance": float(validation["min_target_variance"]),
        }

        audit = pd.read_csv(evaluation_dir / "metrics/cube_metrics.csv")
        audit = audit.rename(columns={"id": "cube_id"})
        if audit.cube_id.isna().any() or audit.cube_id.duplicated().any():
            raise ValueError(f"Missing or duplicate cube IDs in {evaluation_dir}.")
        if len(audit) != dataset["expected_cubes"]:
            raise ValueError(f"Unexpected cube count in {evaluation_dir}: {len(audit)}")
        if not audit.reconstruction_complete.astype(str).str.lower().eq("true").all():
            raise ValueError(
                "Incomplete reconstruction: inspect cube_metrics.csv first."
            )

        for row in audit.to_dict("records"):
            cube_id = str(row["cube_id"])
            print(
                f"Processing {dataset['split']} / fold {dataset['fold']} / {cube_id}",
                flush=True,
            )

            tensor_path = evaluation_dir / "tensors" / f"{cube_id}.zarr"
            matches = list(dataset["source_dir"].glob(f"{cube_id}*.zarr"))
            if len(matches) != 1:
                raise ValueError(
                    f"{cube_id}: expected one source cube, found {matches}"
                )
            source_path = matches[0]

            arrays = load_cube(
                tensor_path, source_path, row["n_valid_target_pixel_times"]
            )
            if int(arrays["vegetation"].sum()) != int(
                row["n_eligible_vegetation_pixels"]
            ):
                raise ValueError(
                    f"{cube_id}: source vegetation count differs from the audit."
                )

            metadata = {
                "configuration": dataset["configuration"],
                "split": dataset["split"],
                "fold": dataset["fold"],
                "cube_id": cube_id,
                "tensor_path": str(tensor_path),
                "source_path": str(source_path),
            }
            result = cube_table(arrays, metadata, True, cube_rules, SLICE_RULES)

            # Verify overall metrics against the original evaluation.
            for approach, suffix in [("Model", ""), ("Persistence", "_base")]:
                overall = result[
                    (result.scope == "overall")
                    & (result.landcover == "all")
                    & (result.approach == approach)
                ].iloc[0]

                for metric, old_name in [
                    ("mae", "mae"),
                    ("mse", "mse"),
                    ("bias", "bias"),
                    ("nse", "r2"),
                    ("nnse", "nnse"),
                ]:
                    reference = row[old_name + suffix]
                    if not np.isclose(
                        overall[metric],
                        reference,
                        rtol=1e-4,
                        atol=1e-7,
                        equal_nan=True,
                    ):
                        raise ValueError(
                            f"{cube_id}: {approach} {metric} differs from the original "
                            f"evaluation (calculated={overall[metric]}, saved={reference})."
                        )

            tables.append(result)

    detail = pd.concat(tables, ignore_index=True)
    summary = macro_summary(detail)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = TABLE_DIR / "metrics_detail.csv"
    summary_path = TABLE_DIR / "metrics_macro_by_fold.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved {len(detail)} detail rows: {detail_path}")
    print(f"Saved {len(summary)} summary rows: {summary_path}")


if __name__ == "__main__":
    main()
