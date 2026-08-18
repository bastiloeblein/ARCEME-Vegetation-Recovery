"""Regression test for the auditable CV-to-final-refit selection step."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "model" / "select_final_refit.py"


def write_cv_run(
    root: Path,
    model_type: str,
    predictor_set: str,
    scores: list[float],
    epochs: list[int],
) -> Path:
    run_dir = root / f"CV_Training_{model_type}_{predictor_set}_2026-08-16_12-00-00"
    run_dir.mkdir()
    s2 = (
        ["kNDVI", "IRECI", "NDMI", "NIRv"]
        if predictor_set == "Indices"
        else ["kNDVI", "B02", "B03", "B04", "B8A"]
    )
    config = {
        "experiment_name": f"CV_Training_{model_type}_{predictor_set}",
        "data": {"variables": {"s2": s2}},
        "training": {
            "seed": 777,
            "validation": {
                "monitor": {"metric": "val/grand_mean_macro/NNSE"},
                "monitor_mode": "max",
                "min_valid_target_coverage": 0.15,
                "min_valid_target_count": 1000,
                "min_target_variance": 1e-6,
            },
            "final_refit": {"enabled": False},
        },
        "cross_validation": {"enabled": True, "k_folds": 3, "type": "llto"},
        "model": {"model_type": model_type, "checkpoint_path": None},
    }
    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    split_info = {}
    fold_results = []
    for fold, (score, epoch) in enumerate(zip(scores, epochs)):
        checkpoint = run_dir / f"best-model-epoch={epoch}-val.ckpt"
        checkpoint.touch()
        split_info[f"fold_{fold}"] = {
            "train_files": [f"cube_{fold}_train.zarr"],
            "val_files": [f"cube_{fold}_val.zarr"],
            "num_train": 1,
            "num_val": 1,
        }
        fold_results.append(
            {
                "fold": fold,
                "best_score": score,
                "best_checkpoint": str(checkpoint),
                "best_epoch": epoch,
            }
        )
    (run_dir / "cv_splits.json").write_text(json.dumps(split_info), encoding="utf-8")
    (run_dir / "cv_summary.json").write_text(
        json.dumps(
            {
                "folds": fold_results,
                "mean_val_score": sum(scores) / len(scores),
                "recommended_final_refit_epochs": sorted(epochs)[1] + 1,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


class FinalRefitSelectionTest(unittest.TestCase):
    def test_selects_equal_fold_mean_winner_and_writes_complete_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runs_dir = Path(temporary_directory) / "wand_db_logs"
            runs_dir.mkdir()
            write_cv_run(runs_dir, "SGConvLSTM", "Indices", [0.6, 0.6, 0.6], [9, 11, 13])
            write_cv_run(runs_dir, "SGConvLSTM", "RGBI", [0.61, 0.61, 0.61], [8, 10, 12])
            write_cv_run(runs_dir, "SGEDConvLSTM", "Indices", [0.62, 0.62, 0.62], [7, 9, 11])
            write_cv_run(runs_dir, "SGEDConvLSTM", "RGBI", [0.7, 0.7, 0.7], [20, 22, 24])
            output_dir = Path(temporary_directory) / "selection"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--runs-dir",
                    str(runs_dir),
                    "--output-dir",
                    str(output_dir),
                    "--final-seed",
                    "2026",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            report = json.loads((output_dir / "selection_report.json").read_text(encoding="utf-8"))
            config = yaml.safe_load((output_dir / "final_refit_config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(report["selected"]["model"], "SGEDConvLSTM")
            self.assertEqual(report["selected"]["predictor_set"], "RGBI_kNDVI")
            self.assertEqual(report["selected"]["recommended_final_refit_epochs"], 23)
            self.assertTrue(report["same_validation_splits_verified"])
            self.assertTrue(config["training"]["final_refit"]["enabled"])
            self.assertEqual(config["training"]["final_refit"]["epochs"], 23)
            self.assertEqual(config["training"]["seed"], 2026)
            self.assertFalse(config["cross_validation"]["enabled"])
            self.assertTrue(config["testing"]["save_metrics"])
            self.assertTrue(config["testing"]["save_tensors"])


if __name__ == "__main__":
    unittest.main()
