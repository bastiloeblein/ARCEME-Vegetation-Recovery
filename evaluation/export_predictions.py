"""Export per-cube prediction tensors for out-of-fold or holdout analysis."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from model.model_manager import ARCEMEPipeline


CUBE_ID_PATTERN = re.compile(r"2\d{3}-\d{4}-[A-Z]{3}")


def _absolute(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _cube_id(path: str | Path) -> str:
    match = CUBE_ID_PATTERN.search(str(path))
    if not match:
        raise ValueError(f"Could not extract ARCEME cube id from: {path}")
    return match.group(0)


def _read_config(run_dir: Path) -> tuple[dict, Path]:
    config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run configuration: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML configuration: {config_path}")
    return cfg, config_path


def _optical_input_label(cfg: dict) -> str:
    variables = cfg["data"]["variables"].get("s2", [])
    names = {str(value).upper() for value in variables}
    raw_bands = {"B02", "B03", "B04", "B8A"}
    indices = {"IRECI", "NDMI", "NIRV"}
    if raw_bands.intersection(names):
        return "rgb_nir"
    if indices.intersection(names):
        return "indices"
    return "custom_" + "_".join(sorted(names))


def _read_test_list(path: str | None) -> list[str] | None:
    if path is None:
        return None
    list_path = Path(path).expanduser().resolve()
    if not list_path.exists():
        raise FileNotFoundError(list_path)
    files = [
        line.strip()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    invalid = [value for value in files if not value.endswith(".zarr")]
    if invalid:
        raise ValueError(
            f"Every test-list line must end in .zarr; invalid examples: {invalid[:3]}"
        )
    return files


def _expected_tensor_paths(output_dir: Path, cube_ids: Iterable[str]) -> list[Path]:
    return [output_dir / "tensors" / f"{cube_id}.zarr" for cube_id in cube_ids]


def _export_if_needed(
    pipeline: ARCEMEPipeline,
    checkpoint: str,
    source_files: list[str],
    output_dir: Path,
    overwrite: bool,
    plot_samples: bool,
) -> None:
    cube_ids = [_cube_id(path) for path in source_files]
    expected = _expected_tensor_paths(output_dir, cube_ids)
    existing = [path for path in expected if path.exists()]
    if len(existing) == len(expected) and not overwrite:
        print(f"Reusing complete export: {output_dir}")
        return
    if existing and not overwrite:
        raise FileExistsError(
            f"Partial export found in {output_dir} ({len(existing)}/{len(expected)} "
            "cubes). Re-run with --overwrite after checking the directory."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.evaluate(
        ckpt_path=checkpoint,
        test_files=source_files,
        output_dir=str(output_dir),
        plot_samples=plot_samples,
    )
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(
            f"Inference completed but {len(missing)} tensor stores are missing. "
            f"First missing path: {missing[0]}"
        )


def _base_manifest_fields(
    cfg: dict, run_dir: Path, config_path: Path
) -> dict[str, str]:
    return {
        "configuration": str(cfg.get("experiment_name", run_dir.name)),
        "architecture": str(cfg["model"]["model_type"]),
        "optical_input": _optical_input_label(cfg),
        "source_run_dir": _absolute(run_dir),
        "config_path": _absolute(config_path),
    }


def _write_manifest(
    rows: list[dict], output_root: Path, metadata: dict
) -> Path:
    frame = pd.DataFrame(rows)
    required = {"configuration", "stage", "cube_id", "tensor_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest rows are missing fields: {sorted(missing)}")
    duplicate = frame.duplicated(["configuration", "stage", "cube_id"], keep=False)
    if duplicate.any():
        values = frame.loc[duplicate, ["configuration", "stage", "cube_id"]]
        raise ValueError(f"Duplicate prediction rows:\n{values.to_string(index=False)}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "prediction_manifest.csv"
    frame.sort_values(["configuration", "stage", "fold", "cube_id"], na_position="last").to_csv(
        manifest_path, index=False
    )
    metadata = {
        **metadata,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": _absolute(manifest_path),
        "n_predictions": int(len(frame)),
        "lead_days": [5, 10, 15, 20, 25, 30],
    }
    with (output_root / "export_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Prediction manifest written to: {manifest_path}")
    return manifest_path


def export_oof(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg, config_path = _read_config(run_dir)
    if args.num_workers is not None:
        cfg["data"]["data_loader"]["num_workers"] = args.num_workers
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    split_path = run_dir / "cv_splits.json"
    summary_path = run_dir / "cv_summary.json"
    if not split_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "OOF export requires both cv_splits.json and cv_summary.json in the run directory."
        )
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        run_dir / "results" / "oof_predictions"
    )

    pipeline = ARCEMEPipeline(config=cfg, mode="eval", run_dir=str(run_dir))
    common = _base_manifest_fields(cfg, run_dir, config_path)
    rows: list[dict] = []
    seen_ids: set[str] = set()
    k_folds = int(cfg["cross_validation"]["k_folds"])

    for fold in range(k_folds):
        fold_key = f"fold_{fold}"
        if fold_key not in splits:
            raise KeyError(f"Missing {fold_key} in {split_path}")
        val_files = [str(path) for path in splits[fold_key]["val_files"]]
        checkpoint = pipeline.get_checkpoint_path(fold, type="best")
        if not checkpoint or not Path(checkpoint).exists():
            raise FileNotFoundError(f"No best checkpoint found for fold {fold}: {checkpoint}")

        fold_ids = [_cube_id(path) for path in val_files]
        overlap = seen_ids.intersection(fold_ids)
        if overlap:
            raise ValueError(
                f"Validation cubes occur in more than one fold: {sorted(overlap)[:5]}"
            )
        seen_ids.update(fold_ids)

        fold_output = output_root / f"fold_{fold}"
        _export_if_needed(
            pipeline,
            checkpoint=checkpoint,
            source_files=val_files,
            output_dir=fold_output,
            overwrite=args.overwrite,
            plot_samples=args.plots,
        )
        for cube_id, source_path in zip(fold_ids, val_files):
            rows.append(
                {
                    **common,
                    "stage": "oof",
                    "fold": fold,
                    "cube_id": cube_id,
                    "tensor_path": _absolute(
                        fold_output / "tensors" / f"{cube_id}.zarr"
                    ),
                    "source_cube_path": _absolute(source_path),
                    "checkpoint_path": _absolute(checkpoint),
                    "cv_summary_path": _absolute(summary_path),
                }
            )

    return _write_manifest(
        rows,
        output_root,
        {
            "stage": "oof",
            "source_run_dir": _absolute(run_dir),
            "recommended_final_refit_epochs": summary.get(
                "recommended_final_refit_epochs"
            ),
        },
    )


def export_holdout(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg, config_path = _read_config(run_dir)
    if args.num_workers is not None:
        cfg["data"]["data_loader"]["num_workers"] = args.num_workers
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    final_summary_path = run_dir / "final_refit_summary.json"
    if not final_summary_path.exists():
        raise FileNotFoundError(
            f"Holdout export requires a completed final refit: {final_summary_path}"
        )
    final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    checkpoint = final_summary.get("final_checkpoint")
    if not checkpoint or not Path(checkpoint).exists():
        raise FileNotFoundError(f"Invalid final checkpoint: {checkpoint}")

    pipeline = ARCEMEPipeline(config=cfg, mode="eval", run_dir=str(run_dir))
    source_files = _read_test_list(args.test_list) or pipeline.prepare_data()
    source_files = sorted(str(path) for path in source_files)
    if not source_files:
        raise RuntimeError("No holdout cubes found.")

    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        run_dir / "results" / "holdout_predictions"
    )
    saved_tensor_dir = run_dir / "tensors"
    saved_tensors = _expected_tensor_paths(
        run_dir, [_cube_id(path) for path in source_files]
    )
    if not args.overwrite and all(path.exists() for path in saved_tensors):
        # A completed model/evaluate.py run already writes these tensors.
        tensor_dir = saved_tensor_dir
        print(f"Reusing saved holdout tensors: {tensor_dir}")
    else:
        _export_if_needed(
            pipeline,
            checkpoint=checkpoint,
            source_files=source_files,
            output_dir=output_root,
            overwrite=args.overwrite,
            plot_samples=args.plots,
        )
        tensor_dir = output_root / "tensors"

    common = _base_manifest_fields(cfg, run_dir, config_path)
    rows = []
    for source_path in source_files:
        cube_id = _cube_id(source_path)
        rows.append(
            {
                **common,
                "stage": "holdout",
                    "fold": pd.NA,
                    "cube_id": cube_id,
                    "tensor_path": _absolute(
                        tensor_dir / f"{cube_id}.zarr"
                    ),
                "source_cube_path": _absolute(source_path),
                "checkpoint_path": _absolute(checkpoint),
                "cv_summary_path": str(final_summary.get("source_cv_run") or ""),
            }
        )
    return _write_manifest(
        rows,
        output_root,
        {
            "stage": "holdout",
            "source_run_dir": _absolute(run_dir),
            "final_refit_summary": final_summary,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--run-dir", required=True)
        subparser.add_argument("--output-dir", default=None)
        subparser.add_argument("--num-workers", type=int, default=None)
        subparser.add_argument("--batch-size", type=int, default=None)
        subparser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace prediction stores for expected cube ids. Stale unrelated files are not deleted.",
        )
        subparser.add_argument(
            "--plots",
            action="store_true",
            help="Also create expensive full-cube qualitative plots.",
        )

    oof = subparsers.add_parser(
        "oof", help="Export every best fold checkpoint on its own validation cubes."
    )
    add_shared(oof)
    oof.set_defaults(func=export_oof)

    holdout = subparsers.add_parser(
        "holdout", help="Export the completed final-refit checkpoint on holdout cubes."
    )
    add_shared(holdout)
    holdout.add_argument(
        "--test-list",
        default=None,
        help="Optional text file with one holdout .zarr path per line.",
    )
    holdout.set_defaults(func=export_holdout)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
