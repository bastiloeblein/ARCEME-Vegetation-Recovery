"""Select one completed CV configuration and generate its final-refit config.

Run this script on Phaestos, where ``wand_db_logs`` and the original
checkpoints are available.  It deliberately selects a *configuration* (not a
single fold) by the equally weighted mean of the three fold-level best
validation scores.  The output is an auditable JSON/CSV report and a complete
YAML configuration for a fresh final refit.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


EXPECTED_FOLDS = {0, 1, 2}
EXPECTED_CONFIGURATION_GRID = {
    ("SGConvLSTM", "Indices"),
    ("SGConvLSTM", "RGBI_kNDVI"),
    ("SGEDConvLSTM", "Indices"),
    ("SGEDConvLSTM", "RGBI_kNDVI"),
}
ROBUST_METRIC_KEYS = (
    "min_valid_target_coverage",
    "min_valid_target_count",
    "min_target_variance",
)


@dataclass
class Candidate:
    """One complete CV run rooted in a single ``wand_db_logs`` directory."""

    run_dir: Path
    config: dict[str, Any]
    summary: dict[str, Any]
    split_signature: dict[str, tuple[str, ...]]
    model_type: str
    predictor_set: str
    monitor_metric: str
    monitor_mode: str
    fold_scores: dict[int, float]
    fold_epochs: dict[int, int]
    mean_score: float
    std_score: float
    recommended_epochs: int
    modified_at: float
    notes: list[str] = field(default_factory=list)

    @property
    def signature(self) -> tuple[str, str]:
        return self.model_type, self.predictor_set


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} does not contain a YAML mapping.")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} does not contain a JSON object.")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite numeric value, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    return number


def _predictor_set(config: dict[str, Any]) -> str:
    variables = config.get("data", {}).get("variables", {})
    s2 = set(variables.get("s2", [])) if isinstance(variables, dict) else set()
    if {"IRECI", "NDMI", "NIRv"}.issubset(s2):
        return "Indices"
    if {"B02", "B03", "B04", "B8A"}.issubset(s2):
        return "RGBI_kNDVI"
    return "Unknown"


def _extract_epoch(checkpoint_path: str | Path) -> int | None:
    match = re.search(r"epoch=(\d+)", str(checkpoint_path))
    return int(match.group(1)) if match else None


def _run_sort_timestamp(run_dir: Path) -> float:
    """Prefer ARCEME's immutable run-name timestamp over filesystem mtime."""
    match = re.search(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$", run_dir.name)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S").timestamp()
    # Fallback for manually renamed run directories.
    return (run_dir / "cv_summary.json").stat().st_mtime


def _normalise_splits(raw_splits: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return the sorted validation-file set for each expected fold."""
    signatures: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for fold in sorted(EXPECTED_FOLDS):
        key = f"fold_{fold}"
        entry = raw_splits.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"cv_splits.json is missing a mapping for {key}.")
        val_files = entry.get("val_files")
        if not isinstance(val_files, list) or not val_files:
            raise ValueError(f"{key}.val_files must be a non-empty list.")
        val_set = tuple(sorted(str(path) for path in val_files))
        if len(val_set) != len(set(val_set)):
            raise ValueError(f"{key}.val_files contains duplicate cube paths.")
        overlap = seen.intersection(val_set)
        if overlap:
            raise ValueError(
                f"Validation cubes overlap between folds; first overlap: {sorted(overlap)[0]}"
            )
        seen.update(val_set)
        signatures[key] = val_set
    extra_folds = {key for key in raw_splits if re.fullmatch(r"fold_\d+", key)}
    if extra_folds != {f"fold_{fold}" for fold in EXPECTED_FOLDS}:
        raise ValueError(
            "cv_splits.json must contain exactly fold_0, fold_1 and fold_2; "
            f"found {sorted(extra_folds)}."
        )
    return signatures


def _validate_current_protocol(config: dict[str, Any]) -> tuple[str, str]:
    cv = config.get("cross_validation", {})
    if not isinstance(cv, dict) or not cv.get("enabled") or cv.get("k_folds") != 3:
        raise ValueError("requires enabled three-fold cross-validation")

    refit = config.get("training", {}).get("final_refit", {})
    if isinstance(refit, dict) and refit.get("enabled"):
        raise ValueError("is already a final-refit configuration, not a CV run")

    validation = config.get("training", {}).get("validation", {})
    if not isinstance(validation, dict):
        raise ValueError("training.validation is missing")
    monitor = validation.get("monitor", {})
    if isinstance(monitor, dict):
        metric = monitor.get("metric")
    else:
        metric = validation.get("selection_metric")
    mode = validation.get("monitor_mode")
    if not isinstance(metric, str) or mode not in {"min", "max"}:
        raise ValueError("training.validation monitor metric/mode is incomplete")

    missing = [key for key in ROBUST_METRIC_KEYS if key not in validation]
    if missing:
        raise ValueError(
            "does not use the current robust R2/NNSE protocol; missing "
            + ", ".join(missing)
        )
    return metric, mode


def _parse_candidate(run_dir: Path) -> Candidate:
    config = _read_yaml(run_dir / "config_used.yaml")
    summary = _read_json(run_dir / "cv_summary.json")
    splits = _read_json(run_dir / "cv_splits.json")
    monitor_metric, monitor_mode = _validate_current_protocol(config)
    split_signature = _normalise_splits(splits)

    raw_folds = summary.get("folds")
    if not isinstance(raw_folds, list) or len(raw_folds) != len(EXPECTED_FOLDS):
        raise ValueError("cv_summary.json must contain exactly three fold results")

    by_fold: dict[int, dict[str, Any]] = {}
    for fold_result in raw_folds:
        if not isinstance(fold_result, dict):
            raise ValueError("cv_summary.json contains a non-mapping fold result")
        fold = fold_result.get("fold")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold in by_fold:
            raise ValueError("cv_summary.json has invalid or duplicate fold identifiers")
        by_fold[fold] = fold_result
    if set(by_fold) != EXPECTED_FOLDS:
        raise ValueError(
            "cv_summary.json must contain fold IDs 0, 1 and 2; "
            f"found {sorted(by_fold)}."
        )

    scores: dict[int, float] = {}
    epochs: dict[int, int] = {}
    for fold, result in by_fold.items():
        scores[fold] = _finite_number(result.get("best_score"), f"fold {fold} best_score")
        checkpoint = result.get("best_checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError(f"fold {fold} has no best checkpoint path")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise ValueError(f"fold {fold} checkpoint is unavailable: {checkpoint_path}")
        epoch = result.get("best_epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            epoch = _extract_epoch(checkpoint)
        if epoch is None or epoch < 0:
            raise ValueError(f"fold {fold} has no valid zero-based best epoch")
        epochs[fold] = epoch

    mean_score = statistics.fmean(scores.values())
    std_score = statistics.stdev(scores.values())
    reported_mean = summary.get("mean_val_score")
    notes: list[str] = []
    if reported_mean is not None:
        reported = _finite_number(reported_mean, "mean_val_score")
        if not math.isclose(reported, mean_score, rel_tol=1e-9, abs_tol=1e-12):
            notes.append(
                "The stored mean_val_score differs from the recomputed equal-fold mean; "
                "selection uses the recomputed value."
            )

    recommended = int(round(statistics.median(epochs.values()))) + 1
    stored_recommendation = summary.get("recommended_final_refit_epochs")
    if stored_recommendation != recommended:
        notes.append(
            "The stored recommended_final_refit_epochs is absent or inconsistent; "
            "the median zero-based best epoch + 1 was recomputed."
        )

    model = config.get("model", {}).get("model_type")
    if not isinstance(model, str) or not model:
        raise ValueError("model.model_type is missing")

    return Candidate(
        run_dir=run_dir.resolve(),
        config=config,
        summary=summary,
        split_signature=split_signature,
        model_type=model,
        predictor_set=_predictor_set(config),
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        fold_scores=scores,
        fold_epochs=epochs,
        mean_score=mean_score,
        std_score=std_score,
        recommended_epochs=recommended,
        modified_at=_run_sort_timestamp(run_dir),
        notes=notes,
    )


def _candidate_row(candidate: Candidate, status: str, reason: str = "") -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "run_dir": str(candidate.run_dir),
        "experiment_name": candidate.config.get("experiment_name"),
        "model": candidate.model_type,
        "predictor_set": candidate.predictor_set,
        "monitor_metric": candidate.monitor_metric,
        "monitor_mode": candidate.monitor_mode,
        "fold_0_score": candidate.fold_scores[0],
        "fold_1_score": candidate.fold_scores[1],
        "fold_2_score": candidate.fold_scores[2],
        "mean_score": candidate.mean_score,
        "std_score": candidate.std_score,
        "fold_0_epoch": candidate.fold_epochs[0],
        "fold_1_epoch": candidate.fold_epochs[1],
        "fold_2_epoch": candidate.fold_epochs[2],
        "recommended_final_refit_epochs": candidate.recommended_epochs,
        "modified_at": datetime.fromtimestamp(candidate.modified_at).isoformat(timespec="seconds"),
        "notes": " | ".join(candidate.notes),
    }


def _make_final_refit_config(
    candidate: Candidate, final_seed: int, experiment_name: str | None
) -> dict[str, Any]:
    config = copy.deepcopy(candidate.config)
    model_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate.model_type)
    predictor_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate.predictor_set)
    config["experiment_name"] = experiment_name or f"FinalRefit_{model_label}_{predictor_label}"

    training = config.setdefault("training", {})
    training["seed"] = final_seed
    training["max_epochs"] = candidate.recommended_epochs
    training["final_refit"] = {
        "enabled": True,
        "epochs": candidate.recommended_epochs,
        "source_cv_run": str(candidate.run_dir),
    }
    config.setdefault("cross_validation", {})["enabled"] = False
    config.setdefault("testing", {})["save_metrics"] = True
    config["testing"]["save_tensors"] = True

    # A final refit must start from scratch; this is also robust to a config
    # copied from an old experiment that happened to retain a model path.
    config.setdefault("model", {})["checkpoint_path"] = None
    config["model"].pop("run_dir", None)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit completed CV runs and generate one full final-refit config."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("wand_db_logs"),
        help="Directory containing CV run folders (default: ./wand_db_logs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New directory for the selection report and generated YAML.",
    )
    parser.add_argument(
        "--final-seed",
        type=int,
        default=2026,
        help="Predeclared random seed to use for the fresh final refit (default: 2026).",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional W&B/run name override for the generated final-refit config.",
    )
    parser.add_argument(
        "--allow-incomplete-grid",
        action="store_true",
        help=(
            "Allow selection without all four planned SG/SGED × Indices/RGBI "
            "configurations. This is intended only for deliberate reduced studies."
        ),
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {runs_dir}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else runs_dir / f"final_refit_selection_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    )
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}. "
            "Choose a new --output-dir."
        )

    accepted: list[Candidate] = []
    rejected: list[dict[str, str]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        required = [run_dir / name for name in ("config_used.yaml", "cv_summary.json", "cv_splits.json")]
        if not any(path.exists() for path in required):
            continue
        if not all(path.is_file() for path in required):
            rejected.append({"run_dir": str(run_dir.resolve()), "reason": "incomplete CV artefacts"})
            continue
        try:
            accepted.append(_parse_candidate(run_dir))
        except (OSError, ValueError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
            rejected.append({"run_dir": str(run_dir.resolve()), "reason": str(error)})

    if not accepted:
        reasons = "\n".join(f"- {row['run_dir']}: {row['reason']}" for row in rejected)
        raise RuntimeError(
            "No complete, current-protocol three-fold CV run was found.\n" + reasons
        )

    # Keep only the newest complete run for each architecture/predictor pair.
    # Earlier runs remain in the audit report instead of being silently mixed in.
    newest_by_signature: dict[tuple[str, str], Candidate] = {}
    superseded: list[Candidate] = []
    for candidate in accepted:
        previous = newest_by_signature.get(candidate.signature)
        if previous is None or candidate.modified_at > previous.modified_at:
            if previous is not None:
                superseded.append(previous)
            newest_by_signature[candidate.signature] = candidate
        else:
            superseded.append(candidate)
    candidates = list(newest_by_signature.values())

    # CHANGED: The thesis comparison is only complete when all planned
    # architecture/input combinations are present.  Failing here prevents an
    # accidental SGED-only selection when a current SG CV run was not exported.
    if not args.allow_incomplete_grid:
        available = {candidate.signature for candidate in candidates}
        missing = EXPECTED_CONFIGURATION_GRID - available
        if missing:
            labels = ", ".join(f"{model} / {predictors}" for model, predictors in sorted(missing))
            raise RuntimeError(
                "The planned four-way CV comparison is incomplete. Missing: " + labels
            )

    metric_mode_pairs = {(item.monitor_metric, item.monitor_mode) for item in candidates}
    if len(metric_mode_pairs) != 1:
        detail = ", ".join(f"{metric} ({mode})" for metric, mode in sorted(metric_mode_pairs))
        raise RuntimeError(
            "The active candidate configurations use different selection metrics/modes: "
            + detail
        )

    reference_splits = candidates[0].split_signature
    mismatch = [item for item in candidates if item.split_signature != reference_splits]
    if mismatch:
        paths = ", ".join(str(item.run_dir) for item in mismatch)
        raise RuntimeError(
            "The active candidates do not use identical validation cubes in every fold. "
            "Do not compare them for model selection. Mismatching runs: " + paths
        )

    _, monitor_mode = next(iter(metric_mode_pairs))
    selected = (
        min(candidates, key=lambda item: item.mean_score)
        if monitor_mode == "min"
        else max(candidates, key=lambda item: item.mean_score)
    )

    output_dir.mkdir(parents=True)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_rule": (
            "lowest equally weighted mean of the three fold-level best validation scores"
            if monitor_mode == "min"
            else "highest equally weighted mean of the three fold-level best validation scores"
        ),
        "monitor_metric": selected.monitor_metric,
        "monitor_mode": selected.monitor_mode,
        "expected_folds": sorted(EXPECTED_FOLDS),
        "same_validation_splits_verified": True,
        "same_validation_cube_counts": {
            fold: len(paths) for fold, paths in reference_splits.items()
        },
        "selected": _candidate_row(selected, "selected"),
        "active_candidates": [_candidate_row(item, "active") for item in candidates],
        "superseded_complete_runs": [
            _candidate_row(item, "superseded", "newer complete run with same model/predictor pair")
            for item in superseded
        ],
        "rejected_runs": rejected,
    }
    with (output_dir / "selection_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    rows = [
        _candidate_row(item, "active") for item in sorted(candidates, key=lambda item: item.signature)
    ]
    rows.extend(
        _candidate_row(item, "superseded", "newer complete run with same model/predictor pair")
        for item in sorted(superseded, key=lambda item: item.signature)
    )
    fields = list(rows[0]) if rows else ["status", "run_dir", "reason"]
    with (output_dir / "cv_candidate_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    refit_config = _make_final_refit_config(
        selected, final_seed=args.final_seed, experiment_name=args.experiment_name
    )
    config_path = output_dir / "final_refit_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(refit_config, handle, sort_keys=False, allow_unicode=True)

    print("\nFinal-refit selection complete")
    print(f"  Selected: {selected.model_type} / {selected.predictor_set}")
    print(f"  CV mean {selected.monitor_metric}: {selected.mean_score:.6f} ± {selected.std_score:.6f}")
    print(f"  CV best epochs (zero-based): {selected.fold_epochs}")
    print(f"  Final-refit epochs (median + 1): {selected.recommended_epochs}")
    print(f"  Generated config: {config_path}")
    print(f"  Audit report: {output_dir / 'selection_report.json'}")


if __name__ == "__main__":
    main()
