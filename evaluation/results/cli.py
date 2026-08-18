"""Command-line workflow for the three core ARCEME results analyses.

The commands intentionally keep model selection, final confirmation and
conditional/exploratory analysis separate::

    python -m evaluation.results.cli cv ...
    python -m evaluation.results.cli holdout ...
    python -m evaluation.results.cli conditional ...
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import yaml

from .core import (
    add_event_tertiles,
    analyze_manifest,
    bootstrap_rows,
    fold_macro_mean,
    grouped_bootstrap_summary,
    load_manifests,
    ratio_of_means_skill,
    read_recommended_refit_epochs,
    standardized_bootstrap_beta,
    stratified_bootstrap_rows,
    summarize_statistic,
)
from .plotting import (
    plot_cv_model_selection,
    plot_environment_groups,
    plot_event_tertiles,
    plot_factor_contrasts,
    plot_holdout_lead_time,
    plot_r30_scatter,
)


def _directories(output_dir: str | Path) -> tuple[Path, Path, Path]:
    root = Path(output_dir).expanduser().resolve()
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return root, tables, figures


def _write_metadata(root: Path, command: str, args: argparse.Namespace, extra: dict) -> None:
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "func"
    }
    with (root / "analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({"command": command, "arguments": serializable, **extra}, handle, indent=2)


def _analyze(
    args: argparse.Namespace, *, include_landcover: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = load_manifests(args.manifest)
    cubes, steps, landcover_cubes, landcover_steps = analyze_manifest(
        manifest,
        include_landcover=include_landcover,
        min_landcover_pixels=getattr(args, "min_landcover_pixels", 100),
        min_valid_target_coverage=args.min_valid_target_coverage,
        min_valid_target_count=args.min_valid_target_count,
        min_target_variance=args.min_target_variance,
        response_threshold=args.response_threshold,
    )
    return manifest, cubes, steps, landcover_cubes, landcover_steps


def _validate_cv_alignment(cubes: pd.DataFrame) -> None:
    configurations = sorted(cubes["configuration"].unique())
    if len(configurations) < 2:
        raise ValueError("CV model selection needs at least two configurations.")
    sets = {
        configuration: set(
            cubes.loc[cubes["configuration"] == configuration, "cube_id"]
        )
        for configuration in configurations
    }
    reference_name = configurations[0]
    reference = sets[reference_name]
    for configuration, cube_ids in sets.items():
        if cube_ids != reference:
            missing = sorted(reference.difference(cube_ids))[:5]
            extra = sorted(cube_ids.difference(reference))[:5]
            raise ValueError(
                f"OOF cube sets differ: {configuration} versus {reference_name}; "
                f"missing={missing}, extra={extra}"
            )

    signature_counts = cubes.groupby("cube_id")["data_signature"].nunique()
    mismatch = signature_counts[signature_counts != 1]
    if not mismatch.empty:
        raise ValueError(
            "true/mask/base differ between configurations for cubes: "
            + ", ".join(mismatch.index[:5])
        )
    fold_counts = cubes.groupby("cube_id")["fold"].nunique(dropna=False)
    mismatch = fold_counts[fold_counts != 1]
    if not mismatch.empty:
        raise ValueError(
            "Fold assignments differ between configurations for cubes: "
            + ", ".join(mismatch.index[:5])
        )


def _cv_summary(
    cubes: pd.DataFrame, n_boot: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    fold_rows = []
    metrics: list[tuple[str, Callable[[pd.DataFrame], float]]] = [
        ("fold_macro_nnse", lambda data: fold_macro_mean(data, "nnse")),
        ("mae", lambda data: fold_macro_mean(data, "mae")),
        ("mse_skill", lambda data: fold_macro_mean(data, "mse_skill")),
        (
            "persistence_win_rate",
            lambda data: fold_macro_mean(data, "persistence_beaten"),
        ),
        ("bias", lambda data: fold_macro_mean(data, "bias")),
    ]
    for configuration, data in cubes.groupby("configuration"):
        meta = data.iloc[0]
        for fold, fold_data in data.groupby("fold"):
            fold_rows.append(
                {
                    "configuration": configuration,
                    "architecture": meta["architecture"],
                    "optical_input": meta["optical_input"],
                    "fold": fold,
                    "n_cubes": int(fold_data["cube_id"].nunique()),
                    "macro_nnse": float(fold_data["nnse"].mean()),
                    "macro_mae": float(fold_data["mae"].mean()),
                    "macro_mse_skill": float(fold_data["mse_skill"].mean()),
                    "persistence_win_rate": float(
                        fold_data["persistence_beaten"].mean()
                    ),
                }
            )
        for metric, statistic in metrics:
            estimate, low, high = stratified_bootstrap_rows(
                data,
                statistic,
                strata="fold",
                n_boot=n_boot,
                rng=rng,
            )
            summary_rows.append(
                {
                    "configuration": configuration,
                    "architecture": meta["architecture"],
                    "optical_input": meta["optical_input"],
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_cubes": int(data["cube_id"].nunique()),
                    "n_folds": int(data["fold"].nunique()),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def _paired_cv_contrasts(
    cubes: pd.DataFrame, n_boot: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    configurations = sorted(cubes["configuration"].unique())
    for left, right in itertools.combinations(configurations, 2):
        columns = ["cube_id", "fold", "nnse", "mse_skill", "mae"]
        left_data = cubes[cubes["configuration"] == left][columns]
        right_data = cubes[cubes["configuration"] == right][columns]
        merged = left_data.merge(
            right_data,
            on=["cube_id", "fold"],
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        for metric in ("nnse", "mse_skill", "mae"):
            effect = merged[["cube_id", "fold"]].copy()
            effect["difference"] = (
                merged[f"{metric}_right"] - merged[f"{metric}_left"]
            )
            effect = effect[np.isfinite(effect["difference"])]
            estimate, low, high = stratified_bootstrap_rows(
                effect,
                lambda data: fold_macro_mean(data, "difference"),
                strata="fold",
                n_boot=n_boot,
                rng=rng,
            )
            rows.append(
                {
                    "contrast": f"{right} - {left}",
                    "left_configuration": left,
                    "right_configuration": right,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_cubes": int(effect["cube_id"].nunique()),
                    "higher_is_better": metric != "mae",
                }
            )
    return pd.DataFrame(rows)


def _preferred_pair(values: Sequence[str], lower: str, upper: str) -> tuple[str, str]:
    values = list(values)
    if lower in values and upper in values:
        return lower, upper
    if len(values) != 2:
        raise ValueError(f"Expected two factor levels, got {values}")
    return tuple(sorted(values))  # type: ignore[return-value]


def _factor_cv_contrasts(
    cubes: pd.DataFrame, n_boot: int, rng: np.random.Generator
) -> pd.DataFrame:
    architectures = cubes["architecture"].dropna().unique().tolist()
    optical_inputs = cubes["optical_input"].dropna().unique().tolist()
    if len(architectures) != 2 or len(optical_inputs) != 2:
        return pd.DataFrame()
    architecture_low, architecture_high = _preferred_pair(
        architectures, "SGConvLSTM", "SGEDConvLSTM"
    )
    input_low, input_high = _preferred_pair(optical_inputs, "rgb_nir", "indices")
    rows = []

    for metric in ("nnse", "mse_skill", "mae"):
        pivot = cubes.pivot_table(
            index=["cube_id", "fold"],
            columns=["architecture", "optical_input"],
            values=metric,
            aggfunc="first",
        ).dropna()
        needed = [
            (architecture_low, input_low),
            (architecture_low, input_high),
            (architecture_high, input_low),
            (architecture_high, input_high),
        ]
        if not set(needed).issubset(pivot.columns):
            continue
        effects = pivot.reset_index()[["cube_id", "fold"]].copy()
        effects["architecture_effect"] = 0.5 * (
            (pivot[(architecture_high, input_low)] - pivot[(architecture_low, input_low)]).to_numpy()
            + (pivot[(architecture_high, input_high)] - pivot[(architecture_low, input_high)]).to_numpy()
        )
        effects["input_effect"] = 0.5 * (
            (pivot[(architecture_low, input_high)] - pivot[(architecture_low, input_low)]).to_numpy()
            + (pivot[(architecture_high, input_high)] - pivot[(architecture_high, input_low)]).to_numpy()
        )
        effects["interaction"] = (
            (
                pivot[(architecture_high, input_high)]
                - pivot[(architecture_high, input_low)]
            )
            - (
                pivot[(architecture_low, input_high)]
                - pivot[(architecture_low, input_low)]
            )
        ).to_numpy()
        labels = {
            "architecture_effect": f"{architecture_high} - {architecture_low}",
            "input_effect": f"{input_high} - {input_low}",
            "interaction": "architecture × input interaction",
        }
        for column, label in labels.items():
            data = effects[np.isfinite(effects[column])]
            estimate, low, high = stratified_bootstrap_rows(
                data,
                lambda sample, value=column: fold_macro_mean(sample, value),
                strata="fold",
                n_boot=n_boot,
                rng=rng,
            )
            rows.append(
                {
                    "contrast": label,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_cubes": int(data["cube_id"].nunique()),
                    "higher_is_better": metric != "mae",
                }
            )
    return pd.DataFrame(rows)


def run_cv(args: argparse.Namespace) -> None:
    root, tables, figures = _directories(args.output_dir)
    rng = np.random.default_rng(args.seed)
    manifest, cubes, steps, _, _ = _analyze(args)
    if set(cubes["stage"]) != {"oof"}:
        raise ValueError("The cv command accepts OOF manifests only.")
    _validate_cv_alignment(cubes)

    summary, folds = _cv_summary(cubes, args.bootstrap, rng)
    pairwise = _paired_cv_contrasts(cubes, args.bootstrap, rng)
    factors = _factor_cv_contrasts(cubes, args.bootstrap, rng)
    selection_rows = summary[summary["metric"] == "fold_macro_nnse"].dropna(
        subset=["estimate"]
    )
    if selection_rows.empty:
        raise RuntimeError("No configuration has an eligible fold-macro NNSE.")
    winner = selection_rows.sort_values("estimate", ascending=False).iloc[0]
    selected = str(winner["configuration"])
    refit_epochs = read_recommended_refit_epochs(manifest, selected)
    if refit_epochs is None:
        raise RuntimeError(
            "Could not recover recommended_final_refit_epochs for the selected run. "
            "Check cv_summary_path in its prediction manifest."
        )
    selected_manifest = manifest[manifest["configuration"] == selected].iloc[0]
    refit_config_path = root / "final_refit_config.yaml"
    source_config_path = Path(str(selected_manifest["config_path"]))
    if not source_config_path.exists():
        raise FileNotFoundError(
            f"Selected run configuration is unavailable: {source_config_path}"
        )
    with source_config_path.open("r", encoding="utf-8") as handle:
        refit_config = copy.deepcopy(yaml.safe_load(handle))
    refit_config["experiment_name"] = f"FinalRefit_{selected}"
    refit_config["model"].pop("run_dir", None)
    refit_config["training"].setdefault("final_refit", {})
    refit_config["training"]["final_refit"].update(
        {
            "enabled": True,
            "epochs": refit_epochs,
            "source_cv_run": str(selected_manifest["source_run_dir"]),
        }
    )
    refit_config.setdefault("testing", {})["save_tensors"] = True
    refit_config["testing"]["save_metrics"] = True
    with refit_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(refit_config, handle, sort_keys=False)
    recommendation = {
        "selection_rule": "highest equal-fold mean of cube-macro NNSE",
        "selected_configuration": selected,
        "architecture": winner["architecture"],
        "optical_input": winner["optical_input"],
        "selection_score": float(winner["estimate"]),
        "selection_ci": [float(winner["ci_low"]), float(winner["ci_high"])],
        "recommended_final_refit_epochs": refit_epochs,
        "generated_final_refit_config": str(refit_config_path),
        "next_step": "Refit this configuration on all development cubes, then evaluate once on holdout.",
    }

    cubes.to_csv(tables / "cv_cube_metrics.csv", index=False)
    steps.to_csv(tables / "cv_step_metrics.csv", index=False)
    folds.to_csv(tables / "cv_fold_metrics.csv", index=False)
    summary.to_csv(tables / "cv_configuration_summary.csv", index=False)
    pairwise.to_csv(tables / "cv_pairwise_contrasts.csv", index=False)
    factors.to_csv(tables / "cv_factor_contrasts.csv", index=False)
    with (root / "selection_recommendation.json").open("w", encoding="utf-8") as handle:
        json.dump(recommendation, handle, indent=2)
    plot_cv_model_selection(summary, figures / "cv_model_selection.png")
    plot_factor_contrasts(factors, figures / "cv_factor_contrasts.png")
    _write_metadata(
        root,
        "cv",
        args,
        {"selected_configuration": selected, "n_configurations": int(cubes["configuration"].nunique())},
    )
    print(json.dumps(recommendation, indent=2))


def _holdout_overall_summary(
    cubes: pd.DataFrame, n_boot: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    definitions: list[tuple[str, Callable[[pd.DataFrame], float], pd.DataFrame]] = [
        ("mae", lambda data: float(data["mae"].mean()), cubes),
        ("mae_base", lambda data: float(data["mae_base"].mean()), cubes),
        ("mae_gain", lambda data: float(data["mae_gain"].mean()), cubes),
        ("mse_skill_ratio", ratio_of_means_skill, cubes),
        ("bias", lambda data: float(data["bias"].mean()), cubes),
        (
            "persistence_win_rate",
            lambda data: float(data["persistence_beaten"].mean()),
            cubes,
        ),
        (
            "r30_direction_accuracy",
            lambda data: float(data["r30_direction_match"].mean()),
            cubes,
        ),
        (
            "nnse",
            lambda data: float(data["nnse"].mean()),
            cubes[cubes["nnse_eligible"]],
        ),
        (
            "nnse_base",
            lambda data: float(data["nnse_base"].mean()),
            cubes[cubes["nnse_eligible"]],
        ),
    ]
    for name, statistic, data in definitions:
        rows.append(
            summarize_statistic(
                data,
                name,
                statistic,
                n_boot=n_boot,
                rng=rng,
            )
        )
    return pd.DataFrame(rows)


def _holdout_step_summary(
    steps: pd.DataFrame, n_boot: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    definitions: list[tuple[str, Callable[[pd.DataFrame], float]]] = [
        ("mae", lambda data: float(data["mae"].mean())),
        ("mae_base", lambda data: float(data["mae_base"].mean())),
        ("mse_skill_ratio", ratio_of_means_skill),
        ("bias", lambda data: float(data["bias"].mean())),
        ("observed_response", lambda data: float(data["observed_response"].mean())),
        ("predicted_response", lambda data: float(data["predicted_response"].mean())),
        ("persistence_win_rate", lambda data: float((data["mse"] < data["mse_base"]).mean())),
        ("spatial_valid_fraction", lambda data: float(data["spatial_valid_fraction"].mean())),
    ]
    for lead_day, data in steps.groupby("lead_day"):
        data = data[data["n_valid"] > 0]
        for name, statistic in definitions:
            estimate, low, high = bootstrap_rows(
                data, statistic, n_boot=n_boot, rng=rng
            )
            rows.append(
                {
                    "lead_day": int(lead_day),
                    "metric": name,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_cubes": int(data["cube_id"].nunique()),
                    "n_valid_pixel_times": int(data["n_valid"].sum()),
                }
            )
    return pd.DataFrame(rows)


def run_holdout(args: argparse.Namespace) -> None:
    root, tables, figures = _directories(args.output_dir)
    rng = np.random.default_rng(args.seed)
    _, cubes, steps, _, _ = _analyze(args)
    if set(cubes["stage"]) != {"holdout"}:
        raise ValueError("The holdout command accepts holdout manifests only.")
    if cubes["configuration"].nunique() != 1:
        raise ValueError(
            "Holdout is confirmatory: provide exactly one final-refit configuration."
        )
    overall = _holdout_overall_summary(cubes, args.bootstrap, rng)
    by_step = _holdout_step_summary(steps, args.bootstrap, rng)
    cubes.to_csv(tables / "holdout_cube_metrics.csv", index=False)
    steps.to_csv(tables / "holdout_cube_step_metrics.csv", index=False)
    overall.to_csv(tables / "holdout_overall.csv", index=False)
    by_step.to_csv(tables / "holdout_by_step.csv", index=False)
    plot_holdout_lead_time(by_step, figures / "holdout_lead_time.png")
    plot_r30_scatter(cubes, figures / "holdout_r30_response.png")
    _write_metadata(
        root,
        "holdout",
        args,
        {
            "configuration": str(cubes["configuration"].iloc[0]),
            "n_holdout_cubes": int(cubes["cube_id"].nunique()),
        },
    )
    print(overall.to_string(index=False))


def _merge_metadata(
    cubes: pd.DataFrame, metadata_path: str, id_column: str
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    if id_column not in metadata.columns:
        raise KeyError(f"Metadata id column '{id_column}' not found in {metadata_path}")
    metadata = metadata.rename(columns={id_column: "cube_id"})
    metadata["cube_id"] = metadata["cube_id"].astype(str)
    if metadata["cube_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate cube ids.")
    merged = cubes.merge(metadata, on="cube_id", how="left", validate="one_to_one", indicator=True)
    missing = merged.loc[merged["_merge"] != "both", "cube_id"].tolist()
    if missing:
        raise ValueError(f"Metadata missing for {len(missing)} OOF cubes: {missing[:5]}")
    return merged.drop(columns="_merge")


def _environment_summary(
    merged: pd.DataFrame,
    landcover: pd.DataFrame,
    *,
    n_boot: int,
    rng: np.random.Generator,
    min_group_cubes: int,
) -> pd.DataFrame:
    rows = []
    climate_column = None
    if "koppen_geiger" in merged.columns:
        climate_column = "koppen_geiger"
    elif "climate_class" in merged.columns:
        climate_column = "climate_class"
    if climate_column:
        merged = merged.copy()
        merged["climate_group"] = merged[climate_column].astype(str).str[0]
        merged.loc[merged["climate_group"].isin(["D", "E"]), "climate_group"] = "D/E"
        summary = grouped_bootstrap_summary(
            merged,
            "climate_group",
            ["mae_gain", "mse_skill", "observed_r30"],
            n_boot=n_boot,
            rng=rng,
        )
        summary = summary.rename(columns={"climate_group": "group"})
        summary["group_type"] = "climate"
        rows.append(summary)

    if not landcover.empty:
        summary = grouped_bootstrap_summary(
            landcover,
            "landcover_group",
            ["mae_gain", "mse_skill", "observed_r30"],
            n_boot=n_boot,
            rng=rng,
        )
        summary = summary.rename(columns={"landcover_group": "group"})
        summary["group_type"] = "landcover"
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    output = pd.concat(rows, ignore_index=True)
    output["included_in_main_plot"] = output["n_cubes"] >= min_group_cubes
    return output


def _event_analysis(
    merged: pd.DataFrame,
    features: Sequence[str],
    *,
    n_boot: int,
    rng: np.random.Generator,
    figures: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    association_rows = []
    tertile_rows = []
    data = merged.copy()
    data["abs_observed_r30"] = data["observed_r30"].abs()
    controls = ["abs_observed_r30", "target_variance", "spatial_valid_fraction"]

    for feature in features:
        if feature not in data.columns:
            raise KeyError(
                f"Event feature '{feature}' is absent from metadata. Add it to an enriched event table."
            )
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
        available = data.dropna(subset=[feature])
        if len(available) < 3:
            continue
        binned = add_event_tertiles(available, feature)
        group_column = f"{feature}_tertile"
        summary = grouped_bootstrap_summary(
            binned.dropna(subset=[group_column]),
            group_column,
            ["observed_r30", "mae_gain", "mse_skill"],
            n_boot=n_boot,
            rng=rng,
        ).rename(columns={group_column: "group"})
        summary["feature"] = feature
        medians = (
            binned.groupby(group_column, observed=True)[feature]
            .median()
            .rename("feature_median")
        )
        summary = summary.merge(medians, left_on="group", right_index=True, how="left")
        tertile_rows.append(summary)
        plot_event_tertiles(summary, feature, figures / f"event_{feature}_tertiles.png")

        association_rows.append(
            standardized_bootstrap_beta(
                available,
                outcome="observed_r30",
                feature=feature,
                controls=(),
                n_boot=n_boot,
                rng=rng,
            )
        )
        association_rows.append(
            standardized_bootstrap_beta(
                available,
                outcome="mae_gain",
                feature=feature,
                controls=controls,
                n_boot=n_boot,
                rng=rng,
            )
        )
    associations = pd.DataFrame(association_rows)
    tertiles = pd.concat(tertile_rows, ignore_index=True) if tertile_rows else pd.DataFrame()
    return associations, tertiles


def run_conditional(args: argparse.Namespace) -> None:
    root, tables, figures = _directories(args.output_dir)
    rng = np.random.default_rng(args.seed)
    _, cubes, steps, landcover, landcover_steps = _analyze(
        args, include_landcover=not args.skip_landcover
    )
    if set(cubes["stage"]) != {"oof"}:
        raise ValueError("Conditional analysis uses selected-model OOF predictions.")
    if cubes["configuration"].nunique() != 1:
        raise ValueError("Provide only the selected CV configuration.")
    merged = _merge_metadata(cubes, args.metadata_csv, args.metadata_id_column)
    environment = _environment_summary(
        merged,
        landcover,
        n_boot=args.bootstrap,
        rng=rng,
        min_group_cubes=args.min_group_cubes,
    )
    features = list(args.event_feature or [])
    if not features and "tp_rollingmax" in merged.columns:
        features = ["tp_rollingmax"]
    associations, tertiles = _event_analysis(
        merged,
        features,
        n_boot=args.bootstrap,
        rng=rng,
        figures=figures,
    )

    merged.to_csv(tables / "oof_cube_response_and_skill.csv", index=False)
    steps.to_csv(tables / "oof_cube_step_metrics.csv", index=False)
    landcover.to_csv(tables / "oof_landcover_metrics.csv", index=False)
    landcover_steps.to_csv(tables / "oof_landcover_step_metrics.csv", index=False)
    environment.to_csv(tables / "oof_environment_summary.csv", index=False)
    associations.to_csv(tables / "oof_event_associations.csv", index=False)
    tertiles.to_csv(tables / "oof_event_tertiles.csv", index=False)
    plot_environment_groups(
        environment[environment["included_in_main_plot"]]
        if not environment.empty
        else environment,
        figures / "oof_environment_skill.png",
    )
    _write_metadata(
        root,
        "conditional",
        args,
        {
            "configuration": str(cubes["configuration"].iloc[0]),
            "n_oof_cubes": int(cubes["cube_id"].nunique()),
            "event_features": features,
            "interpretation": "Cross-validated exploratory analysis; not an independent holdout test.",
        },
    )
    print(f"Conditional analysis written to {root}")


def _add_shared_analysis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Prediction manifest CSV. Repeat for multiple CV configurations.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--min-valid-target-coverage", type=float, default=0.15)
    parser.add_argument("--min-valid-target-count", type=int, default=1000)
    parser.add_argument("--min-target-variance", type=float, default=1e-6)
    parser.add_argument(
        "--response-threshold",
        type=float,
        default=0.005,
        help="Absolute mean kNDVI change treated as stable for direction accuracy.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cv = subparsers.add_parser("cv", help="Compare and select CV configurations.")
    _add_shared_analysis_args(cv)
    cv.set_defaults(func=run_cv)

    holdout = subparsers.add_parser(
        "holdout", help="Confirm the final refit on independent holdout cubes."
    )
    _add_shared_analysis_args(holdout)
    holdout.set_defaults(func=run_holdout)

    conditional = subparsers.add_parser(
        "conditional",
        help="Analyze climate, land cover and event conditioning on selected-model OOF predictions.",
    )
    _add_shared_analysis_args(conditional)
    conditional.add_argument("--metadata-csv", required=True)
    conditional.add_argument("--metadata-id-column", default="DisNo.")
    conditional.add_argument(
        "--event-feature",
        action="append",
        default=None,
        help="Numeric event feature in metadata; repeat for multiple features.",
    )
    conditional.add_argument("--min-group-cubes", type=int, default=10)
    conditional.add_argument("--min-landcover-pixels", type=int, default=100)
    conditional.add_argument(
        "--skip-landcover",
        action="store_true",
        help="Skip loading ESA_LC from source cubes.",
    )
    conditional.set_defaults(func=run_conditional)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    args.func(args)


if __name__ == "__main__":
    main()
