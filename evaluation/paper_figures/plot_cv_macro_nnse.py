from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


GUIDED_RUNS = [
    {
        "key": "sg_indices",
        "label": "SGConvLSTM\nIndices",
        "patterns": ("CV_Training_SGConvLSTM_Indices_*",),
    },
    {
        "key": "sg_rgbi",
        "label": "SGConvLSTM\nRGB+NIR",
        "patterns": ("CV_Training_SGConvLSTM_RGBI_*",),
    },
    {
        "key": "sged_indices",
        "label": "SGEDConvLSTM\nIndices",
        "patterns": ("CV_Training_SGEDConvLSTM_Indices_*",),
    },
    {
        "key": "sged_rgbi",
        "label": "SGEDConvLSTM\nRGB+NIR",
        "patterns": ("CV_Training_SGEDConvLSTM_RGBI_*",),
    },
]

AUTOREGRESSIVE_RUNS = [
    {
        "key": "sg_kndvi_ar",
        "label": "SGConvLSTM\nAR kNDVI",
        "patterns": ("CV_Training_SGConvLSTM_kNDVI_AR_*",),
    },
    {
        "key": "sged_kndvi_ar",
        "label": "SGEDConvLSTM\nAR kNDVI",
        "patterns": ("CV_Training_SGEDConvLSTM_kNDVI_AR_*",),
    },
]

SELECTED_MODEL_KEY = "sged_rgbi"


Y_LIMITS = (0.54, 0.7)
Y_TICK_STEP = 0.02

EXPECTED_FOLDS = (0, 1, 2)
FOLD_COLORS = ("#0072B2", "#E69F00", "#009E73")
MEAN_COLOR = "#333333"

METRIC_KEYS = {
    "NNSE": "val/grand_mean_macro/NNSE",
    "NSE": "val/grand_mean_macro/R2",  #  NSE is referred to R2 in the code
    "MAE": "val/grand_mean_macro/MAE",
    "MSE": "val/grand_mean_macro/MSE",
    "Bias": "val/grand_mean_macro/Bias",
}

BASE_METRIC_KEYS = {name: f"{key}_base" for name, key in METRIC_KEYS.items()}

ELIGIBLE_KEY = "val/grand_mean_macro/R2_NNSE_eligible_cubes"
EXCLUDED_KEY = "val/grand_mean_macro/R2_NNSE_excluded_cubes"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_runs_dir = repo_root / "model" / "wand_db_logs"
    default_output_dir = Path(__file__).resolve().parent / "final_plots"

    parser = argparse.ArgumentParser(
        description="Plot cross-validation NNSE results from cv_summary.json files."
    )
    parser.add_argument("--runs-dir", type=Path, default=default_runs_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def find_run_dir(runs_dir: Path, patterns: tuple[str, ...]) -> Path:
    for pattern in patterns:
        matches = sorted(path for path in runs_dir.glob(pattern) if path.is_dir())
        if matches:
            return matches[-1]
    raise FileNotFoundError(f"No run directory matched: {', '.join(patterns)}")


def as_finite_float(value: object, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite")
    return number


def optional_metric(metrics: dict, key: str) -> float:
    try:
        return as_finite_float(metrics[key], key)
    except (KeyError, TypeError, ValueError):
        return float("nan")


def load_cv_run(runs_dir: Path, spec: dict) -> dict:
    run_dir = find_run_dir(runs_dir, spec["patterns"])
    summary_path = run_dir / "cv_summary.json"

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    raw_folds = summary["folds"]
    if not isinstance(raw_folds, list):
        raise TypeError("'folds' must be a list")

    folds = {}
    for entry in raw_folds:
        fold = int(entry["fold"])
        metrics = entry["metrics"]

        if fold in folds:
            raise ValueError(f"Fold {fold} occurs more than once")

        nnse = as_finite_float(metrics[METRIC_KEYS["NNSE"]], METRIC_KEYS["NNSE"])
        eligible = int(as_finite_float(metrics[ELIGIBLE_KEY], ELIGIBLE_KEY))
        excluded = int(as_finite_float(metrics[EXCLUDED_KEY], EXCLUDED_KEY))

        if eligible < 0 or excluded < 0:
            raise ValueError("Cube counts must be non-negative")

        model_metrics = {
            name: nnse if name == "NNSE" else optional_metric(metrics, key)
            for name, key in METRIC_KEYS.items()
        }
        baseline_metrics = {
            name: optional_metric(metrics, key)
            for name, key in BASE_METRIC_KEYS.items()
        }

        folds[fold] = {
            "fold": fold,
            "validation_cubes": eligible + excluded,
            "eligible_cubes": eligible,
            "metrics": model_metrics,
            "baseline_metrics": baseline_metrics,
        }

    if tuple(sorted(folds)) != EXPECTED_FOLDS:
        raise ValueError(
            f"Expected folds {EXPECTED_FOLDS}, found {tuple(sorted(folds))}"
        )

    return {
        "key": spec["key"],
        "label": spec["label"],
        "run_dir": run_dir,
        "folds": folds,
    }


def load_available_runs(runs_dir: Path, specs: list[dict]) -> list[dict]:
    loaded = []
    for spec in specs:
        try:
            result = load_cv_run(runs_dir, spec)
            loaded.append(result)
            print(
                f"Loaded  {spec['label'].replace(chr(10), ' ')}: {result['run_dir'].name}"
            )
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            print(f"Skipped {spec['label'].replace(chr(10), ' ')}: {error}")
    return loaded


def persistence_from_run(source: dict) -> dict:
    folds = {}
    for fold, result in source["folds"].items():
        nnse = result["baseline_metrics"]["NNSE"]
        if not math.isfinite(nnse):
            raise ValueError("Persistence NNSE is missing or invalid")

        folds[fold] = {
            **result,
            "metrics": result["baseline_metrics"],
        }

    return {
        "key": "persistence",
        "label": "Persistence\nbaseline",
        "run_dir": source["run_dir"],
        "folds": folds,
    }


def fold_values(result: dict, metric: str = "NNSE") -> np.ndarray:
    return np.asarray(
        [result["folds"][fold]["metrics"][metric] for fold in EXPECTED_FOLDS],
        dtype=float,
    )


def determine_y_limits(results: list[dict], override: tuple[float, float] | None):
    if override is not None:
        if override[0] >= override[1]:
            raise ValueError(
                "The lower y-axis limit must be smaller than the upper limit"
            )
        return override

    values = np.concatenate([fold_values(result) for result in results])
    lower = float(np.min(values))
    upper = float(np.max(values))
    padding = max(0.005, 0.08 * (upper - lower))
    return lower - padding, upper + padding


def draw_grouped_bars(ax, results: list[dict], y_limits: tuple[float, float]):
    x_positions = np.arange(len(results), dtype=float)
    bar_width = 0.19
    offsets = np.asarray((-1.5, -0.5, 0.5, 1.5)) * bar_width
    lower_limit = y_limits[0]

    persistence = next(result for result in results if result["key"] == "persistence")

    series = []

    for fold, color in zip(EXPECTED_FOLDS, FOLD_COLORS):
        values = np.asarray(
            [result["folds"][fold]["metrics"]["NNSE"] for result in results],
            dtype=float,
        )
        persistence_value = persistence["folds"][fold]["metrics"]["NNSE"]
        series.append((values, color, False, persistence_value))

    mean_values = np.asarray(
        [float(np.mean(fold_values(result))) for result in results],
        dtype=float,
    )
    mean_persistence = float(np.mean(fold_values(persistence)))
    series.append((mean_values, MEAN_COLOR, True, mean_persistence))

    for offset, (values, color, is_mean, persistence_value) in zip(offsets, series):
        bars = ax.bar(
            x_positions + offset,
            values - lower_limit,
            width=bar_width,
            bottom=lower_limit,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

        for bar, result, value in zip(bars, results, values):
            if result["key"] == "persistence":
                improvement = 0.0
            else:
                improvement = (
                    100.0 * (value - persistence_value) / abs(persistence_value)
                )

            ax.annotate(
                f"{value:.4f}\n{improvement:+.2f}%",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold" if is_mean else "normal",
                linespacing=0.9,
                clip_on=False,
            )
    ax.set_xticks(x_positions, [result["label"] for result in results])
    ax.set_xlim(-0.55, len(results) - 0.45)
    ax.set_ylabel("Validation NNSE")
    ax.grid(axis="y", color="0.86", linewidth=0.7, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_fold_legend(ax, count_source: dict):
    handles = []

    for fold, color in zip(EXPECTED_FOLDS, FOLD_COLORS):
        result = count_source["folds"][fold]
        handles.append(
            Patch(
                facecolor=color,
                edgecolor="none",
                label=(
                    f"Fold {fold + 1}: n={result['validation_cubes']} "
                    f"({result['eligible_cubes']} eligible)"
                ),
            )
        )

    handles.append(
        Patch(
            facecolor=MEAN_COLOR,
            edgecolor="none",
            label="Mean across folds",
        )
    )

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#BDBDBD",
        framealpha=0.92,
        fontsize=8.5,
        borderaxespad=0.0,
        handlelength=1.6,
        labelspacing=0.4,
    )


def write_fold_table(results: list[tuple[str, dict]], output_path: Path):
    fields = [
        "group",
        "model",
        "source_run",
        "fold",
        "validation_cubes",
        "nnse_eligible_cubes",
        "macro_NNSE",
        "macro_NSE",
        "macro_MAE",
        "macro_MSE",
        "macro_Bias",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for group, result in results:
            for fold in EXPECTED_FOLDS:
                fold_result = result["folds"][fold]
                metrics = fold_result["metrics"]
                writer.writerow(
                    {
                        "group": group,
                        "model": result["label"].replace("\n", " "),
                        "source_run": result["run_dir"].name,
                        "fold": fold + 1,
                        "validation_cubes": fold_result["validation_cubes"],
                        "nnse_eligible_cubes": fold_result["eligible_cubes"],
                        **{f"macro_{name}": metrics[name] for name in METRIC_KEYS},
                    }
                )


def write_summary_table(results: list[tuple[str, dict]], output_path: Path):
    fields = ["group", "model", "source_run", "n_folds"]
    for metric in METRIC_KEYS:
        fields.extend((f"mean_macro_{metric}", f"sd_macro_{metric}"))

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for group, result in results:
            row = {
                "group": group,
                "model": result["label"].replace("\n", " "),
                "source_run": result["run_dir"].name,
                "n_folds": len(EXPECTED_FOLDS),
            }
            for metric in METRIC_KEYS:
                values = fold_values(result, metric)
                finite = values[np.isfinite(values)]
                row[f"mean_macro_{metric}"] = (
                    float(np.mean(finite)) if finite.size else ""
                )
                row[f"sd_macro_{metric}"] = (
                    float(np.std(finite, ddof=1)) if finite.size > 1 else ""
                )
            writer.writerow(row)


def main():
    args = parse_args()
    runs_dir = args.runs_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if (args.ymin is None) != (args.ymax is None):
        raise ValueError("Use --ymin and --ymax together")
    y_override = (args.ymin, args.ymax) if args.ymin is not None else Y_LIMITS

    guided_results = load_available_runs(runs_dir, GUIDED_RUNS)
    if not guided_results:
        raise RuntimeError(f"No complete guided-model summaries found in {runs_dir}")

    selected = next(
        (result for result in guided_results if result["key"] == SELECTED_MODEL_KEY),
        None,
    )
    if selected is None:
        selected = guided_results[0]
        print(
            f"Selected model unavailable; using {selected['label'].replace(chr(10), ' ')}"
        )

    reference_results = []
    reference_results = [persistence_from_run(selected)]

    ar_results = load_available_runs(runs_dir, AUTOREGRESSIVE_RUNS)
    reference_results.extend(ar_results)
    print("Ref: ", len(reference_results))
    print("AR : ", len(ar_results))
    print("guided: ", len(guided_results))

    plot_results = reference_results + guided_results
    print("PLOT : ", len(plot_results))

    count_profiles = {
        tuple(
            (
                result["folds"][fold]["validation_cubes"],
                result["folds"][fold]["eligible_cubes"],
            )
            for fold in EXPECTED_FOLDS
        )
        for result in guided_results + ar_results
    }
    if len(count_profiles) > 1:
        print("Warning: validation or eligibility counts differ between guided runs")

    y_limits = determine_y_limits(plot_results, y_override)

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, ax = plt.subplots(figsize=(13.5, 5.4), constrained_layout=True)

    draw_grouped_bars(ax, plot_results, y_limits)
    add_fold_legend(ax, selected)

    ax.set_ylim(*y_limits)
    if Y_TICK_STEP is not None:
        first_tick = math.ceil(y_limits[0] / Y_TICK_STEP) * Y_TICK_STEP
        ticks = np.arange(first_tick, y_limits[1] + Y_TICK_STEP / 2, Y_TICK_STEP)
        ax.set_yticks(ticks)

    figure.savefig(
        output_dir / "cv_macro_nnse_comparison.png", dpi=300, bbox_inches="tight"
    )

    unique_results = [("guided", result) for result in guided_results]
    unique_results.extend(
        ("reference", result)
        for result in reference_results
        if result["key"] not in {item["key"] for item in guided_results}
    )
    write_fold_table(unique_results, output_dir / "cv_fold_metrics.csv")
    write_summary_table(unique_results, output_dir / "cv_metric_summary.csv")

    print(f"Saved figure and tables to {output_dir}")

    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
