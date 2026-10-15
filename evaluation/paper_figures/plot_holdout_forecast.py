"""Plot observed RGB, kNDVI forecasts and signed error for one holdout cube."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.scripts.plot_helpers_new import plot_rgb


DEFAULT_RUN = "FinalRefit_SGEDConvLSTM_RGBI_kNDVI_2026-10-07_19-31-23"
KNDVI_CMAP = "YlGn"
ERROR_CMAP = "RdBu_r"
INVALID_COLOR = "#d9d9d9"
TIME_DIM = "time_sentinel_2_l2a"
RGB_BANDS = ("B04", "B03", "B02")


def select_cubes(metrics_path: Path, top_n: int) -> dict:
    with metrics_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No cube metrics found in {metrics_path}")

    id_column = "id" if "id" in rows[0] else "cube_id"
    candidates = []
    seen = set()
    for row in rows:
        cube_id = row[id_column]
        if cube_id in seen:
            raise ValueError(f"Duplicate cube ID: {cube_id}")
        seen.add(cube_id)
        if row.get("reconstruction_complete", "True").lower() != "true":
            continue
        count = int(row["n_valid_target_pixel_times"])
        if count > 0:
            candidates.append({**row, "id": cube_id, "valid_count": count})
    if not candidates:
        raise ValueError("No complete cube with valid target observations found.")

    # Selection depends on observation availability, not prediction performance.
    return sorted(candidates, key=lambda row: (-row["valid_count"], row["id"]))[:top_n]


def load_tensors(path: Path, expected_count: int) -> tuple[np.ndarray, ...]:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing saved predictions: {path}")
    arrays = {}
    with xr.open_zarr(path, consolidated=None, chunks=None) as dataset:
        for name in ("true", "pred", "mask"):
            if name not in dataset:
                raise ValueError(f"Missing variable '{name}' in {path}")
            if set(dataset[name].dims) != {"time", "y", "x"}:
                raise ValueError(
                    f"Unexpected dimensions for {name}: {dataset[name].dims}"
                )
            arrays[name] = np.asarray(
                dataset[name].transpose("time", "y", "x").values, dtype=np.float32
            )

    observed, predicted, mask = (arrays[name] for name in ("true", "pred", "mask"))
    if observed.shape != predicted.shape or observed.shape != mask.shape:
        raise ValueError("Prediction, target and mask shapes do not match.")
    if observed.shape[0] != 6 or min(observed.shape[1:]) < 1:
        raise ValueError(
            f"Expected six non-empty forecast images, got {observed.shape}"
        )

    # The saved evaluation mask already combines vegetation and target validity.
    valid = mask > 0.5
    if np.count_nonzero(valid) != expected_count:
        raise ValueError(
            "Saved mask and CSV valid-observation count differ; check the run."
        )
    if (
        not np.isfinite(observed[valid]).all()
        or not np.isfinite(predicted[valid]).all()
    ):
        raise ValueError("Non-finite values found at valid target locations.")
    return observed, predicted, valid


def colorbar_extend(values: np.ndarray, lower: float, upper: float) -> str:
    below = bool(np.any(values < lower))
    above = bool(np.any(values > upper))
    return (
        "both" if below and above else "min" if below else "max" if above else "neither"
    )


def source_cube_path(run_dir: Path, cube_id: str, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    with (run_dir / "config_used.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    test_dir = Path(config["data"]["test_data_dir"]).expanduser()
    if not test_dir.is_absolute():
        test_dir = REPO_ROOT / test_dir
    matches = sorted(p for p in test_dir.glob(f"*{cube_id}*.zarr") if p.is_dir())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one source cube for {cube_id} in {test_dir}, found {len(matches)}. "
            "Use --source-cube to specify its path."
        )
    return matches[0]


def load_rgb_target(path: Path, observed: np.ndarray, valid: np.ndarray) -> xr.Dataset:
    with xr.open_zarr(path, consolidated=None, chunks=None) as dataset:
        required = {*RGB_BANDS, "kNDVI"}
        missing = required.difference(dataset.data_vars)
        if missing:
            raise ValueError(f"Source cube is missing variables: {sorted(missing)}")
        times = pd.DatetimeIndex(dataset[TIME_DIM].values)
        cutoff = pd.Timestamp(dataset.attrs["precip_end_date"])
        if (
            not times.is_monotonic_increasing
            or not times.is_unique
            or cutoff not in times
        ):
            raise ValueError(
                "Source time axis does not match the model's cutoff convention."
            )
        # Match ARCEME_Dataset: the first six timestamps strictly after the cutoff.
        positions = np.flatnonzero(times > cutoff)[: observed.shape[0]]
        if len(positions) != observed.shape[0]:
            raise ValueError("Source cube does not contain all forecast steps.")
        target = (
            dataset[list(RGB_BANDS) + ["kNDVI"]]
            .isel({TIME_DIM: positions})
            .transpose(TIME_DIM, "y", "x")
            .load()
        )
    source_target = np.asarray(target["kNDVI"].values)
    if source_target.shape != observed.shape or not np.allclose(
        source_target[valid], observed[valid], rtol=1e-5, atol=1e-6
    ):
        raise ValueError(
            "Source target kNDVI does not match the saved GT; check the source cube."
        )
    return target[list(RGB_BANDS)]


def load_context(path: Path, steps: int) -> xr.Dataset:
    with xr.open_zarr(path, consolidated=None, chunks=None) as dataset:
        times = pd.DatetimeIndex(dataset[TIME_DIM].values)
        cutoff = pd.Timestamp(dataset.attrs["precip_end_date"])
        positions = np.flatnonzero(times <= cutoff)
        if steps <= 0 or steps > len(positions):
            raise ValueError(
                f"Requested {steps} context steps; available: {len(positions)}."
            )
        context = (
            dataset[list(RGB_BANDS) + ["kNDVI", "target_mask", "is_veg"]]
            .isel({TIME_DIM: positions[-steps:]})
            .load()
        )
    context["kNDVI"] = context["kNDVI"].where(
        (context["target_mask"] > 0.5)
        & (context["is_veg"] > 0.5)
        & np.isfinite(context["kNDVI"])
    )
    return context


def make_figure(
    observed: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
    cube_id: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    error_limit: float = 0.15,
    rgb_target: xr.Dataset | None = None,
    context: xr.Dataset | None = None,
) -> plt.Figure:
    if (
        not np.isfinite([vmin, vmax, error_limit]).all()
        or vmax <= vmin
        or error_limit <= 0
    ):
        raise ValueError("Require finite vmin < vmax and a positive error limit.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    kndvi_cmap = plt.get_cmap(KNDVI_CMAP).with_extremes(bad=INVALID_COLOR)
    error_cmap = plt.get_cmap(ERROR_CMAP).with_extremes(bad=INVALID_COLOR)
    kndvi_norm = Normalize(vmin=vmin, vmax=vmax)
    error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
    error = predicted - observed

    row_offset = int(rgb_target is not None)
    context_steps = 0 if context is None else context.sizes[TIME_DIM]
    if context_steps and rgb_target is None:
        raise ValueError("Context display requires the RGB target data.")
    fig = plt.figure(
        figsize=(14 + 2 * context_steps, 9.5 if row_offset else 7.5), facecolor="white"
    )
    grid = fig.add_gridspec(
        3 + row_offset,
        7 + context_steps,
        width_ratios=[1] * (6 + context_steps) + [0.055],
        left=0.105,
        right=0.935,
        bottom=0.11,
        top=0.88,
        wspace=0.06,
        hspace=0.09,
    )
    if rgb_target is not None:
        rgb_display = (
            xr.concat([context[list(RGB_BANDS)], rgb_target], dim=TIME_DIM)
            if context_steps
            else rgb_target
        )
        for step in range(context_steps + 6):
            ax = fig.add_subplot(grid[0, step])
            ax.set_facecolor(INVALID_COLOR)
            rgb_figure = plot_rgb(rgb_display, step, ax=ax, stretch_pct=(2, 98))
            if rgb_figure is not fig or not ax.images:
                if rgb_figure is not fig:
                    plt.close(rgb_figure)
                plt.close(fig)
                raise ValueError(f"plot_rgb failed for forecast step {step + 1}")
            ax.images[-1].set_interpolation("nearest")
            ax.images[-1].set_rasterized(True)
            ax.set_axis_on()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            date = pd.Timestamp(rgb_display[TIME_DIM].values[step]).strftime("%Y-%m-%d")
            title = (
                f"Context {step - context_steps + 1}"
                if step < context_steps
                else f"Step {step - context_steps + 1}"
            )
            ax.set_title(f"{title}\n{date}", pad=8, fontsize=10)
            if step == 0:
                ax.set_ylabel(
                    "Observed\nRGB", rotation=0, ha="right", va="center", labelpad=12
                )

    images = []
    row_labels = ["Observed\nkNDVI", "Predicted\nkNDVI", "Prediction\nerror"]
    if context_steps:
        context_values = context["kNDVI"].transpose(TIME_DIM, "y", "x").values
        for row in range(3):
            for step in range(context_steps):
                ax = fig.add_subplot(grid[row + row_offset, step])
                if row == 0:
                    ax.imshow(
                        np.ma.masked_invalid(context_values[step]),
                        cmap=kndvi_cmap,
                        norm=kndvi_norm,
                        origin="upper",
                        interpolation="nearest",
                        rasterized=True,
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if step == 0:
                    ax.set_ylabel(
                        row_labels[row],
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=12,
                    )
        left = grid[0, context_steps - 1].get_position(fig).x1
        right = grid[0, context_steps].get_position(fig).x0
        boundary = (left + right) / 2
        fig.add_artist(
            Line2D(
                [boundary, boundary],
                [0.11, 0.88],
                transform=fig.transFigure,
                color="red",
                linewidth=1.3,
            )
        )

    for row, values in enumerate((observed, predicted, error)):
        for step in range(6):
            ax = fig.add_subplot(grid[row + row_offset, step + context_steps])
            image = ax.imshow(
                np.ma.masked_where(
                    ~np.isfinite(values[step]) if row == 1 else ~valid[step],
                    values[step],
                ),
                cmap=error_cmap if row == 2 else kndvi_cmap,
                norm=error_norm if row == 2 else kndvi_norm,
                origin="upper",
                interpolation="nearest",
                rasterized=True,
            )
            images.append(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0 and not row_offset:
                ax.set_title(f"Step {step + 1}", pad=8, fontsize=11)
            if step == 0 and not context_steps:
                ax.set_ylabel(
                    row_labels[row], rotation=0, ha="right", va="center", labelpad=12
                )

    valid_values = np.concatenate((observed[valid], predicted[np.isfinite(predicted)]))
    if context_steps:
        valid_values = np.concatenate(
            (valid_values, context_values[np.isfinite(context_values)])
        )
    kndvi_bar = fig.colorbar(
        images[0],
        cax=fig.add_subplot(grid[row_offset : row_offset + 2, -1]),
        extend=colorbar_extend(valid_values, vmin, vmax),
    )
    kndvi_bar.set_label("kNDVI")
    kndvi_bar.outline.set_visible(False)
    error_bar = fig.colorbar(
        images[12],
        cax=fig.add_subplot(grid[2 + row_offset, -1]),
        extend=colorbar_extend(error[valid], -error_limit, error_limit),
        ticks=[-error_limit, 0, error_limit],
    )
    error_bar.set_label("Predicted − observed")
    error_bar.outline.set_visible(False)
    fig.suptitle(f"Cube {cube_id}", x=0.105, y=0.97, ha="left", fontsize=13)
    footer = (
        "RGB: observed scene, with per-image contrast stretching. Dates label five-day target bins.\n"
        "Observed kNDVI/error: valid vegetated targets only. Predictions: all finite values. "
        "Blue error = underestimation; red = overestimation."
        if row_offset
        else "Six five-day forecast steps. Grey: invalid target or non-vegetated pixel. "
        "Error: blue = underestimation; red = overestimation."
    )
    if context_steps:
        footer = footer.replace("five-day target bins", "five-day bins").replace(
            "valid vegetated targets only", "valid vegetated observations only"
        )
    fig.text(
        0.105,
        0.025,
        footer,
        fontsize=9,
        ha="left",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "model" / "wand_db_logs" / DEFAULT_RUN,
    )
    parser.add_argument(
        "--source-cube",
        type=Path,
        default=None,
        help="Override the source RGB cube path.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=0.8)
    parser.add_argument("--error-limit", type=float, default=0.15)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of cubes ranked by valid target observation count.",
    )
    parser.add_argument(
        "--cube-id", type=str, default=None, help="Plot a specific cube."
    )
    parser.add_argument(
        "--context-steps",
        type=int,
        default=0,
        help="Show the last N context steps to the left of the forecast.",
    )
    args = parser.parse_args()
    if args.context_steps < 0:
        parser.error("--context-steps must be non-negative")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    run_dir = args.run_dir.expanduser().resolve()
    if args.context_steps:
        with (run_dir / "config_used.yaml").open(encoding="utf-8") as handle:
            context_length = int(yaml.safe_load(handle)["data"]["context_length"])
        if args.context_steps > context_length:
            parser.error(
                f"--context-steps cannot exceed the model context length ({context_length})"
            )
    metrics_path = run_dir / "metrics" / "cube_metrics.csv"
    if args.top_n < 1:
        parser.error("--top-n must be positive")

    if args.cube_id:
        selected_cubes = [
            cube
            for cube in select_cubes(metrics_path, sys.maxsize)
            if cube["id"] == args.cube_id
        ]
        if not selected_cubes:
            parser.error(f"Cube not found or has no valid targets: {args.cube_id}")
    else:
        # get top_n cubes with most valid pixels in target
        selected_cubes = select_cubes(metrics_path, args.top_n)

    for rank, selected in enumerate(selected_cubes):
        cube_id = selected["id"]
        print(f"\nProcessing rank {rank}/{len(selected_cubes)}: {cube_id}")
        tensor_path = run_dir / "tensors" / f"{cube_id}.zarr"
        print(
            f"Selected {cube_id}: {selected['valid_count']:,} valid target observations."
        )
        observed, predicted, valid = load_tensors(tensor_path, selected["valid_count"])
        source_path = source_cube_path(run_dir, cube_id, args.source_cube)
        rgb_target = load_rgb_target(source_path, observed, valid)
        context = (
            load_context(source_path, args.context_steps)
            if args.context_steps
            else None
        )
        fig = make_figure(
            observed,
            predicted,
            valid,
            cube_id,
            args.vmin,
            args.vmax,
            args.error_limit,
            rgb_target=rgb_target,
            context=context,
        )

        # output_dir = args.output_dir or REPO_ROOT / "evaluation" / "paper_figures" / "output" / "holdout_anaylsis" /  "plots" / "qualitative"
        output_dir = (
            args.output_dir
            or REPO_ROOT
            / "evaluation"
            / "paper_figures"
            / "final_plots"
            / "holdout_forecast"
        )
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / f"holdout_forecast_{cube_id}"
        try:
            path = stem.with_suffix(".png")
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved {path}")
        finally:
            plt.close(fig)

        metadata = {
            "cube_id": cube_id,
            "selection": (
                "Explicit cube ID"
                if args.cube_id
                else "Ranked by valid target count among completely reconstructed cubes"
            ),
            "metrics_path": str(metrics_path),
            "tensor_path": str(tensor_path),
            "source_cube_path": str(source_path),
            "target_dates": [
                str(pd.Timestamp(value).date()) for value in rgb_target[TIME_DIM].values
            ],
            "context_dates": (
                [str(pd.Timestamp(value).date()) for value in context[TIME_DIM].values]
                if context is not None
                else []
            ),
            "rgb_function": "data_processing.scripts.plot_helpers_new.plot_rgb",
            "rgb_bands": list(RGB_BANDS),
            "rgb_stretch_percentiles": [2, 98],
            "valid_target_observations": int(valid.sum()),
            "valid_observations_per_step": valid.sum(axis=(1, 2)).tolist(),
            "kndvi_color_limits": [args.vmin, args.vmax],
            "error_color_limits": [-args.error_limit, args.error_limit],
            "error_definition": "predicted minus observed kNDVI",
            "masking": (
                "Observed kNDVI and errors: valid vegetated observations only; "
                "predicted kNDVI: all finite values; RGB: available spectral observations"
            ),
            "orientation": "Saved tensor row order; no geographic coordinates assumed",
        }
        stem.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
