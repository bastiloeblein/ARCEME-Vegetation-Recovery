"""Plot RGB and observed/predicted kNDVI changes from a fixed context reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_processing.scripts.plot_helpers_new import plot_rgb

TIME = "time_sentinel_2_l2a"
RGB = ["B04", "B03", "B02"]
GREY = "#d9d9d9"
DEFAULT_RUN = (
    ROOT / "model/wand_db_logs/FinalRefit_SGEDConvLSTM_RGBI_kNDVI_2026-10-07_19-31-23"
)


def ranked_cubes(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        rows = list(
            csv.DictReader(
                handle, delimiter=";" if ";" in sample.splitlines()[0] else ","
            )
        )
    selected, seen = [], set()
    for row in rows:
        cube_id = row.get("id", row.get("cube_id", "")).strip()
        if not cube_id or cube_id in {"MEAN", "TOTAL"}:
            continue
        if cube_id in seen:
            raise ValueError(f"Duplicate cube ID: {cube_id}")
        seen.add(cube_id)
        if row.get("reconstruction_complete", "True").lower() != "true":
            raise ValueError(f"Incomplete reconstruction: {cube_id}")
        count = int(row["n_valid_target_pixel_times"])
        if count > 0:
            selected.append({"id": cube_id, "valid_count": count})
    if not selected:
        raise ValueError("No cubes with valid target observations found.")
    return sorted(selected, key=lambda row: (-row["valid_count"], row["id"]))


def read_predictions(path: Path, expected_count: int) -> dict[str, np.ndarray]:
    with xr.open_zarr(path, consolidated=None, chunks=None) as ds:
        arrays = {}
        for name in ("true", "pred", "mask", "base"):
            arrays[name] = np.asarray(
                ds[name].transpose("time", "y", "x").values, dtype=np.float32
            )
    if len({a.shape for a in arrays.values()}) != 1:
        raise ValueError(f"Tensor shapes differ in {path}")
    valid = arrays["mask"] > 0.5
    if int(valid.sum()) != expected_count:
        raise ValueError(f"Saved mask and CSV valid count differ in {path}")
    for name in ("true", "pred", "base"):
        if not np.isfinite(arrays[name][valid]).all():
            raise ValueError(f"Non-finite {name} at valid targets in {path}")
    if not np.allclose(
        arrays["base"], arrays["base"][:1], atol=1e-6, rtol=0, equal_nan=True
    ):
        raise ValueError(
            "Saved persistence reference changes over the forecast horizon."
        )
    arrays["valid"] = valid
    return arrays


def last_valid_reference(
    values: np.ndarray, quality: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if values.shape != quality.shape or values.ndim != 3:
        raise ValueError(
            "Context kNDVI and target_mask must have matching (time, y, x) shapes."
        )
    reference = np.full(values.shape[1:], np.nan, dtype=np.float32)
    index = np.full(values.shape[1:], -1, dtype=np.int32)
    for step in range(len(values) - 1, -1, -1):
        take = (index < 0) & (quality[step] > 0.5) & np.isfinite(values[step])
        reference[take] = values[step][take]
        index[take] = step
    return reference, index


def load_source(path: Path, arrays: dict, context_length: int) -> tuple:
    steps = arrays["true"].shape[0]
    with xr.open_zarr(path, consolidated=None, chunks=None) as ds:
        times = pd.DatetimeIndex(ds[TIME].values)
        cutoff = pd.Timestamp(ds.attrs["precip_end_date"])
        if (
            not times.is_unique
            or not times.is_monotonic_increasing
            or cutoff not in times
        ):
            raise ValueError(
                "Source timestamps do not match the model's cutoff convention."
            )
        context = np.flatnonzero(times <= cutoff)[-context_length:]
        target = np.flatnonzero(times > cutoff)[:steps]
        if len(context) != context_length or len(target) != steps:
            raise ValueError(
                "Source cube lacks the configured context or forecast horizon."
            )
        ctx = (
            ds[["kNDVI", "target_mask"]]
            .isel({TIME: context})
            .transpose(TIME, "y", "x")
            .load()
        )
        future = (
            ds[RGB + ["kNDVI", "target_mask"]]
            .isel({TIME: target})
            .transpose(TIME, "y", "x")
            .load()
        )
        vegetation = ds["is_veg"]
        if TIME in vegetation.dims:
            vegetation = vegetation.isel({TIME: 0})
        vegetation = np.asarray(vegetation.transpose("y", "x").values)
        if not np.isin(vegetation, [0, 1]).all():
            raise ValueError("Source is_veg must contain only zero and one.")
    valid = arrays["valid"]
    truth = future.kNDVI.values
    if truth.shape != valid.shape or not np.allclose(
        truth[valid], arrays["true"][valid], rtol=1e-5, atol=1e-6
    ):
        raise ValueError("Source kNDVI and saved target do not match.")
    reference, last_index = last_valid_reference(
        ctx.kNDVI.values, ctx.target_mask.values
    )
    found = last_index >= 0
    if not np.allclose(
        reference[found], arrays["base"][0][found], rtol=1e-5, atol=1e-6
    ):
        raise ValueError(
            "Context reference and saved persistence baseline do not match."
        )
    age = np.full(reference.shape, np.nan)
    days = np.asarray((cutoff - times[context]) / pd.Timedelta(days=1))
    age[found] = days[last_index[found]]
    source_valid = (future.target_mask.values > 0.5) & np.isfinite(truth)
    if not np.array_equal(source_valid & (vegetation > 0.5)[None], valid):
        raise ValueError("Source vegetation validity and saved evaluation mask differ.")
    rgb = future[RGB]
    rgb.attrs["vegetation_pixels"] = int(vegetation.sum())
    rgb.attrs["valid_pixels_per_step"] = source_valid.sum(axis=(1, 2)).tolist()
    return rgb, reference, found, age


def make_change_figure(
    arrays: dict,
    rgb: xr.Dataset,
    reference: np.ndarray,
    found: np.ndarray,
    cube_id: str,
    delta_limit: float = 0.2,
    error_limit: float = 0.15,
    show_error: bool = True,
) -> plt.Figure:
    if (
        not np.isfinite([delta_limit, error_limit]).all()
        or min(delta_limit, error_limit) <= 0
    ):
        raise ValueError("Colour limits must be finite and positive.")
    valid = arrays["valid"] & found[None]
    if not valid.any():
        raise ValueError(f"No valid target with a valid context reference: {cube_id}")
    observed = arrays["true"] - reference
    predicted = arrays["pred"] - reference
    error = arrays["pred"] - arrays["true"]
    steps, rows = len(observed), 4 if show_error else 3
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(2.15 * steps + 1.5, 2.15 * rows + 1.1))
    grid = fig.add_gridspec(
        rows,
        steps + 1,
        width_ratios=[1] * steps + [0.055],
        left=0.115,
        right=0.935,
        top=0.84,
        bottom=0.12,
        hspace=0.09,
        wspace=0.06,
    )
    delta_norm = TwoSlopeNorm(vmin=-delta_limit, vcenter=0, vmax=delta_limit)
    error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0, vmax=error_limit)
    delta_cmap = plt.get_cmap("YlGn").with_extremes(bad=GREY)
    error_cmap = plt.get_cmap("RdBu_r").with_extremes(bad=GREY)
    row_values = [None, observed, predicted] + ([error] if show_error else [])
    labels = [
        "Observed\nRGB",
        "Observed ΔkNDVI\n$Y_t-R(p)$",
        "Predicted ΔkNDVI\n$\\hat{Y}_t-R(p)$",
        "Prediction\nerror",
    ]
    scalar_images = {}
    for row, values in enumerate(row_values):
        for step in range(steps):
            ax = fig.add_subplot(grid[row, step])
            ax.set_facecolor(GREY)
            if row == 0:
                result = plot_rgb(rgb, step, ax=ax)
                if result is not fig or not ax.images:
                    if result is not fig:
                        plt.close(result)
                    plt.close(fig)
                    raise ValueError(f"RGB plotting failed for step {step + 1}")
                ax.set_axis_on()
                ax.images[-1].set_rasterized(True)
                ax.images[-1].set_interpolation("nearest")
                date = pd.Timestamp(rgb[TIME].values[step]).strftime("%Y-%m-%d")
                title = f"Step {step + 1}\n{date}"
                if "valid_pixels_per_step" in rgb.attrs:
                    n_valid = rgb.attrs["valid_pixels_per_step"][step]
                    title += f"\nValid: {n_valid:,}\nValid veg.: {int(arrays['valid'][step].sum()):,}"
                ax.set_title(title, pad=8, fontsize=9.5)
            else:
                scalar_images[row] = ax.imshow(
                    # np.ma.masked_where(~valid[step], values[step]),
                    np.ma.masked_where(
                        ~(
                            (found if row == 2 else arrays["valid"][step])
                            & np.isfinite(values[step])
                        ),
                        values[step],
                    ),
                    origin="upper",
                    cmap=error_cmap if row == 3 else delta_cmap,
                    norm=error_norm if row == 3 else delta_norm,
                    interpolation="nearest",
                    rasterized=True,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if step == 0:
                ax.set_ylabel(
                    labels[row], rotation=0, ha="right", va="center", labelpad=12
                )

    def extension(values, limit):
        below, above = np.any(values < -limit), np.any(values > limit)
        return (
            "both"
            if below and above
            else "min" if below else "max" if above else "neither"
        )

    combined = np.concatenate([observed[valid], predicted[valid]])
    bar = fig.colorbar(
        scalar_images[1],
        cax=fig.add_subplot(grid[1:3, -1]),
        extend=extension(combined, delta_limit),
    )
    bar.set_label("Change from last valid context kNDVI")
    bar.outline.set_visible(False)
    if show_error:
        bar = fig.colorbar(
            scalar_images[3],
            cax=fig.add_subplot(grid[3, -1]),
            extend=extension(error[valid], error_limit),
            ticks=[-error_limit, 0, error_limit],
        )
        bar.set_label("Predicted − observed")
        bar.outline.set_visible(False)
    title = f"Cube {cube_id}"
    if "vegetation_pixels" in rgb.attrs:
        title += f" | Vegetation pixels: {rgb.attrs['vegetation_pixels']:,}"
    fig.suptitle(title, x=0.115, y=0.98, ha="left", fontsize=13)
    fig.text(
        0.115,
        0.025,
        "Fixed pixel-wise reference: last valid observation within the context window.\n"
        "Grey in ΔkNDVI/error: invalid vegetated target or missing context reference. "
        "RGB: per-image contrast stretch; dates label target bins.",
        fontsize=8.5,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--cube-id", help="Select one cube instead of the top-N ranking."
    )
    parser.add_argument("--delta-limit", type=float, default=0.2)
    parser.add_argument("--error-limit", type=float, default=0.15)
    parser.add_argument(
        "--no-error", action="store_true", help="Use three rows instead of four."
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--svg", action="store_true")
    args = parser.parse_args()
    if args.top_n < 1 or args.dpi < 1:
        parser.error("--top-n and --dpi must be positive")
    run = args.run_dir.expanduser().resolve()
    config = yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8"))
    source_dir = (args.source_dir or Path(config["data"]["test_data_dir"])).expanduser()
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir
    cubes = ranked_cubes(run / "metrics/cube_metrics.csv")
    if args.cube_id:
        cubes = [row for row in cubes if row["id"] == args.cube_id]
        if not cubes:
            parser.error("--cube-id is not a complete holdout cube with valid targets")
    else:
        cubes = cubes[: args.top_n]

    # output = args.output_dir or run / "plots/qualitative_changes"
    output = (
        args.output_dir
        or ROOT
        / "evaluation"
        / "paper_figures"
        / "final_plots"
        / "holdout_forecast_changes"
    )
    output.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(cubes, 1):
        cube_id = row["id"]
        matches = sorted(source_dir.glob(f"*{cube_id}*.zarr"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one source cube for {cube_id}, found {len(matches)}"
            )
        arrays = read_predictions(
            run / "tensors" / f"{cube_id}.zarr", row["valid_count"]
        )
        if len(arrays["true"]) != int(config["data"]["target_length"]):
            raise ValueError("Saved horizon and config target_length differ.")
        rgb, reference, found, age = load_source(
            matches[0], arrays, int(config["data"]["context_length"])
        )
        print(
            f"[{rank}/{len(cubes)}] {cube_id}: {int((arrays['valid'] & found[None]).sum()):,} usable change observations"
        )
        fig = make_change_figure(
            arrays,
            rgb,
            reference,
            found,
            cube_id,
            args.delta_limit,
            args.error_limit,
            not args.no_error,
        )
        stem = output / f"holdout_changes_{cube_id}"
        try:
            for suffix in ["png", "pdf", "svg"] if args.svg else ["png", "pdf"]:
                fig.savefig(
                    stem.with_suffix(f".{suffix}"),
                    dpi=args.dpi,
                    bbox_inches="tight",
                    facecolor="white",
                )
        finally:
            plt.close(fig)
        used = arrays["valid"] & found[None]
        used_age = age[used.any(axis=0)]
        metadata = {
            "cube_id": cube_id,
            "source_cube": str(matches[0]),
            "reference": "Fixed last valid context observation per pixel; no zero fallback",
            "context_length": int(config["data"]["context_length"]),
            "target_dates": [str(pd.Timestamp(t).date()) for t in rgb[TIME].values],
            "valid_target_observations": int(arrays["valid"].sum()),
            "vegetation_spatial_pixels": rgb.attrs["vegetation_pixels"],
            "valid_pixels_per_step": rgb.attrs["valid_pixels_per_step"],
            "valid_vegetation_pixels_per_step": arrays["valid"]
            .sum(axis=(1, 2))
            .tolist(),
            "valid_change_observations": int(used.sum()),
            "valid_change_observations_per_step": used.sum(axis=(1, 2)).tolist(),
            "target_observations_excluded_missing_reference": int(
                (arrays["valid"] & ~found[None]).sum()
            ),
            "reference_age_days_relative_to_context_cutoff_median": float(
                np.median(used_age)
            ),
            "reference_age_days_relative_to_context_cutoff_max": float(
                np.max(used_age)
            ),
            "delta_limits": [-args.delta_limit, args.delta_limit],
            "error_limits": [-args.error_limit, args.error_limit],
            "selection": "Descending original valid target count; independent of model skill",
        }
        stem.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"Saved {stem}.png")


if __name__ == "__main__":
    main()
