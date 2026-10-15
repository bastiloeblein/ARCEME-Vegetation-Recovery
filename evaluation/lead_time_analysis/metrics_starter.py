"""Small, explicit metric calculations for saved prediction cubes."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

TIME_DIM = "time_sentinel_2_l2a"
METRICS = ["mae", "mse", "rmse", "bias", "nse", "nnse"]
LANDCOVER_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    60: "Bare or sparse vegetation",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def load_cube(tensor_path, source_path, expected_count):
    """Read one cube and check that source data match the saved evaluation."""
    with xr.open_zarr(Path(tensor_path), chunks=None, consolidated=None) as saved:
        arrays = {}
        for name in ["true", "pred", "base", "mask"]:
            if set(saved[name].dims) != {"time", "y", "x"}:
                raise ValueError(
                    f"Unexpected dimensions for {name}: {saved[name].dims}"
                )
            arrays[name] = saved[name].transpose("time", "y", "x").values
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError("Saved target, prediction, baseline and mask shapes differ.")
    valid = arrays["mask"] > 0.5
    if int(valid.sum()) != int(expected_count):
        raise ValueError("Saved mask count differs from cube_metrics.csv.")
    for name in ["true", "pred", "base"]:
        if not np.isfinite(arrays[name][valid]).all():
            raise ValueError(f"Non-finite {name} at valid target locations.")
    if not np.allclose(arrays["base"], arrays["base"][0:1], equal_nan=True):
        raise ValueError("The persistence baseline changes over time.")

    n_steps = valid.shape[0]
    with xr.open_zarr(Path(source_path), chunks=None, consolidated=None) as source:
        dates = pd.DatetimeIndex(source[TIME_DIM].values)
        cutoff = pd.Timestamp(source.attrs["precip_end_date"])
        if (
            not dates.is_unique
            or not dates.is_monotonic_increasing
            or cutoff not in dates
        ):
            raise ValueError("Invalid source dates or context cutoff.")
        indices = np.flatnonzero(dates > cutoff)[:n_steps]
        if len(indices) != n_steps:
            raise ValueError("Not enough source target steps.")
        target = source.isel({TIME_DIM: indices})
        source_true = target.kNDVI.transpose(TIME_DIM, "y", "x").values
        source_valid = target.target_mask.transpose(TIME_DIM, "y", "x").values > 0.5
        statics = {}
        for name in ["is_veg", "ESA_LC"]:
            layer = source[name]
            if TIME_DIM in layer.dims:
                first = layer.isel({TIME_DIM: 0})
                for t in range(1, layer.sizes[TIME_DIM]):
                    if not np.array_equal(
                        first.values, layer.isel({TIME_DIM: t}).values
                    ):
                        raise ValueError(f"{name} is not static.")
                layer = first
            statics[name] = layer.transpose("y", "x").values

    if source_true.shape != valid.shape or source_valid.shape != valid.shape:
        raise ValueError("Source and saved target shapes differ.")
    for name, layer in statics.items():
        if layer.shape != valid.shape[1:] or not np.isfinite(layer).all():
            raise ValueError(f"Invalid {name} spatial layer.")
    if not np.isin(statics["is_veg"], [0, 1]).all():
        raise ValueError("is_veg must be binary.")
    vegetation = statics["is_veg"] > 0.5
    source_mask = source_valid & np.isfinite(source_true) & vegetation[None]
    if not np.array_equal(source_mask, valid):
        raise ValueError("Source and saved evaluation masks differ.")
    if not np.allclose(source_true[valid], arrays["true"][valid], rtol=1e-5, atol=1e-6):
        raise ValueError("Source kNDVI differs from the saved targets.")
    arrays["valid"] = valid
    arrays["vegetation"] = vegetation
    arrays["landcover"] = statics["ESA_LC"]
    arrays["dates"] = dates[indices]
    return arrays


def calculate_metrics(
    observed,
    predicted,
    baseline,
    valid,
    spatial_mask,
    complete,
    rules,
    filter_errors=False,
):
    """Calculate model and persistence metrics on exactly the same observations."""
    selected = valid & spatial_mask[None]
    y = np.asarray(observed[selected], dtype=np.float64)
    n_valid = y.size
    n_spatial = int(spatial_mask.sum())
    n_possible = n_spatial * observed.shape[0]
    coverage = n_valid / n_possible if n_possible else np.nan
    variance = float(np.mean((y - y.mean()) ** 2)) if n_valid else np.nan

    reasons = []
    if not complete:
        reasons.append("incomplete_reconstruction")
    if n_spatial == 0:
        reasons.append("no_vegetation_in_group")
    if n_valid == 0:
        reasons.append("no_valid_targets")
    elif n_valid < rules["min_count"]:
        reasons.append("insufficient_count")
    if n_possible and coverage < rules["min_coverage"]:
        reasons.append("insufficient_coverage")

    # Apply count and coverage to slice errors, but preserve original overall errors.
    eligible_errors = not reasons if filter_errors else bool(n_valid and complete)
    error_exclusion_reason = ";".join(reasons) if not eligible_errors else ""

    if n_valid and (variance <= 0 or variance < rules["min_variance"]):
        reasons.append("insufficient_variance")

    common = {
        "n_valid": n_valid,
        "n_vegetation_pixels": n_spatial,
        "n_possible": n_possible,
        "coverage": coverage,
        "target_variance": variance,
        "reconstruction_complete": complete,
        "eligible_errors": eligible_errors,
        "error_exclusion_reason": error_exclusion_reason,
        "eligible_nnse": not reasons,
        "exclusion_reason": ";".join(reasons),
        "min_count": rules["min_count"],
        "min_coverage": rules["min_coverage"],
        "min_variance": rules["min_variance"],
        "observed_mean": float(y.mean()) if n_valid else np.nan,
        "observed_std": float(np.sqrt(variance)),
    }
    rows = []
    for approach, values in [("Model", predicted), ("Persistence", baseline)]:
        row = {
            **common,
            "approach": approach,
            **dict.fromkeys(METRICS, np.nan),
            **dict.fromkeys([f"{metric}_raw" for metric in METRICS], np.nan),
            "predicted_mean": np.nan,
            "predicted_std": np.nan,
        }
        if n_valid and complete:
            y_hat = np.asarray(values[selected], dtype=np.float64)
            error = y_hat - y
            mse = float(np.mean(error**2))
            row.update(
                mae_raw=float(np.mean(np.abs(error))),
                mse_raw=mse,
                rmse_raw=float(np.sqrt(mse)),
                bias_raw=float(np.mean(error)),
                predicted_mean=float(y_hat.mean()),
                predicted_std=float(y_hat.std(ddof=0)),
            )
            if eligible_errors:
                for metric in ["mae", "mse", "rmse", "bias"]:
                    row[metric] = row[f"{metric}_raw"]
            if variance > 0:
                row["nse_raw"] = 1 - mse / variance
                row["nnse_raw"] = 1 / (2 - row["nse_raw"])
            if not reasons:
                row["nse"] = row["nse_raw"]
                row["nnse"] = row["nnse_raw"]
        rows.append(row)
    return rows


def cube_table(arrays, metadata, complete, cube_rules, slice_rules):
    """One row per cube x period x landcover x approach; step 0 means all steps."""
    vegetation = arrays["vegetation"]
    regions = {"all": vegetation}
    for code in np.unique(arrays["landcover"][vegetation]):
        code = int(code)
        name = LANDCOVER_NAMES.get(code, f"Class {code}")
        regions[f"{code}: {name}"] = vegetation & (arrays["landcover"] == code)
    n_steps = arrays["true"].shape[0]
    records = []
    for landcover, spatial_mask in regions.items():
        for step in range(n_steps + 1):
            window = slice(None) if step == 0 else slice(step - 1, step)
            scope = "overall" if step == 0 else "step"
            primary = step == 0 and landcover == "all"
            rules = cube_rules if primary else slice_rules
            rows = calculate_metrics(
                arrays["true"][window],
                arrays["pred"][window],
                arrays["base"][window],
                arrays["valid"][window],
                spatial_mask,
                complete,
                rules,
                filter_errors=not primary,
            )
            for row in rows:
                records.append(
                    {
                        **metadata,
                        "scope": scope,
                        "step": step,
                        "landcover": landcover,
                        "target_date": (
                            "" if step == 0 else str(arrays["dates"][step - 1].date())
                        ),
                        "policy": (
                            "cube_eligibility"
                            if primary
                            else "exploratory_slice_eligibility"
                        ),
                        **row,
                    }
                )
    return pd.DataFrame(records)


def macro_summary(table):
    """Equal cube weights within each run, split, fold, period and landcover."""
    keys = ["configuration", "split", "fold", "scope", "step", "landcover", "approach"]
    if table.duplicated(keys + ["cube_id"]).any():
        raise ValueError("Duplicate cube records: check dataset selection.")
    grouped = table.groupby(keys, dropna=False, sort=False)
    summary = grouped[METRICS].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_cubes"] = grouped.size()
    summary["n_valid_total"] = grouped.n_valid.sum()
    summary["coverage_mean"] = grouped.coverage.mean()
    return summary.reset_index()
