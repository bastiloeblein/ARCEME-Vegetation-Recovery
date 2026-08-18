"""I/O, metrics and bootstrap utilities for saved ARCEME prediction cubes."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr


REQUIRED_TENSOR_VARS = ("pred", "true", "mask", "base")
SPATIAL_DIMS = ("y", "x")
LANDCOVER_GROUPS: dict[str, tuple[int, ...]] = {
    "forest": (10, 95),
    "shrub_grass": (20, 30),
    "cropland": (40,),
    "other_vegetation": (60, 90, 100),
}


def load_manifests(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Load and validate one or more prediction manifests."""
    frames = []
    for path in paths:
        manifest_path = Path(path).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        frame = pd.read_csv(manifest_path)
        frame["manifest_path"] = str(manifest_path)
        frames.append(frame)
    if not frames:
        raise ValueError("At least one prediction manifest is required.")

    frame = pd.concat(frames, ignore_index=True)
    required = {
        "configuration",
        "architecture",
        "optical_input",
        "stage",
        "fold",
        "cube_id",
        "tensor_path",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction manifest is missing columns: {sorted(missing)}")
    frame["cube_id"] = frame["cube_id"].astype(str)
    duplicate = frame.duplicated(
        ["configuration", "stage", "cube_id"], keep=False
    )
    if duplicate.any():
        raise ValueError(
            "Duplicate configuration/stage/cube rows in prediction manifests:\n"
            + frame.loc[
                duplicate, ["configuration", "stage", "cube_id", "tensor_path"]
            ].to_string(index=False)
        )
    missing_paths = [
        str(path)
        for path in frame["tensor_path"]
        if not Path(str(path)).expanduser().exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} prediction stores are missing; first: {missing_paths[0]}"
        )
    return frame


def _as_thw(dataset: xr.Dataset, variable: str) -> np.ndarray:
    array = dataset[variable]
    expected = {"time", "y", "x"}
    if set(array.dims) != expected:
        raise ValueError(
            f"{variable} must have exactly dimensions {sorted(expected)}, got {array.dims}"
        )
    return np.asarray(array.transpose("time", "y", "x").values, dtype=np.float32)


def open_prediction_cube(path: str | Path) -> dict[str, np.ndarray]:
    """Load one saved prediction Zarr while keeping memory bounded per cube."""
    tensor_path = Path(path).expanduser()
    with xr.open_zarr(tensor_path, consolidated=None) as dataset:
        missing = set(REQUIRED_TENSOR_VARS).difference(dataset.data_vars)
        if missing:
            raise ValueError(f"{tensor_path} is missing variables: {sorted(missing)}")
        arrays = {name: _as_thw(dataset, name) for name in REQUIRED_TENSOR_VARS}

    shapes = {name: value.shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Tensor shapes differ in {tensor_path}: {shapes}")
    time, height, width = arrays["pred"].shape
    if time < 1 or height < 1 or width < 1:
        raise ValueError(f"Empty prediction tensor: {tensor_path} {arrays['pred'].shape}")

    # The saved baseline is the same last-observation persistence image at all
    # lead times. A changing baseline would invalidate the comparisons below.
    if time > 1 and not np.allclose(
        arrays["base"], arrays["base"][0:1], rtol=0.0, atol=1e-6, equal_nan=True
    ):
        raise ValueError(f"Persistence baseline changes over time in {tensor_path}")
    return arrays


def _static_2d(data_array: xr.DataArray, reducer: str = "first") -> np.ndarray:
    """Reduce a replicated static variable to a y/x array."""
    array = data_array
    for dim in list(array.dims):
        if dim not in SPATIAL_DIMS:
            if reducer == "any":
                array = array.astype(bool).any(dim=dim)
            else:
                array = array.isel({dim: 0})
    if set(array.dims) != set(SPATIAL_DIMS):
        raise ValueError(f"Could not reduce {data_array.name} to y/x; dims={array.dims}")
    return np.asarray(array.transpose("y", "x").values)


def load_source_statics(
    path: str | Path | None,
    spatial_shape: tuple[int, int],
    include_landcover: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load eligible vegetation and optionally WorldCover from a source cube."""
    if path is None or str(path).strip() in {"", "nan", "<NA>"}:
        return None, None
    source_path = Path(str(path)).expanduser()
    if not source_path.exists():
        warnings.warn(
            f"Source cube unavailable; coverage/land-cover diagnostics skipped: {source_path}",
            stacklevel=2,
        )
        return None, None

    with xr.open_zarr(source_path, consolidated=None) as dataset:
        eligible = None
        landcover = None
        if "is_veg" in dataset:
            eligible = _static_2d(dataset["is_veg"], reducer="any").astype(bool)
        if include_landcover:
            if "ESA_LC" not in dataset:
                warnings.warn(f"ESA_LC missing in {source_path}", stacklevel=2)
            else:
                landcover = _static_2d(dataset["ESA_LC"], reducer="first").astype(
                    np.int16
                )

    for name, value in (("is_veg", eligible), ("ESA_LC", landcover)):
        if value is not None and value.shape != spatial_shape:
            raise ValueError(
                f"{name} shape {value.shape} does not match predictions {spatial_shape} "
                f"for {source_path}"
            )
    return eligible, landcover


def _digest_arrays(arrays: Iterable[np.ndarray]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return float("nan")
    return float(numerator / denominator)


def _direction(value: float, threshold: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    if value > threshold:
        return 1.0
    if value < -threshold:
        return -1.0
    return 0.0


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    if np.std(x64) <= 1e-12 or np.std(y64) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x64, y64)[0, 1])


def compute_prediction_metrics(
    arrays: dict[str, np.ndarray],
    *,
    eligible_veg: np.ndarray | None = None,
    min_valid_target_coverage: float = 0.15,
    min_valid_target_count: int = 1000,
    min_target_variance: float = 1e-6,
    response_threshold: float = 0.005,
    calculate_signature: bool = True,
) -> tuple[dict, list[dict]]:
    """Compute cube- and step-level state, baseline and response metrics."""
    pred = arrays["pred"]
    true = arrays["true"]
    mask = arrays["mask"]
    base = arrays["base"]
    time, height, width = pred.shape
    valid = (
        (mask > 0.5)
        & np.isfinite(pred)
        & np.isfinite(true)
        & np.isfinite(base)
    )

    step_rows: list[dict] = []
    total_sq_err = total_abs_err = total_bias = 0.0
    total_sq_err_base = total_abs_err_base = total_bias_base = 0.0
    total_y = total_y_sq = 0.0
    total_valid = 0

    for timestep in range(time):
        current = valid[timestep]
        n_valid = int(np.count_nonzero(current))
        if n_valid == 0:
            step_rows.append(
                {
                    "step": timestep + 1,
                    "lead_day": (timestep + 1) * 5,
                    "n_valid": 0,
                    "spatial_valid_fraction": 0.0,
                    **{
                        key: float("nan")
                        for key in (
                            "mse",
                            "mae",
                            "bias",
                            "mse_base",
                            "mae_base",
                            "bias_base",
                            "mse_skill",
                            "mae_gain",
                            "nse",
                            "nnse",
                            "observed_response",
                            "predicted_response",
                            "response_mean_abs_error",
                            "response_amplitude_ratio",
                            "response_spatial_correlation",
                            "direction_match",
                        )
                    },
                }
            )
            continue

        y_true = np.asarray(true[timestep][current], dtype=np.float64)
        y_pred = np.asarray(pred[timestep][current], dtype=np.float64)
        y_base = np.asarray(base[timestep][current], dtype=np.float64)
        err = y_pred - y_true
        err_base = y_base - y_true
        observed_delta = y_true - y_base
        predicted_delta = y_pred - y_base

        sq_err = float(np.dot(err, err))
        abs_err = float(np.abs(err).sum())
        bias_sum = float(err.sum())
        sq_err_base = float(np.dot(err_base, err_base))
        abs_err_base = float(np.abs(err_base).sum())
        bias_base_sum = float(err_base.sum())
        mse = sq_err / n_valid
        mse_base = sq_err_base / n_valid
        mae = abs_err / n_valid
        mae_base = abs_err_base / n_valid
        y_mean = float(y_true.mean())
        sst = float(np.square(y_true - y_mean).sum())
        nse = 1.0 - sq_err / sst if sst > 1e-12 else float("nan")
        nnse = 1.0 / (2.0 - nse) if np.isfinite(nse) else float("nan")
        observed_response = float(observed_delta.mean())
        predicted_response = float(predicted_delta.mean())
        observed_amplitude = float(np.abs(observed_delta).mean())
        predicted_amplitude = float(np.abs(predicted_delta).mean())

        step_rows.append(
            {
                "step": timestep + 1,
                "lead_day": (timestep + 1) * 5,
                "n_valid": n_valid,
                "spatial_valid_fraction": n_valid / (height * width),
                "mse": mse,
                "mae": mae,
                "bias": bias_sum / n_valid,
                "mse_base": mse_base,
                "mae_base": mae_base,
                "bias_base": bias_base_sum / n_valid,
                "mse_skill": 1.0 - _safe_ratio(mse, mse_base),
                "mae_gain": mae_base - mae,
                "nse": nse,
                "nnse": nnse,
                "observed_response": observed_response,
                "predicted_response": predicted_response,
                "response_mean_abs_error": abs(
                    predicted_response - observed_response
                ),
                "response_amplitude_ratio": _safe_ratio(
                    predicted_amplitude, observed_amplitude
                ),
                "response_spatial_correlation": _correlation(
                    observed_delta, predicted_delta
                ),
                "direction_match": float(
                    _direction(observed_response, response_threshold)
                    == _direction(predicted_response, response_threshold)
                ),
            }
        )

        total_sq_err += sq_err
        total_abs_err += abs_err
        total_bias += bias_sum
        total_sq_err_base += sq_err_base
        total_abs_err_base += abs_err_base
        total_bias_base += bias_base_sum
        total_y += float(y_true.sum())
        total_y_sq += float(np.dot(y_true, y_true))
        total_valid += n_valid

    finite_steps = [row for row in step_rows if row["n_valid"] > 0]
    if total_valid == 0:
        raise ValueError("Prediction cube has no finite, valid target pixels.")

    mse = total_sq_err / total_valid
    mae = total_abs_err / total_valid
    mse_base = total_sq_err_base / total_valid
    mae_base = total_abs_err_base / total_valid
    target_mean = total_y / total_valid
    target_sst = max(0.0, total_y_sq - total_valid * target_mean**2)
    target_variance = target_sst / total_valid
    nse = 1.0 - total_sq_err / target_sst if target_sst > 1e-12 else float("nan")
    nse_base = (
        1.0 - total_sq_err_base / target_sst if target_sst > 1e-12 else float("nan")
    )
    nnse = 1.0 / (2.0 - nse) if np.isfinite(nse) else float("nan")
    nnse_base = 1.0 / (2.0 - nse_base) if np.isfinite(nse_base) else float("nan")

    eligible_pixels = (
        int(np.count_nonzero(eligible_veg)) if eligible_veg is not None else None
    )
    possible_target_pixel_times = (
        time * eligible_pixels if eligible_pixels is not None else None
    )
    valid_target_coverage = (
        total_valid / possible_target_pixel_times
        if possible_target_pixel_times
        else float("nan")
    )
    reasons = []
    if total_valid < min_valid_target_count:
        reasons.append("insufficient_valid_target_count")
    if (
        np.isfinite(valid_target_coverage)
        and valid_target_coverage < min_valid_target_coverage
    ):
        reasons.append("insufficient_valid_target_coverage")
    if target_variance < min_target_variance:
        reasons.append("target_variance_below_minimum")

    observed_trajectory = np.asarray(
        [row["observed_response"] for row in finite_steps], dtype=float
    )
    predicted_trajectory = np.asarray(
        [row["predicted_response"] for row in finite_steps], dtype=float
    )
    final_step = step_rows[-1]
    cube_row = {
        "n_valid_target_pixel_times": total_valid,
        "n_eligible_vegetation_pixels": eligible_pixels,
        "spatial_valid_fraction": total_valid / (time * height * width),
        "valid_target_coverage": valid_target_coverage,
        "mse": mse,
        "mae": mae,
        "bias": total_bias / total_valid,
        "mse_base": mse_base,
        "mae_base": mae_base,
        "bias_base": total_bias_base / total_valid,
        "mse_skill": 1.0 - _safe_ratio(mse, mse_base),
        "mae_gain": mae_base - mae,
        "persistence_beaten": float(mse < mse_base),
        "target_sst": target_sst,
        "target_variance": target_variance,
        "nse": nse if not reasons else float("nan"),
        "nnse": nnse if not reasons else float("nan"),
        "nse_base": nse_base if not reasons else float("nan"),
        "nnse_base": nnse_base if not reasons else float("nan"),
        "nnse_eligible": not reasons,
        "nnse_exclusion_reason": ";".join(reasons),
        "observed_response_auc": float(np.mean(observed_trajectory)),
        "predicted_response_auc": float(np.mean(predicted_trajectory)),
        "observed_r30": float(final_step["observed_response"]),
        "predicted_r30": float(final_step["predicted_response"]),
        "r30_abs_error": float(final_step["response_mean_abs_error"]),
        "r30_direction_match": float(final_step["direction_match"]),
        "response_trajectory_correlation": _correlation(
            observed_trajectory, predicted_trajectory
        ),
        "data_signature": (
            _digest_arrays((true, mask, base)) if calculate_signature else ""
        ),
    }
    return cube_row, step_rows


def _metadata_fields(row: pd.Series) -> dict:
    fields = {
        "configuration": row["configuration"],
        "architecture": row["architecture"],
        "optical_input": row["optical_input"],
        "stage": row["stage"],
        "fold": row.get("fold", pd.NA),
        "cube_id": str(row["cube_id"]),
        "tensor_path": str(row["tensor_path"]),
        "source_cube_path": str(row.get("source_cube_path", "")),
    }
    return fields


def analyze_manifest(
    manifest: pd.DataFrame,
    *,
    include_landcover: bool = False,
    min_landcover_pixels: int = 100,
    min_valid_target_coverage: float = 0.15,
    min_valid_target_count: int = 1000,
    min_target_variance: float = 1e-6,
    response_threshold: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze all manifest rows without holding multiple cubes in memory."""
    cube_rows: list[dict] = []
    step_rows: list[dict] = []
    landcover_cube_rows: list[dict] = []
    landcover_step_rows: list[dict] = []

    for _, row in manifest.iterrows():
        metadata = _metadata_fields(row)
        arrays = open_prediction_cube(row["tensor_path"])
        spatial_shape = arrays["pred"].shape[-2:]
        eligible, landcover = load_source_statics(
            row.get("source_cube_path"),
            spatial_shape,
            include_landcover=include_landcover,
        )
        cube_metrics, steps = compute_prediction_metrics(
            arrays,
            eligible_veg=eligible,
            min_valid_target_coverage=min_valid_target_coverage,
            min_valid_target_count=min_valid_target_count,
            min_target_variance=min_target_variance,
            response_threshold=response_threshold,
        )
        cube_rows.append({**metadata, **cube_metrics})
        step_rows.extend({**metadata, **values} for values in steps)

        if include_landcover and landcover is not None:
            for group, codes in LANDCOVER_GROUPS.items():
                group_mask = np.isin(landcover, codes)
                n_group_pixels = int(np.count_nonzero(group_mask))
                if n_group_pixels < min_landcover_pixels:
                    continue
                group_arrays = dict(arrays)
                group_arrays["mask"] = arrays["mask"] * group_mask[None, :, :]
                try:
                    lc_cube, lc_steps = compute_prediction_metrics(
                        group_arrays,
                        eligible_veg=group_mask,
                        min_valid_target_coverage=0.0,
                        min_valid_target_count=min_valid_target_count,
                        min_target_variance=min_target_variance,
                        response_threshold=response_threshold,
                        calculate_signature=False,
                    )
                except ValueError:
                    continue
                landcover_cube_rows.append(
                    {
                        **metadata,
                        "landcover_group": group,
                        "n_landcover_pixels": n_group_pixels,
                        **lc_cube,
                    }
                )
                landcover_step_rows.extend(
                    {
                        **metadata,
                        "landcover_group": group,
                        "n_landcover_pixels": n_group_pixels,
                        **values,
                    }
                    for values in lc_steps
                )

    return (
        pd.DataFrame(cube_rows),
        pd.DataFrame(step_rows),
        pd.DataFrame(landcover_cube_rows),
        pd.DataFrame(landcover_step_rows),
    )


def percentile_interval(values: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    return float(np.quantile(finite, tail)), float(np.quantile(finite, 1.0 - tail))


def bootstrap_rows(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap a statistic where each row is one independent cube."""
    if frame.empty:
        return float("nan"), float("nan"), float("nan")
    estimate = float(statistic(frame))
    indices = np.arange(len(frame))
    samples = []
    for _ in range(n_boot):
        sampled = rng.choice(indices, size=len(indices), replace=True)
        value = float(statistic(frame.iloc[sampled]))
        if np.isfinite(value):
            samples.append(value)
    low, high = percentile_interval(samples)
    return estimate, low, high


def stratified_bootstrap_rows(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    strata: str,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Resample cubes within folds and keep all folds represented."""
    if frame.empty:
        return float("nan"), float("nan"), float("nan")
    groups = [group.reset_index(drop=True) for _, group in frame.groupby(strata)]
    estimate = float(statistic(frame))
    samples = []
    for _ in range(n_boot):
        pieces = []
        for group in groups:
            indices = rng.choice(len(group), size=len(group), replace=True)
            pieces.append(group.iloc[indices])
        value = float(statistic(pd.concat(pieces, ignore_index=True)))
        if np.isfinite(value):
            samples.append(value)
    low, high = percentile_interval(samples)
    return estimate, low, high


def fold_macro_mean(frame: pd.DataFrame, column: str) -> float:
    """Mean cube metric per fold, then equal-weight mean across folds."""
    values = frame.groupby("fold", dropna=False)[column].mean()
    return float(values.mean())


def ratio_of_means_skill(frame: pd.DataFrame) -> float:
    model = float(frame["mse"].mean())
    baseline = float(frame["mse_base"].mean())
    return 1.0 - _safe_ratio(model, baseline)


def summarize_statistic(
    frame: pd.DataFrame,
    name: str,
    statistic: Callable[[pd.DataFrame], float],
    *,
    n_boot: int,
    rng: np.random.Generator,
    stratify_by_fold: bool = False,
) -> dict:
    bootstrap = stratified_bootstrap_rows if stratify_by_fold else bootstrap_rows
    kwargs = {"strata": "fold"} if stratify_by_fold else {}
    estimate, low, high = bootstrap(
        frame, statistic, n_boot=n_boot, rng=rng, **kwargs
    )
    return {
        "metric": name,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "n_cubes": int(frame["cube_id"].nunique()),
    }


def grouped_bootstrap_summary(
    frame: pd.DataFrame,
    group_column: str,
    value_columns: Sequence[str],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for group, data in frame.groupby(group_column, dropna=False):
        for value in value_columns:
            clean = data[np.isfinite(pd.to_numeric(data[value], errors="coerce"))]
            if clean.empty:
                continue
            estimate, low, high = bootstrap_rows(
                clean,
                lambda sample, column=value: float(sample[column].mean()),
                n_boot=n_boot,
                rng=rng,
            )
            rows.append(
                {
                    group_column: group,
                    "metric": value,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_cubes": int(clean["cube_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def add_event_tertiles(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Add robust rank-based low/middle/high bins for visualization."""
    output = frame.copy()
    values = pd.to_numeric(output[feature], errors="coerce")
    valid = values.notna()
    output[f"{feature}_tertile"] = pd.NA
    if valid.sum() >= 3:
        ranked = values.loc[valid].rank(method="average")
        labels = ["low", "middle", "high"]
        output.loc[valid, f"{feature}_tertile"] = pd.qcut(
            ranked, q=3, labels=labels
        ).astype(str)
    return output


def standardized_bootstrap_beta(
    frame: pd.DataFrame,
    *,
    outcome: str,
    feature: str,
    controls: Sequence[str] = (),
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Return an exploratory standardized OLS coefficient with cube bootstrap CI."""
    columns = [outcome, feature, *controls]
    data = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    # Drop constant controls, but the focal feature/outcome must vary.
    controls = [column for column in controls if data[column].std(ddof=0) > 1e-12]
    columns = [outcome, feature, *controls]
    data = data[columns]
    minimum = max(10, len(columns) + 3)
    if len(data) < minimum or data[outcome].std(ddof=0) <= 1e-12 or data[
        feature
    ].std(ddof=0) <= 1e-12:
        return {
            "outcome": outcome,
            "feature": feature,
            "controls": ";".join(controls),
            "standardized_beta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_cubes": int(len(data)),
        }

    def fit(sample: pd.DataFrame) -> float:
        standardized = (sample - sample.mean()) / sample.std(ddof=0).replace(0, np.nan)
        standardized = standardized.dropna()
        if len(standardized) < minimum:
            return float("nan")
        y = standardized[outcome].to_numpy(dtype=float)
        x = standardized[[feature, *controls]].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        return float(beta[1])

    estimate = fit(data)
    indices = np.arange(len(data))
    samples = []
    for _ in range(n_boot):
        sampled = data.iloc[rng.choice(indices, size=len(indices), replace=True)]
        value = fit(sampled)
        if np.isfinite(value):
            samples.append(value)
    low, high = percentile_interval(samples)
    return {
        "outcome": outcome,
        "feature": feature,
        "controls": ";".join(controls),
        "standardized_beta": estimate,
        "ci_low": low,
        "ci_high": high,
        "n_cubes": int(len(data)),
    }


def read_recommended_refit_epochs(manifest: pd.DataFrame, configuration: str) -> int | None:
    paths = manifest.loc[
        manifest["configuration"] == configuration, "cv_summary_path"
    ].dropna()
    for path in paths.astype(str).unique():
        summary_path = Path(path)
        if summary_path.is_dir():
            summary_path = summary_path / "cv_summary.json"
        if summary_path.exists() and summary_path.name == "cv_summary.json":
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            value = summary.get("recommended_final_refit_epochs")
            if isinstance(value, int) and value > 0:
                return value
    return None
