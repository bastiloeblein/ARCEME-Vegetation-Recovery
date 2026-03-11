import numpy as np
import xarray as xr
from typing import List, Dict


def normalize_dem(ds: xr.Dataset, max_earth_height: float = 8849.0) -> xr.Dataset:
    """Scales COP_DEM to [0, 1] based on a physical maximum."""
    assert "COP_DEM" in ds.data_vars, "COP_DEM not found in dataset variables."

    normalized = ds["COP_DEM"] / max_earth_height

    # Assert physical plausibility
    dem_max = float(normalized.max())
    assert (
        dem_max <= 1.0
    ), f"DEM normalization failed: Max is {dem_max}, expected <= 1.0"
    assert (
        normalized.min() >= -0.05
    ), f"DEM has unexpected negative values: {normalized.min().values}"

    ds["COP_DEM"] = normalized.astype("float32")
    return ds


def calculate_global_era5_stats(
    cubes: Dict[str, xr.Dataset], vars_to_norm: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculates Z-score parameters for ERA5 variables across all cubes.
    Expects squeezed 1D scalar time series (time dimension only).
    """
    stats = {}

    for var in vars_to_norm:
        all_time_values = []

        for key, ds in cubes.items():
            if var not in ds.data_vars:
                continue

            # Verification: Ensure it's 1D (squeezed) before calculating stats
            assert len(ds[var].dims) == 1, f"Variable {var} in cube {key} is not 1D."

            sample = ds[var].values
            valid_sample = sample[~np.isnan(sample)]

            if valid_sample.size > 0:
                all_time_values.append(valid_sample)

        if all_time_values:
            combined_values = np.concatenate(all_time_values)

            # Calculate Z-Score Parameters
            mean_val = np.mean(combined_values)
            std_val = np.std(combined_values)

            # Prevent division by zero
            if std_val == 0:
                std_val = 1.0

            stats[var] = {"mean": float(mean_val), "std": float(std_val)}
            print(f"✅ {var}: Mean={mean_val:.2f}, Std={std_val:.2f}")
        else:
            print(f"⚠️ No data for {var}")

    return stats


def normalize_era5(ds, stats):
    """Applies robust z-score normalization"""
    for ds_var in ds.data_vars:
        matched_stat = None
        for stat_key in stats.keys():
            if ds_var.startswith(stat_key):
                matched_stat = stats[stat_key]
                break

        if matched_stat:
            # Get params from stats
            m = matched_stat["mean"]
            s = matched_stat["std"]

            # Z-Score Standardization
            normalized = (ds[ds_var] - m) / s

            ds[ds_var] = normalized.astype("float32")

    return ds


def final_clipping_instance(ds: xr.Dataset) -> xr.Dataset:
    """Safety net: Clips all variables except ESA_Lc to interval of [-5; 5]."""

    for var in ds.data_vars:
        if var != "ESA_LC":
            ds[var] = ds[var].clip(-5, 5)

    return ds


def check_standardization(ds: xr.Dataset, vars_to_check: List[str]):
    """Prints range and mean for verification of the normalization process."""
    print(f"{'Variable':<25} | {'Min':<8} | {'Max':<8} | {'Mean':<8}")
    print("-" * 60)
    for var in vars_to_check:
        if var not in ds.data_vars:
            continue

        v_min = float(ds[var].min())
        v_max = float(ds[var].max())
        v_mean = float(ds[var].mean())
        print(f"{var:<25} | {v_min:8.4f} | {v_max:8.4f} | {v_mean:8.4f}")
