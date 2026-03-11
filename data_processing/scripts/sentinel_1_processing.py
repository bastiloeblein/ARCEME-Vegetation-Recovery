import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter
from typing import Dict
import gc
import spyndex

# S1 Bandnamen (VH/VV)
BAND_MAP_S1: Dict[str, str] = {
    "VV": "vv",  #'vv_speckle',
    "VH": "vh",  #'vh_speckle',
}


def find_global_veg_clipping_values(
    cubes, sample_size=5, veg_classes=[10, 20, 30, 40, 60, 90, 95, 100]
):
    """
    Calculates global clipping thresholds (99.5th percentile) for SAR VV and VH bands
    based specifically on pixels classified as vegetation across all provided data cubes.
    """
    all_vv_samples = []
    all_vh_samples = []

    # 1. Randomly select 'sample_size' keys from the dictionary
    # cube_keys = list(cubes.keys())
    # selected_keys = random.sample(cube_keys, min(len(cube_keys), sample_size))

    # print(f"Sampling from {len(selected_keys)} random cubes...")

    for key in cubes.keys():
        ds = cubes[key]

        # 1. Create a vegetation mask based on Land Cover classes
        # Uses the first available Land Cover map in the time series
        veg_mask = ds.ESA_LC.isel(time_esa_worldcover=0).isin(veg_classes)

        # 2. Identify pixels where both SAR bands have valid numerical data
        sar_valid = ds.vv.notnull() & ds.vh.notnull()

        # 3. Combine masks: pixels must be vegetation AND contain valid SAR data
        combined_mask = veg_mask & sar_valid

        # 4. Optional (for better RAM usage)
        # Subsampling directly on dataset, before calling .values
        # Only select every 10. pixel -> saving 90% RAM sampling
        # sample_vv = ds.vv.isel(x=slice(0, None, 10), y=slice(0, None, 10))
        # sample_vh = ds.vh.isel(x=slice(0, None, 10), y=slice(0, None, 10))
        # vv_veg = sample_vv.where(combined_mask).values.flatten()
        # vh_veg = sample_vh.where(combined_mask).values.flatten()

        # 4. Extract valid SAR values and flatten into a 1D array
        # Using .where() to nullify non-veg pixels before flattening
        vv_veg = ds.vv.where(combined_mask).values.flatten()
        vh_veg = ds.vh.where(combined_mask).values.flatten()

        # 5. Remove NaNs (non-vegetation or missing data)
        valid_indices = ~np.isnan(vv_veg)
        vv_veg = vv_veg[valid_indices]
        vh_veg = vh_veg[valid_indices]

        if len(vv_veg) > 0:
            # 6. Randomly sample to manage memory usage while maintaining statistical representation
            # Limits to 500,000 pixels per cube
            indices = np.random.choice(
                len(vv_veg), min(len(vv_veg), 500000), replace=False
            )
            all_vv_samples.append(vv_veg[indices])
            all_vh_samples.append(vh_veg[indices])
        del vv_veg, vh_veg  # , sample_vv, sample_vh
        gc.collect()

    # 7. Compute the global 99.5th percentile across the concatenated pool of all samples
    # This acts as the clipping threshold to remove extreme outliers/noise
    global_vv_max = np.nanpercentile(np.concatenate(all_vv_samples), 99.5)
    global_vh_max = np.nanpercentile(np.concatenate(all_vh_samples), 99.5)

    return global_vv_max, global_vh_max


def clip_s1_data(ds, global_vv_max, global_vh_max):
    # Clip to calculated quantile
    ds["vv"] = ds.vv.clip(min=0, max=global_vv_max)
    ds["vh"] = ds.vh.clip(min=0, max=global_vh_max)

    return ds


def fast_lee_filter_optimized(da, size=7, cu=0.25):
    """
    Applies an optimized Speckle Filter (Lee Filter) to a 3D xarray DataArray.
    Designed for memory efficiency by using float32 and vectorized operations.
    """
    # 1. Cast to float32 to save 50% RAM compared to float64
    img = da.values.astype(np.float32)

    # 2. Create a boolean mask for NaNs
    valid_mask = np.isfinite(img)

    # 3. Replace NaNs with 0 in-place to allow calculation with uniform_filter
    img[~valid_mask] = 0

    # Define 3D kernel (Time=1, Y=size, X=size) - only filters spatially per timestep
    assert da.dims[0] == "time_sentinel_1_rtc", "Expected time dimension first!"
    kernel = (1, size, size)
    s2 = size**2  # Window area

    # 4. Calculate local means using a moving average window
    # mode='constant' + cval=0 handles edges correctly in combination with count_valid
    img_mean_raw = uniform_filter(img, size=kernel, mode="constant", cval=0)
    img_sqr_mean_raw = uniform_filter(img**2, size=kernel, mode="constant", cval=0)

    # 5. Count valid pixels within each window to handle NaNs/edges properly
    count_valid = (
        uniform_filter(
            valid_mask.astype(np.float32), size=kernel, mode="constant", cval=0
        )
        * s2
    )
    count_valid = np.maximum(count_valid, 1e-9)  # Avoid division by zero

    # 6. Normalize local statistics based on valid pixel count
    img_mean = (img_mean_raw * s2) / count_valid
    img_sqr_mean = (img_sqr_mean_raw * s2) / count_valid

    # Free RAM
    del img_mean_raw, img_sqr_mean_raw
    gc.collect()

    # Calculate local variance (In-place to save memory)
    img_var = np.maximum(0, img_sqr_mean - img_mean**2)

    # 7. Calculate Lee weights (Signal-to-Noise ratio logic)
    noise_var = (cu * img_mean) ** 2
    weights = img_var / (img_var + noise_var + 1e-9)
    weights = np.clip(weights, 0, 1)

    # Force weights to 0 in homogeneous areas (simple noise reduction)
    weights[img_var < noise_var] = 0

    # Free RAM
    del img_var, noise_var
    gc.collect()

    # 8. Final reconstruction (Overwriting 'img' to return the result)
    img = img_mean + weights * (img - img_mean)

    # Restore original NaN mask
    img[~valid_mask] = np.nan

    return xr.DataArray(img, coords=da.coords, dims=da.dims, attrs=da.attrs)


def apply_lee_to_ds(ds, bands=["vv", "vh"], win_size=7, cu=0.25):
    """Vectorized application of the Lee Filter to specific dataset bands"""
    # Using a shallow copy to modify the dataset efficiently
    ds_filtered = ds.copy()

    for band in bands:
        print(f"Starting vectorized filtering for band: {band}...")

        # Save number of nans before for assertion check
        no_nans_before = ds_filtered[band].isnull().sum().values

        # Processes all timesteps at once via 3D array broadcasting in fast_lee_filter
        filtered_da = fast_lee_filter_optimized(ds[band], size=win_size, cu=cu)
        ds_filtered[band] = filtered_da.astype("float32")

        # Sanity check: Ensure NaN count remains identical after filtering
        assert ds_filtered[band].isnull().sum().values == no_nans_before

    return ds_filtered


def normalize_s1_vars(ds, vv_max, vh_max, bands=["vv", "vh"]):
    """
    Normalizes SAR bands to a [0, 1] range using global max values.
    Ensures that the NaN mask remains unchanged.
    """
    # Map max values for easy access during the loop
    max_values = {"vv": vv_max, "vh": vh_max}

    for band in bands:
        # 1. Count NaNs before processing for the safety check
        nans_before = int(ds[band].isnull().sum())

        # 2. Perform Min-Max scaling (Min is assumed 0 due to previous clipping)
        # Overwriting the variable directly saves memory
        ds[band] = (ds[band] / max_values[band]).clip(0, 1).astype(np.float32)

        # 3. Validation: Ensure no new NaNs were introduced (e.g., by division errors)
        nans_after = int(ds[band].isnull().sum())
        assert (
            nans_before == nans_after
        ), f"NaN mismatch in {band}! Before: {nans_before}, After: {nans_after}"

        print(f"Normalized {band} using max {max_values[band]}. NaN check passed.")

    return ds


def calculate_SAR_index(ds, index_name, bands_map=BAND_MAP_S1):
    """Calculates a SAR index via spyndex and returns a raw float32 DataArray."""

    # Map internal spyndex codes (e.g., 'VV', 'VH') to your dataset bands
    params = {code: ds[band] for code, band in bands_map.items() if band in ds}

    # Check if all required bands for this specific index are present
    # spyndex.computeIndex returns a DataArray with an 'index' dimension
    raw_da = spyndex.computeIndex(index=[index_name], params=params)

    # .squeeze() removes the 'index' dimension, but we must ensure
    # we don't accidentally squeeze out a valid time or spatial dimension
    # if it only has a size of 1.
    if "index" in raw_da.dims:
        raw_da = raw_da.sel(index=index_name).drop_vars("index")

    return raw_da.astype("float32")


# def aggregate_s1_causal_nearest(
#     ds,
#     s1_vars=["vv", "vh"],
#     s1_dim="time_sentinel_1_rtc",
#     s2_dim="time_sentinel_2_l2a",
#     tolerance_days=12,
# ):
#     """
#     Aligns Sentinel-1 radar data to Sentinel-2 optical timestamps using a causal
#     (past-looking) forward-fill method.
#     """
#     s2_times = ds[s2_dim].values
#     s1_times = ds[s1_dim].values

#     print("NaNs before aggregation:", int(ds[s1_vars[0]].isnull().sum()))

#     # 1. Reindex S1 to S2 time axis using Forward-Fill (ffill)
#     # Causal logic: It looks for the most recent S1 image in the PAST relative to S2.
#     # Tolerance: If the last S1 image is older than 'tolerance_days', it remains NaN.
#     ds_s1_res = (
#         ds[s1_vars]
#         .reindex(
#             {s1_dim: s2_times},
#             method="ffill",
#             tolerance=np.timedelta64(tolerance_days, "D"),
#         )
#         .rename({s1_dim: s2_dim})
#     )

#     print("NaNs after aggregation: ", int(ds_s1_res[s1_vars[0]].isnull().sum()))

#     # --- DATA USAGE ANALYSIS ---

#     # Track which specific S1 timestamps were chosen for each S2 slot
#     s1_time_da = xr.DataArray(s1_times, coords={s1_dim: s1_times}, dims=[s1_dim])
#     chosen_s1_times = s1_time_da.reindex(
#         {s1_dim: s2_times},
#         method="ffill",
#         tolerance=np.timedelta64(tolerance_days, "D"),
#     ).values

#     # 1. Identify "Empty Bins": S2 dates where no S1 image was found within the tolerance
#     empty_bins_mask = np.isnat(chosen_s1_times)
#     empty_s2_dates = s2_times[empty_bins_mask]

#     # 2. Identify "Duplicates": Instances where the same S1 image is assigned to multiple S2 dates
#     # (Happens if S2 frequency > S1 frequency)
#     is_duplicate = np.zeros(len(chosen_s1_times), dtype=bool)
#     for i in range(1, len(chosen_s1_times)):
#         if (
#             not np.isnat(chosen_s1_times[i])
#             and chosen_s1_times[i] == chosen_s1_times[i - 1]
#         ):
#             is_duplicate[i] = True

#     duplicate_s2_dates = s2_times[is_duplicate]

#     # --- OUTPUT STATISTICS ---
#     print("--- S1-S2 Causal Analysis ---")
#     print(f"Total S1 Timesteps: {len(s1_times)}")
#     print(f"Total S2 Timesteps: {len(s2_times)}")
#     print(
#         f"Empty Timesteps (No S1 within {tolerance_days} days): {len(empty_s2_dates)}"
#     )
#     if len(empty_s2_dates) > 0:
#         print(f"  First 5 empty dates: {empty_s2_dates[:5]}")

#     print(f"Re-used S1 images (Duplicates): {len(duplicate_s2_dates)}")
#     if len(duplicate_s2_dates) > 0:
#         print(f"  Examples of S2 dates with duplicates: {duplicate_s2_dates[:5]}")
#     print("----------------------------")

#     # Add a quality mask indicating where S1 data is actually present
#     ds_s1_res["s1_quality_mask"] = ds_s1_res[s1_vars[0]].notnull().astype("uint8")

#     # Create a mask with (dim = time_sentinel_2_l2a): 1 if there is at least one S1 observation present for this timestep and 0 if there isnt
#     ds["s1_any_data_available"] = (ds.s1_quality_mask.sum(dim=["x", "y"]) > 0).astype(
#         "uint8"
#     )

#     # Drop the original S1 dimension to allow merging back into the S2-dimensioned dataset
#     s2_only = ds.drop_dims(s1_dim)

#     return xr.merge([s2_only, ds_s1_res])


# ## This function maybe useful, but probably not
# def apply_sar_quality_mask(
#     ds,
#     threshold_vv=1.0,
#     threshold_vh=0.5,
#     veg_classes=[10, 20, 30, 40, 60, 90, 95, 100],
# ):
#     """
#     Identifiziert Pixel, die jemals physikalisch unplausible SAR-Werte hatten
#     und setzt diese für alle Zeitschritte auf NaN (oder 0).
#     """
#     # 1. Wo liegen die Extremwerte? (Einzelne Ausreißer finden)
#     # Wir nehmen die Originalbänder vor dem Clipping
#     is_bad_vv = ds.vv > threshold_vv
#     is_bad_vh = ds.vh > threshold_vh
#     is_veg = ds.ESA_LC.isel(time_esa_worldcover=0).isin(veg_classes)

#     # 2. 2D-Maske erstellen: Pixel, die IRGENDWANN mal schlecht waren
#     bad_pixel_mask = (is_bad_vv | is_bad_vh) & is_veg

#     # 3. Das Dataset bereinigen
#     # Wir setzen diese Pixel im gesamten Cube auf NaN (oder 0, je nach Wunsch)
#     ds_clean = ds.copy()
#     ds_clean["vv"] = ds.vv.where(~bad_pixel_mask, np.nan)
#     ds_clean["vh"] = ds.vh.where(~bad_pixel_mask, np.nan)

#     # Optional: Auch die bereits geclippten Bänder maskieren
#     if "vv_clipped" in ds:
#         ds_clean["vv_clipped"] = ds.vv_clipped.where(~bad_pixel_mask, np.nan)
#         ds_clean["vh_clipped"] = ds.vh_clipped.where(~bad_pixel_mask, np.nan)

#     return ds_clean, bad_pixel_mask
