import xarray as xr
import numpy as np
import pandas as pd
from typing import Optional, List
import pyproj
import folium
from IPython.display import display


def subset_era5_spatial(
    era5_cube: xr.Dataset, ds_target: xr.Dataset, plot_check: bool = False
) -> Optional[xr.Dataset]:
    """
    Spatially subsets the ERA5 cube to the geographic bounding box of the target
    dataset (ds_target) and optionally plots the spatial alignment check.

    Parameters:
        era5_cube (xr.Dataset): The source ERA5 dataset (WGS 84).
        ds_target (xr.Dataset): The target S2 dataset (UTM, contains 'x' and 'y').
        plot_check (bool): If True, generates and displays a Folium map showing
                           the S2 bounding box and ERA5 pixel centers.

    Returns:
        xr.Dataset: The spatially subsetted ERA5 dataset, or None on failure.
    """

    try:
        # 1. Define CRSs and Transformer (UTM -> WGS 84)
        crs_ds = f"EPSG:{ds_target.attrs['epsg']}"
        utm_crs = pyproj.CRS(crs_ds)
        wgs84_crs = pyproj.CRS("EPSG:4326")

        transformer = pyproj.Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)

        # 2. Get Corner Coordinates (UTM in meters)
        min_x = ds_target.x.min().item()
        max_x = ds_target.x.max().item()
        min_y = ds_target.y.min().item()
        max_y = ds_target.y.max().item()

        # 3. Convert the corner points to Geographic (lon, lat)
        lon_min, lat_min = transformer.transform(min_x, min_y)
        lon_max, lat_max = transformer.transform(max_x, max_y)

        # 4. Define the final Geographic Bounds (WGS 84)
        box_min_lon = min(lon_min, lon_max)
        box_max_lon = max(lon_min, lon_max)
        box_min_lat = min(lat_min, lat_max)
        box_max_lat = max(lat_min, lat_max)

        # 5. Spatially Subset the ERA5 cube
        # Latitude must be sliced from max (North) to min (South) as coordinates usually run that way.
        era5_subset = era5_cube.sel(
            longitude=slice(box_min_lon, box_max_lon),
            latitude=slice(box_max_lat, box_min_lat),
        )

        # 6. Optional Plotting Check
        m = None
        if plot_check:
            # Extract coordinates of the centers of the ERA5 pixels
            era5_lats = era5_subset.latitude.values
            era5_lons = era5_subset.longitude.values

            # Create Folium Map
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

            # Add SENTINEL Bounding Box Rectangle (Red)
            folium.Rectangle(
                bounds=[(lat_min, lon_min), (lat_max, lon_max)],
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.1,
            ).add_to(m)

            # Add Individual ERA5 Pixel Centers (Blue Markers)
            for lat in era5_lats:
                for lon in era5_lons:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=4,
                        color="blue",
                        fill=True,
                        fill_color="blue",
                        fill_opacity=0.6,
                        tooltip="ERA5 Pixel Center",
                    ).add_to(m)

            display(m)

        return era5_subset, m

    except KeyError as e:
        print(f"Error: Missing required attribute or coordinate: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during spatial subsetting: {e}")
        return None


def subset_era5_time(
    era5_cube: xr.Dataset, ds_target: xr.Dataset, time_dim: str = "time_sentinel_2_l2a"
) -> Optional[xr.Dataset]:
    if time_dim not in ds_target.coords:
        print(f"Error: Time dimension '{time_dim}' not found in target dataset.")
        return None

    # 1. Define target period
    s2_start = ds_target[time_dim].min().values.astype("datetime64[D]")
    s2_end = ds_target[time_dim].max().values.astype("datetime64[D]")

    start_period = s2_start - np.timedelta64(5, "D")
    end_period = s2_end

    # 2. Created expected timeline (daily)
    target_time_range = np.arange(
        start_period, end_period + np.timedelta64(1, "D"), np.timedelta64(1, "D")
    )

    try:
        # 3. Use reindex to match target dimension
        era5_subset = era5_cube.reindex(Ti=target_time_range, method=None)

        # 3.1 Set chunks
        era5_subset = era5_subset.chunk({"Ti": -1})

        # 4. Sanity check
        if len(era5_subset.Ti) != len(target_time_range):
            raise ValueError(
                f"Subset length mismatch. Expected {len(target_time_range)}, got {len(era5_subset.Ti)}"
            )

        assert era5_subset.Ti[0] == ds_target.time_sentinel_2_l2a[0] - np.timedelta64(
            5, "D"
        )
        assert era5_subset.Ti[-1] == ds_target.time_sentinel_2_l2a[-1]

        return era5_subset
    except Exception as e:
        print(
            f"Error during temporal subsetting for {era5_cube} and {ds_target.attrs.get('cube_id')}: {e}"
        )
        return None


def aggregate_era5_metrics_new(
    era5_data: xr.Dataset,
    ds_target: xr.Dataset,
    era5_var_names: List[str],
    era5_time_dim: str = "Ti",
    target_time_dim: str = "time_sentinel_2_l2a",
) -> Optional[xr.Dataset]:

    # Get start date as base for the binning
    # start_date = pd.to_datetime(ds_target[target_time_dim].min().values)
    start_date = pd.to_datetime(ds_target.precip_end_date)

    # 1. ERA5 resampled on the same grid as our target dimension
    resampler = era5_data[era5_var_names].resample(
        {era5_time_dim: "5D"},
        label="right",
        origin=start_date
        - pd.Timedelta(
            days=5
        ),  # so for the first s2 observation the climate info of the 5 previous days will be taken
    )

    # 2.Calculate statistics
    ds_mean = resampler.mean().rename({v: f"{v}_mean" for v in era5_var_names})
    ds_min = resampler.min().rename({v: f"{v}_min" for v in era5_var_names})
    ds_max = resampler.max().rename({v: f"{v}_max" for v in era5_var_names})

    # 3. Merge
    era5_res = xr.merge([ds_mean, ds_min, ds_max])

    # Rename dimensions and restrict to target dim
    era5_res = era5_res.rename({era5_time_dim: target_time_dim})

    ds_target[target_time_dim] = ds_target[target_time_dim].dt.floor("D")
    era5_res[target_time_dim] = era5_res[target_time_dim].dt.floor("D")

    era5_aligned = era5_res.reindex(
        {target_time_dim: ds_target[target_time_dim]}, method=None
    )

    return era5_aligned


def check_time_alignment(ds_target, ds_era5, time_dim):
    target_times = ds_target[time_dim].values
    era5_times = ds_era5[time_dim].values

    # Prüfen ob die Länge stimmt
    if len(target_times) != len(era5_times):
        print(
            f"❌ Zeitachsen-Länge unterscheidet sich: S2={len(target_times)}, ERA5={len(era5_times)}"
        )
        return

    # Prüfen ob die Werte identisch sind
    diff = np.sum(target_times != era5_times)
    if diff == 0:
        print("✅ Zeitstempel sind zu 100% identisch.")
    else:
        print(f"❌ {diff} Zeitstempel weichen voneinander ab!")


# def aggregate_era5_metrics(
#     era5_data: xr.Dataset,
#     ds_target: xr.Dataset,
#     era5_var_names: List[str],
#     era5_time_dim: str = "Ti",
#     target_time_dim: str = "time_sentinel_2_l2a",
# ) -> Optional[xr.Dataset]:
#     """
#     Aggregates specified ERA5 variables across intervals
#     defined by the target dataset's time steps. Calculates mean, min, and max
#     for each variable within each interval.

#     Parameters:
#         era5_data (xr.Dataset): Spatially subsetted ERA5 data (daily resolution).
#         ds_target (xr.Dataset): The target S2 dataset defining the time bounds (ds_train).
#         era5_var_names (List[str]): The ERA5 variables to aggregate (e.g., ['pei_30', 'pei_90']).
#         era5_time_dim (str): The name of the time dimension in era5_data.
#         target_time_dim (str): The name of the time dimension in ds_target.

#     Returns:
#         xr.Dataset: The time-aligned dataset containing all aggregated features,
#                     or None on error.
#     """

#     # Check for missing input variables
#     if not all(var in era5_data for var in era5_var_names):
#         missing = [var for var in era5_var_names if var not in era5_data]
#         print(f"Error: Missing ERA5 variables in input data: {missing}")
#         return None

#     s2_times = ds_target[target_time_dim].values

#     # 1. Define the custom time intervals (bins)
#     # The first interval begins 5 days before the first s2 observation
#     bin_start_times = s2_times - np.timedelta64(5, "D")
#     bin_end_times = s2_times

#     # List to hold the aggregated dataset for each S2 timestep
#     all_timesteps_data = []

#     # 2. Loop through each Sentinel-2 timestep
#     for i in range(len(s2_times)):
#         start_time = bin_start_times[i]
#         end_time = bin_end_times[i]

#         # Select the daily data for the current interval
#         interval_data = era5_data[era5_var_names].sel(
#             {era5_time_dim: slice(start_time, end_time)}
#         )

#         if interval_data[era5_time_dim].size == 0:
#             # Fallback if an interval is empty (e.g., S2 dates too close)
#             # Use the nearest single daily observation to avoid NaNs
#             print(f"No era5 data found between {start_time} and {end_time}")
#             interval_data = era5_data[era5_var_names].sel(
#                 {era5_time_dim: end_time}, method="nearest"
#             )  # Check here again

#         # 2. Vectorized calculation for ALL variables at once
#         # Instead of looping through var_names, xarray can do them all in one go
#         ds_mean = interval_data.mean(dim=era5_time_dim).rename(
#             {v: f"{v}_mean" for v in era5_var_names}
#         )
#         # ds_min = interval_data.min(dim=era5_time_dim).rename({v: f"{v}_min" for v in era5_var_names})
#         # ds_max = interval_data.max(dim=era5_time_dim).rename({v: f"{v}_max" for v in era5_var_names})

#         # Merge metrics for this timestep
#         # current_step = xr.merge([ds_mean, ds_min, ds_max])
#         current_step = ds_mean

#         # Add metadata and time coordinate
#         current_step[target_time_dim] = end_time
#         current_step = current_step.set_coords(target_time_dim)
#         current_step = current_step.expand_dims(target_time_dim)

#         all_timesteps_data.append(current_step)

#     # 3. Final Concatenation
#     era5_aligned = xr.concat(all_timesteps_data, dim=target_time_dim)

#     # Merge event metadata back from ds_target
#     # This assumes ds_target has these as coordinates or 1D variables
#     event_vars = [
#         "precip_start_date",
#         "precip_end_date",
#         "drought_start_date",
#         "drought_end_date",
#     ]
#     for v in event_vars:
#         if v in ds_target.attrs:
#             era5_aligned.attrs[v] = ds_target.attrs[v]
#         else:
#             print(f"{v} not in target")

#     return era5_aligned


def create_uniform_era5_features(
    ds_target: xr.Dataset,
    era5_aligned: xr.Dataset,
    target_time_dim: str = "time_sentinel_2_l2a",
) -> Optional[xr.Dataset]:
    """
    Creates a new Dataset containing uniform (block-filled) ERA5 features
    by broadcasting all variables from era5_aligned across the create_uniform_era5_features
    target Sentinel spatial grid (y, x).

    Parameters:
        ds_target (xr.Dataset): The target S2 dataset defining the target grid and time coordinate (ds_train).
        era5_aligned (xr.Dataset): The time-aggregated ERA5 data

    Returns:
        xr.Dataset: The final spatially uniform (time, y, x) ERA5 feature dataset, or None.
    """

    # 1. Define all variables to be broadcasted dynamically
    era5_vars = [i for i in era5_aligned.data_vars]

    # 2. Get list of vars that have time_sentinel_2_l2a as time dim
    valid_vars = [
        v for v in ds_target.data_vars if target_time_dim in ds_target[v].dims
    ]

    template_var = valid_vars[0]
    template_da = ds_target[template_var]

    # 2. Prepare the target templates and coordinates
    # We use a known S2 band (e.g., 'B04' or 'y') as the full 3D template for shape inference.
    try:
        y_coord = ds_target["y"]
        x_coord = ds_target["x"]
    except KeyError as e:
        print(f"Error: Missing coordinate {e} in ds_target.")
        return None

    # 3. Broadcast and assign coordinates for each variable
    new_data_vars = {}

    for var_name in era5_vars:
        # a. Get the 1D time series (e.g., pei_30_mean)
        time_series = era5_aligned[var_name]

        # b. Broadcast the 1D time series across the spatial dimensions (y, x)
        # This creates the (time, y, x) structure where y/x are uniform blocks
        broadcasted_array = time_series.broadcast_like(template_da)

        # c. Create a new DataArray with the correct UTM coordinates
        # The broadcast operation creates the shape, but we need to assign the UTM coordinates.
        new_da = broadcasted_array.assign_coords(y=y_coord, x=x_coord).astype("float32")

        new_data_vars[var_name] = new_da

    # 4. Create the final, clean Dataset
    era5_final_ds = xr.Dataset(new_data_vars)

    # 5. Final assignment and cleanup

    # Assign the correct time coordinate values
    era5_final_ds = era5_final_ds.assign_coords(
        {
            target_time_dim: ds_target[target_time_dim],
            "y": ds_target["y"],
            "x": ds_target["x"],
        }
    )

    # Drop any temporary/old geographic coordinates that might have been carried over
    era5_final_ds = era5_final_ds.drop_vars(["latitude", "longitude"], errors="ignore")

    # Remove any dimensions that now have a size of 1 (like old latitude/longitude dimensions)
    era5_final_ds = era5_final_ds.squeeze(drop=True)

    return era5_final_ds


def verify_era5_alignment(
    ds_target: xr.Dataset,
    era5_final: xr.Dataset,
    target_time_dim: str = "time_sentinel_2_l2a",
) -> bool:
    """
    Performs comprehensive structural, coordinate, and value uniformity checks
    between the target S2 dataset (ds_target) and the aligned ERA5 feature dataset.

    Parameters:
        ds_target (xr.Dataset): The target S2 dataset (e.g., ds_train).
        era5_aligned_features (xr.Dataset): The final, broadcasted ERA5 features (e.g., pei_final).
        target_time_dim (str): The name of the common time dimension.

    Returns:
        bool: True if all critical tests pass, False otherwise.
    """
    print("--- Starting ERA5 Alignment Verification ---")
    all_tests_passed = True

    # Identify all ERA5 variables to check
    era5_vars_to_check = [v for v in era5_final.data_vars]

    # --- 1. Dimensionality and Shape Checks ---
    print("\n[1] Dimensionality Checks:")

    # We use a known target variable for comparison (e.g., 'y' coordinate dimension)
    target_y_size = ds_target.sizes.get("y", -1)
    target_x_size = ds_target.sizes.get("x", -1)
    target_time_size = ds_target.sizes.get(target_time_dim, -1)

    # Check 1.1: Time Dimension Size
    era5_time_size = era5_final.sizes.get(target_time_dim, 0)
    if era5_time_size != target_time_size:
        print(
            f"❌ FAIL: Time dimension size mismatch. S2={target_time_size}, ERA5={era5_time_size}"
        )
        all_tests_passed = False
    else:
        print(f"✅ PASS: Time dimension size ({target_time_size}) matches.")

    # Check 1.2: Spatial Dimension Size (Check one ERA5 variable)
    if era5_vars_to_check:
        v = era5_final[era5_vars_to_check[0]]
        if v.sizes.get("y") != target_y_size or v.sizes.get("x") != target_x_size:
            print(
                f"❌ FAIL: Spatial shape mismatch. S2=({target_y_size}, {target_x_size}), ERA5=({v.sizes.get('y')}, {v.sizes.get('x')})"
            )
            all_tests_passed = False
        else:
            print(f"✅ PASS: Spatial shape ({target_y_size}x{target_x_size}) matches.")

    # --- 2. Coordinate Alignment Checks ---
    print("\n[2] Coordinate Value Checks (UTM):")

    # Check 2.1: Time Coordinate Alignment (Crucial for merging)
    try:
        time_coords_match = np.all(
            ds_target[target_time_dim].values == era5_final[target_time_dim].values
        )
        if not time_coords_match:
            print(
                "❌ FAIL: Time coordinate values DO NOT match. Merging will fail or misalignment."
            )
            all_tests_passed = False
        else:
            print("✅ PASS: Time coordinate values are identical.")
    except Exception as e:
        print(f"❌ FAIL: Time coordinate check failed: {e}")
        all_tests_passed = False

    # Check 2.2: Spatial Coordinate Alignment (Must be identical UTM values)
    try:
        x_coords_match = np.allclose(ds_target["x"].values, era5_final["x"].values)
        y_coords_match = np.allclose(ds_target["y"].values, era5_final["y"].values)
        if not x_coords_match or not y_coords_match:
            print("❌ FAIL: Spatial coordinate values (x or y) DO NOT match.")
            all_tests_passed = False
        else:
            print("✅ PASS: Spatial coordinate values (x, y) are identical.")
    except Exception as e:
        print(f"❌ FAIL: Spatial coordinate check failed: {e}")
        all_tests_passed = False

    # --- 3. Value Uniformity Check (For ERA5 Variables) ---
    print("\n[3] Spatial Uniformity Check (Block Filling):")

    # Check that every ERA5 variable is uniform across space (y, x) for every timestep
    for var_name in era5_vars_to_check:
        # Select the data and compute (needed for lazy dask arrays)
        data_var = era5_final[var_name].compute()

        # Check the first timestep (representative sample)
        sample_array = data_var.isel({target_time_dim: 0}).values

        # Find unique non-NaN values
        valid_values = sample_array[~np.isnan(sample_array)]

        # A successful block-fill means there is only 1 unique value (the single ERA5 value)
        is_uniform = len(np.unique(valid_values)) <= 1

        if not is_uniform:
            print(
                f"❌ FAIL: Variable '{var_name}' is NOT spatially uniform (found {len(np.unique(valid_values))} unique values)."
            )
            all_tests_passed = False
        else:
            uniform_val = valid_values[0] if len(valid_values) > 0 else np.nan
            print(f"✅ PASS: Variable '{var_name}' is uniform. Value={uniform_val:.4f}")

    print("\n--- Verification Summary ---")
    if all_tests_passed:
        print(
            "🟢 SUCCESS: All critical alignment and uniformity checks passed. Ready to merge."
        )
    else:
        print("🔴 FAILURE: One or more critical checks failed. DO NOT merge.")

    return all_tests_passed
