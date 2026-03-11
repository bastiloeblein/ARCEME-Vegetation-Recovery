from typing import Dict
import numpy as np
import xarray as xr
import spyndex


# -------------------------------------------------------------- CONSTANTS --------------------------------------------------------------------------------------

# SCL: 0=No Data, 1=Saturated/Defective, 2=Dark Area/Shadow
SCL_INVALID = [0, 1, 2]
# SCL: 3=Cloud Shadow, 8=Cloud Medium Prob, 9=Cloud High Prob, 10=Cirrus
SCL_CLOUDS = [3, 8, 9, 10]
# cloud_mask: 1, 2, 3 represent different cloud levels
CM_CLOUDS = [1, 2, 3]

# ESA vegetation classes
VEGETATION_CLASSES = [10, 20, 30, 40, 60, 90, 95, 100]

# Sentinel-2 L2A reflectance values usually range from 0 to 10000 (0-100% reflectance)
REFLECTANCE_MIN = 0
REFLECTANCE_MAX = 10000
NORM_FACTOR = 10000.0

# Expected physical range for NDVI
NDVI_MIN = -1.0
NDVI_MAX = 1.0

# S2 Bandnamen für spyndex
BAND_MAP_S2: Dict[str, str] = {
    "B": "B02_normalized",
    "G": "B03_normalized",
    "R": "B04_normalized",
    "RE1": "B05_normalized",
    "RE2": "B06_normalized",
    "RE3": "B07_normalized",
    "N": "B08_normalized",
    "S1": "B11_normalized",
    "S2": "B12_normalized",
}


# Expected intervals of indices
INDEX_THRESHOLDS = {
    "NDVI": (-1.0, 1.0),
    "kNDVI": (0.0, 1.0),  # kNDVI ist durch tanh und Quadrat immer positiv
    "NDMI": (-1.0, 1.0),
    "NDWI": (-1.0, 1.0),
    "NIRv": (-1.0, 1.0),  # Da NIR und NDVI meist positiv sind in Veg-Zonen
    "IRECI": (-1.0, 2.0),  # IRECI kann leicht über 1 gehen
    "CIRE": (-1.0, 25.0),  # Ratio-Index, kann bei sehr vitalem Chlorophyll hoch sein
}


# ---------------------------------------------------------------------- S2 Functions ---------------------------------------------------------------------------


def get_s2_quality_masks(ds: xr.Dataset) -> xr.Dataset:
    """
    Creates binary physical quality masks and removes source classification bands.
    """

    # Check for hardware issues and general cloud mask noise
    is_hardware_ok = ~ds.SCL.isin(SCL_INVALID)
    is_cm_ok = ~ds.cloud_mask.isin(CM_CLOUDS)

    # Check for clouds using the Scene Classification Layer (SCL)
    is_scl_cloud_free = ~ds.SCL.isin(SCL_CLOUDS)

    # Generate binary masks (1 = Valid, 0 = Masked)
    # Basic: Hardware and Cloud Mask check
    ds["mask_phys_basic"] = (is_hardware_ok & is_cm_ok).astype("uint8")

    # Strict: Basic check + strict SCL cloud filtering
    ds["mask_phys_strict"] = (ds["mask_phys_basic"] & is_scl_cloud_free).astype("uint8")

    # --- SANITY CHECKS ---
    # 1. Binary check: Ensure values are strictly 0 or 1
    unique_vals = np.unique(ds["mask_phys_basic"].values)
    assert set(unique_vals).issubset(
        {0, 1}
    ), f"Basic Mask contains non-binary values: {unique_vals}!"

    # 2. Hierarchy invariant: Strict mask must be a subset of basic mask
    if int(ds["mask_phys_strict"].sum().compute().item()) >= int(
        ds["mask_phys_basic"].sum().compute().item()
    ):
        raise ValueError(
            "Logic Error: Strict mask has more valid pixels than basic mask!"
        )

    # --- CLEANUP ---
    ds = ds.drop_vars(["SCL", "cloud_mask"], errors="ignore")

    return ds


def get_vegetation_mask(ds: xr.Dataset) -> xr.Dataset:
    """
    Creates a binary vegetation mask based on ESA WorldCover classes
    and removes the source land cover band.
    """

    # Create mask: 1 for vegetation classes, 0 for everything else
    # We take the first (and usually only) timestamp of WorldCover
    mask = (
        ds.ESA_LC.isel(time_esa_worldcover=0).isin(VEGETATION_CLASSES).astype("uint8")
    )

    # Assign to dataset
    ds["is_veg"] = mask

    # --- SANITY CHECK ---
    # Ensure the mask is strictly binary (0 or 1)
    unique_vals = np.unique(ds["is_veg"].values)
    assert set(unique_vals).issubset(
        {0, 1}
    ), f"Vegetation mask contains non-binary values: {unique_vals}!"

    return ds


def apply_masking(
    ds, masking_type="strict", ref_band="B04", target_dim="time_sentinel_2_l2a"
):
    """
    Applies a physical cloud/quality mask to Sentinel-2 variables and validates
    the consistency of resulting NaNs.
    """

    # Calculate spatial dimensions for reporting
    total_pixels_per_step = ds.x.size * ds.y.size

    # --- 1. PRE-MASKING: Identify Permanent Sensor Gaps ---
    # Check for pixels that are NaN for all timesteps
    perm_nan_before = ds[ref_band].isnull().all(dim=target_dim)
    count_perm_before = int(perm_nan_before.sum())

    if count_perm_before > 0:
        print("--- 1. Pre-Masking Report (Sensor Gaps) ---")
        print(
            f"Permanent NaN pixels (Sensor errors): {count_perm_before} / {total_pixels_per_step}"
        )
        print(
            f"Percentage of total area:           {count_perm_before/total_pixels_per_step:.2%}"
        )

    # --- 2. APPLY MASK ---
    # Load the specific mask (e.g., mask_phys_strict)
    mask = ds[f"mask_phys_{masking_type}"].compute()

    # Count how many data points are marked for removal (mask value 0)
    total_mask_zeros = int((mask == 0).sum().compute().item())

    # Identify all S2-specific data variables (those sharing the S2 time dimension)
    # Excludes existing mask layers
    s2_vars = [v for v in ds.data_vars if target_dim in ds[v].dims and "mask" not in v]

    # Create a copy to avoid modifying the original dataset in place
    ds_masked = ds.copy()

    for var in s2_vars:
        # Apply the mask: .where(mask == 1) preserves valid data and sets mask == 0 to NaN
        ds_masked[var] = ds_masked[var].where(mask == 1).astype("float32")

        # --- 3. CONSISTENCY CHECK ---
        # Verify that every pixel flagged by the mask (0) is indeed now a NaN
        is_correctly_masked = ds_masked[var].where(mask == 0).isnull().all()

        assert (
            is_correctly_masked
        ), f"Spatial Masking Error: Some pixels in {var} that should be masked (mask=0) still contain numerical values!"

        # Count total NaNs after operation
        current_nans = int(ds_masked[var].isnull().sum().compute().item())

        # Logic: Current NaNs must be at least the number of zeros in the mask
        assert (
            current_nans >= total_mask_zeros
        ), f"Masking failed for {var}! Found fewer NaNs than masked pixels."

    return ds_masked


def report_permanent_nans_for_var(
    ds: xr.Dataset, var_name: str, target_dim: str = "time_sentinel_2_l2a"
):
    """
    Analyzes how many pixels in a specific variable are permanently NaN
    across the entire time series (spatial gaps).
    """
    if var_name not in ds.data_vars:
        print(f"Variable '{var_name}' not found in the dataset.")
        return

    # Logic: Identify which pixels have NO valid data points at all across time.
    # .notnull().any(dim=target_dim) returns True if at least one timestamp has data.
    # The tilde (~) operator inverts this: True now means the pixel is ALWAYS NaN.
    perm_nan_mask = ~ds[var_name].notnull().any(dim=target_dim)

    # Calculate statistics based on the spatial grid (x, y)
    count_perm = int(perm_nan_mask.sum())
    total_px = ds.x.size * ds.y.size
    pct = (count_perm / total_px) * 100

    print(f"--- Spatial Analysis for: {var_name} ---")
    print(f"Permanent NaN pixels (All time): {count_perm} / {total_px}")
    print(f"Percentage of total area:       {pct:.2f}%")

    return perm_nan_mask


def clean_and_normalize_bands(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalizes Sentinel-2 reflectance bands to a 0-1 range and handles
    physical out-of-bounds values by setting them to NaN.
    """

    # 1. Get list of all bands
    all_b_bands = [v for v in ds.data_vars if v.startswith("B") and v[1:].isalnum()]

    for band in all_b_bands:
        # 1. Identify physically valid pixels based on reflectance thresholds
        is_valid = (ds[band] > REFLECTANCE_MIN) & (ds[band] <= REFLECTANCE_MAX)

        # Count pixels that have data but fall outside the physical bounds (e.g., negative values)
        num_invalid = (~is_valid) & ds[band].notnull()
        print(
            f"Band {band}: {num_invalid.sum().values} pixels out of bounds [set to NaN]"
        )

        # 2. Mask invalid values and normalize to [0, 1] range
        ds[f"{band}_normalized"] = (ds[band].where(is_valid) / NORM_FACTOR).astype(
            "float32"
        )

        # 3. Drop the original bands
        ds = ds.drop_vars(band)

    return ds


def calculate_s2_index(ds: xr.Dataset, index_name: str) -> xr.DataArray:
    """
    Calculates spectral indices (e.g., NDVI) and verifies NaN consistency
    against input masks.
    """

    # 1. Select the correct band mapping based on prior cleaning steps
    bands_map = BAND_MAP_S2

    # Build parameters for spyndex computation
    params = {code: ds[band] for code, band in bands_map.items() if band in ds}

    # 2. Handle Special Cases like kNDVI
    if index_name == "kNDVI":
        params["kNN"] = 1.0
        # Kernel computation requires NIR (N) and Red (R)
        params["kNR"] = spyndex.computeKernel(
            kernel="RBF",
            params={
                "a": params["N"],
                "b": params["R"],
                "sigma": 0.5 * (params["N"] + params["R"]),
            },
        )

    # 3. Compute Index
    ds[index_name] = (
        spyndex.computeIndex(index=[index_name], params=params)
        .squeeze()
        .astype("float32")
    )

    # --- SANITY CHECK ---
    validate_index_against_masks(ds, index_name)

    return ds


def validate_index_against_masks(ds, index_var):
    """
    Validates existing values of a spectral index against physical bounds.
    Reports out-of-range values and raises errors for critical anomalies.
    """
    # 1. Retrieve dynamic thresholds (Defaults to [-1, 1] if not specified)
    lower_bound, upper_bound = INDEX_THRESHOLDS.get(index_var, (-1.0, 1.0))

    # 2. Extract min/max values while ignoring NaNs
    v_max = float(ds[index_var].max().values)
    v_min = float(ds[index_var].min().values)

    # Handle cases where the entire data slice is NaN
    if np.isnan(v_max) or np.isnan(v_min):
        print(f"⚠️ {index_var}: Contains only NaNs. Skipping validation.")
        return

    # 3. Validation and Reporting Logic
    if v_max > upper_bound or v_min < lower_bound:
        print(
            f"❌ Range Report {index_var}: [{v_min:.4f}, {v_max:.4f}] "
            f"(Erwartet: [{lower_bound}, {upper_bound}])"
        )

        # Raise error for normalized indices (NDVI/kNDVI) if values are physically impossible
        if index_var in ["NDVI", "kNDVI", "NDWI", "NDMI"] and (
            v_max > 1.0 or v_min < -1.0
        ):
            raise ValueError(f"🚨 Critical data anomaly detected in {index_var}!")
    else:
        print(f"✅ {index_var} within plausible range: [{v_min:.4f}, {v_max:.4f}]")


def filter_static_vegetation_outliers(
    ds, threshold_pct=0.75, time_dim="time_sentinel_2_l2a"
):
    """
    Identifies 'static bad pixels' that frequently show non-vegetation values
    and saves them in a permanent static mask.
    The static mask can then be used to refine vegetation masks.
    """
    # Find clear pixel values
    is_clear = ds["mask_phys_strict"] == 1

    # 2. Define which values indicate non vegetation
    is_non_veg_val = (
        ds["NDVI"] < 0.05
    ) | (  # NDVI < 0.1 usually is bare soil or cloud or water
        ds["NDWI"] > 0.3
    )  # NDWI greater than 0.3 usually open water

    # 3. Find pixels that indicate no vegetation, are not cloud covered and classified as vegetation
    invalid_and_clear = is_non_veg_val & is_clear & (ds.is_veg == 1)

    # 4. Calculate frequency of errors
    clear_days_count = is_clear.sum(dim=time_dim)
    error_days_count = invalid_and_clear.sum(dim=time_dim)

    # Freq: percentage of 'clear' days that showed unlikely vegetation values
    freq = xr.where(clear_days_count > 5, error_days_count / clear_days_count, 0)

    # 4. Create the Static Bad Mask (2D)
    # True where the pixel is out of expected interval for more then 75%
    static_bad_mask = freq > threshold_pct
    mask_name = "static_veg_filter"
    ds[mask_name] = (~static_bad_mask).astype("uint8")

    # --- ASSERTION LOGIC ---
    # We check how many pixels in 'is_veg' are being "hit" by this new static mask
    # valid_before: All pixels currently marked as vegetation
    # pixels_to_remove: Pixels that are veg but now marked as static bad
    valid_before = int(ds.is_veg.sum())
    pixels_to_remove = int((ds.is_veg & static_bad_mask).sum())

    valid_after = int((ds.is_veg & ds[mask_name]).sum())

    # Check if the math adds up
    assert (
        valid_before - valid_after == pixels_to_remove
    ), f"Assertion failed! Before: {valid_before}, After: {valid_after}, Expected Removal: {pixels_to_remove}"

    # 5. Logging
    num_static_bad = int(static_bad_mask.sum())
    if num_static_bad > 0:
        print(
            f"📍 Created {mask_name}: {num_static_bad} pixels identified as permanent outliers."
        )

    return ds


def integrate_veg_and_wrongly_classified_mask(
    ds: xr.Dataset, mask_name="static_veg_filter"
) -> xr.Dataset:
    """
    Updates the vegetation mask by removing misclassified pixels
    and cleans up the temporary static mask.
    """
    static_mask_2d = ds[mask_name]

    # Use bitwise multiplication to keep it uint8 and ensure correct broadcasting
    ds["is_veg"] = (ds["is_veg"] & static_mask_2d).astype("uint8")

    # Cleanup: The static mask's information is now baked into 'is_veg'
    ds = ds.drop_vars(mask_name, errors="ignore")

    return ds


# def final_data_integrity_check(ds, index_name="NDVI"):
#     """
#     Ultimativer Sanity Check vor dem Modell-Training.
#     Prüft auf NaNs, Wertebereiche, Datentypen und Masken-Logik.
#     """
#     print(f"🚀 Starte finalen Integritäts-Check für {index_name}...")

#     # 1. Check: Absolute NaN-Freiheit
#     # Ein einziges NaN kann den Gradienten im Training auf 'NaN' setzen (exploding gradients).
#     nan_count = int(ds[index_name].isnull().sum())
#     assert nan_count == 0, f"❌ CRITICAL: {index_name} enthält {nan_count} NaNs!"

#     # 2. Check: Wertebereich für Vegetation (Intervall [0, 1])
#     # Wir prüfen nur Pixel, die laut deiner Master-Maske als gültige Vegetation gelten.
#     # Wir nutzen ein kleines Delta (1e-6) für Float-Präzision.
#     veg_data = ds[index_name].where(ds.vegetation_mask_final == 1)

#     # Wir ignorieren hier die Nullen der optischen Maske (Wolken)
#     valid_veg_data = veg_data.where(ds.optical_mask_final_strict == 1)

#     max_val = float(valid_veg_data.max())
#     min_val = float(valid_veg_data.min())

#     assert max_val <= 1.000001, f"❌ VALUE ERROR: Max Wert {max_val} in {index_name} übersteigt 1.0!"
#     assert min_val >= -0.000001, f"❌ VALUE ERROR: Min Wert {min_val} in {index_name} ist unter 0.0!"

#     # 3. Check: Datentypen (Wichtig für GPU-Memory)
#     # Masken sollten uint8 sein, Messwerte float32.
#     assert ds[index_name].dtype == 'float32', f"Typ-Fehler: {index_name} sollte float32 sein, ist {ds[index_name].dtype}"
#     assert ds.vegetation_mask_final.dtype == 'uint8', "Typ-Fehler: Masken sollten uint8 sein!"

#     # 4. Check: Räumliche Konsistenz (Static Mask)
#     # Die Vegetation-Maske darf über die Zeit nicht variieren (muss statisch sein).
#     # Wir prüfen, ob die Standardabweichung über die Zeit Null ist.
#     flicker = ds.vegetation_mask_final.std(dim="time_sentinel_2_l2a").sum()
#     assert flicker == 0, "❌ LOGIC ERROR: 'vegetation_mask_final' variiert über die Zeit!"

#     # 5. Check: "Zero-Leckage"
#     # Wenn beide Masken 1 sind, darf der Wert im NDVI nicht 'NaN' gewesen sein (was wir zu 0 gefüllt haben).
#     # Das stellt sicher, dass wir keine 'Fake-Nullen' als echte Daten verkaufen.
#     # Wir nutzen hierfür deine valid_NDVI_mask von vorhin.
#     leakage = ((ds.optical_mask_final_strict == 1) & (ds.vegetation_mask_final == 1) & (ds.valid_NDVI_mask == 0)).sum()
#     assert int(leakage) == 0, f"❌ MASK ERROR: {int(leakage)} ungültige Berechnungen als 'Gültig' markiert!"

#     print("✅ FINALER CHECK BESTANDEN: Der Cube ist bereit für das ConvLSTM.")
#     return True
