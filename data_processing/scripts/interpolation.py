import xarray as xr
import pandas as pd
from typing import List


def trim_to_first_s2_acquisition(ds, masking_type="strict"):
    """
    Entfernt leere Timesteps am Anfang des Datasets,
    bis die erste valide S2-Akquisition (laut Maske) erscheint.
    """
    mask_name = f"s2_final_mask_{masking_type}"

    # 1. Wir berechnen den Max-Wert pro Timestep NUR EINMAL
    # und laden das Ergebnis (nur 146 Werte) sofort in den RAM (.compute())
    print("Checking data availability per timestep...")
    has_data = ds[mask_name].max(dim=["x", "y"]).compute()

    # 2. Jetzt arbeiten wir auf einem winzigen Array im RAM
    # argmax findet den ersten Index, der 1 ist
    first_valid_idx = int(has_data.argmax())

    # Check falls alles 0 ist
    if has_data[first_valid_idx] == 0:
        print("⚠️ Warning: No S2 data found in the entire cube!")
        return ds

    if first_valid_idx > 0:
        print(f"✂️ Trimming: {first_valid_idx} empty timesteps removed.")
        ds = ds.isel(time_sentinel_2_l2a=slice(first_valid_idx, None))

    vars_to_drop = [mask_name, "s1_final_mask"]
    ds = ds.drop_vars([v for v in vars_to_drop if v in ds.data_vars])

    return ds


def interpolate_context_only(
    ds: xr.Dataset, vars_to_interpolate: List[str], max_gap_days: int = 30
) -> xr.Dataset:
    """
    Interpolates gaps only within the context period (up to precip_end_date).
    Target data remains untouched.
    """
    time_dim = "time_sentinel_2_l2a"
    limit_bins = max_gap_days // 5

    # 1. Get the cutoff date from attributes
    cutoff_date = pd.to_datetime(ds.attrs["precip_end_date"])

    # 2. Split the dataset into Context and Target
    context_part = ds.sel(
        {time_dim: slice(None, cutoff_date)}
    )  # all timesteps <= cutoff_date will be in context
    target_part = ds.sel(
        {time_dim: slice(cutoff_date + pd.Timedelta(seconds=1), None)}
    )  # all timesteps > cutoff date in target

    print(f"--- Interpolating Context (until {cutoff_date.date()}) ---")

    # 3. Interpolate variables in the Context part
    interpolated_vars = {}
    for var in vars_to_interpolate:

        var_data = context_part[var].chunk({time_dim: -1})

        interpolated_vars[var] = var_data.interpolate_na(
            dim=time_dim,
            method="linear",
            limit=limit_bins,  # if more than 5 timesteps in a row not available then will not be interpolated, also no extrapolation as inteprolation needs two value to interpolate inbetween
            use_coordinate=True,
        )

    # Now update with interpolated vars
    context_part = context_part.assign(interpolated_vars)

    # 4. Merge back together
    # combine_by_coords maintains the temporal order
    ds_final = xr.concat([context_part, target_part], dim=time_dim)

    # Ensure attributes are preserved after concat
    ds_final.attrs = ds.attrs

    return ds_final


def create_final_binary_masks(ds: xr.Dataset) -> xr.Dataset:
    """
    Creates final uint8 masks for S1 and S2 data.
    1 = Valid data (Original or Interpolated)
    0 = Gap (Gap too large, leading NaNs, or persistent missing data)
    """
    print("--- Creating Final Binary Masks & Validating ---")

    # 1. Count NaNs before masking (Basis for the integrity check)
    expected_s2_nans = int(ds["kNDVI"].isnull().sum().compute())
    expected_s1_union_nans = int(
        (ds["vv"].isnull() | ds["vh"].isnull()).sum().compute()
    )

    # 2. Create Masks (notnull() returns True/1 for data, False/0 for NaN)
    ds["mask_s2"] = ds["kNDVI"].notnull().astype("uint8")
    ds["mask_s1"] = (ds["vv"].notnull() & ds["vh"].notnull()).astype("uint8")

    # The target_mask is used for the Loss function during training.
    # It is identical to the S2 mask since kNDVI is our target.
    ds["target_mask"] = ds["mask_s2"]

    # 3. Validation via Assert
    # We check if the number of zeros in the new masks matches the previous NaN count
    actual_s2_zeros = int((ds["mask_s2"] == 0).sum().compute())
    actual_s1_zeros = int((ds["mask_s1"] == 0).sum().compute())

    assert (
        expected_s2_nans == actual_s2_zeros
    ), f"S2 Mask mismatch! NaNs: {expected_s2_nans}, Mask-Zeros: {actual_s2_zeros}"
    assert (
        expected_s1_union_nans == actual_s1_zeros
    ), f"S1 Mask mismatch! NaNs: {expected_s1_union_nans}, Mask-Zeros: {actual_s1_zeros}"

    print(f"✅ S2 Mask created:               {expected_s2_nans} gaps masked.")
    print(f"✅ S1 Mask created (Union VV/VH): {expected_s1_union_nans} gaps masked.")

    return ds
