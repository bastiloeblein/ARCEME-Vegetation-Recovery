import xarray as xr
import numpy as np
import os


def verify_cube_chunks(save_path):
    """
    Checks if spatial variables have the expected 4x4 chunk structure (250px each).
    Only prints if inconsistencies are found.
    """
    ds_check = xr.open_zarr(save_path, consolidated=True)
    cube_name = os.path.basename(save_path)
    found_issue = False

    for var in ds_check.data_vars:
        # Only check spatial variables (S1, S2, DEM, ESA_LC)
        if "x" in ds_check[var].dims and "y" in ds_check[var].dims:
            chunks = ds_check[var].chunks

            # Expecting y at index -2 and x at index -1
            y_chunks = len(chunks[-2])
            x_chunks = len(chunks[-1])

            # Check if 1000px is divided into 4 chunks of 250px
            if not (x_chunks == 4 and y_chunks == 4):
                if not found_issue:
                    print(f"\n⚠️ Chunk Issues detected in: {cube_name}")
                    found_issue = True
                print(
                    f"  ❌ {var:15}: WRONG ({y_chunks}x{x_chunks} chunks) - Expected 4x4"
                )

    ds_check.close()


def find_good_indices(ds, threshold=0.99):
    """
    Finds indices of Sentinel 2 timesteps, that are almost cloud free.
    Based on the "mask_phys_strict" mask
    """
    # 1. Calculates number of valid pixels per timestep
    valid_pixels = ds.kNDVI.notnull().sum(dim=("x", "y")).compute()

    # 2. Calculate percentage for each timestep
    total_pixels = ds.x.size * ds.y.size
    valid_pixel_perc = valid_pixels / total_pixels

    # 3. Find indices where the percentage is above the threshold
    indices = np.where(valid_pixel_perc > threshold)[0]

    # 4. Logging
    count = len(indices)
    print(f"Found good timesteps (> {threshold*100}%): {count}")

    return indices.tolist()
