import os
import re
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm


def extract_cube_id(path):
    pattern = r"2\d{3}-\d{4}-[A-Z]{3}"
    match = re.search(pattern, path)
    return match.group(0) if match else "UNKNOWN"


def compute_valid_ratio(mask_array):
    """
    mask_array: (H, W)
    returns: ratio of valid pixels
    """
    return np.sum(mask_array > 0) / mask_array.size


def check_cube(
    path,
    context_len,
    target_len,
    ctx_min_valid_ratio,
    ctx_required_fraction,
    ctx_min_valid_overall,
    tgt_min_valid_ratio,
    tgt_min_timesteps,
):
    """
    Returns:
        (is_valid, reason)
    """

    try:
        ds = xr.open_zarr(path, consolidated=True)
    except Exception:
        ds = xr.open_zarr(path, consolidated=False)

    try:
        cutoff_date = pd.to_datetime(ds.attrs["precip_end_date"])
        full_time = pd.to_datetime(ds.time_sentinel_2_l2a.values)

        if cutoff_date not in full_time:
            return False, "cutoff_date_missing"

        # --- Context ---
        ds_ctx = ds.sel(time_sentinel_2_l2a=slice(None, cutoff_date)).tail(
            time_sentinel_2_l2a=context_len
        )

        if len(ds_ctx.time_sentinel_2_l2a) < context_len:
            return False, f"Too few ctx timesteps ({len(ds_ctx.time_sentinel_2_l2a)})"

        # use mask_s2 as quality proxy
        ctx_masks = ds_ctx["mask_s2"].values  # (T, H, W)

        valid_pixels_overall = np.mean(ctx_masks > 0)
        if valid_pixels_overall < ctx_min_valid_overall:
            return (
                False,
                f"Too few valid pixels overall in context ({valid_pixels_overall:.2%})",
            )

        valid_ctx_steps = 0
        for t in range(ctx_masks.shape[0]):
            ratio = compute_valid_ratio(ctx_masks[t])
            if ratio >= ctx_min_valid_ratio:
                valid_ctx_steps += 1

        if valid_ctx_steps < int(context_len * ctx_required_fraction):
            return False, f"Too few good ctx steps ({valid_ctx_steps})"

        # --- Target ---
        ds_target = ds.where(ds.time_sentinel_2_l2a > cutoff_date, drop=True).head(
            time_sentinel_2_l2a=target_len
        )

        if len(ds_target.time_sentinel_2_l2a) == 0:
            return False, "No target timesteps"

        tgt_masks = ds_target["mask_s2"].values

        valid_tgt_steps = 0
        for t in range(tgt_masks.shape[0]):
            ratio = compute_valid_ratio(tgt_masks[t])
            if ratio >= tgt_min_valid_ratio:
                valid_tgt_steps += 1

        if valid_tgt_steps < tgt_min_timesteps:
            return False, f"Too few valid target steps ({valid_tgt_steps})"

        return True, "ok"

    except Exception as e:
        return False, f"error: {str(e)}"

    finally:
        ds.close()


def update_exclusion_list(
    processed_dir,
    exclude_csv_path,
    cfg,
):
    """
    Main function to scan all cubes and update exclusion list.
    """

    # Load existing exclusions
    if os.path.exists(exclude_csv_path):
        df_ex = pd.read_csv(exclude_csv_path)
        existing_ids = set(df_ex["cube_id"].astype(str).str.strip())
    else:
        df_ex = pd.DataFrame(columns=["cube_id", "reason"])
        existing_ids = set()

    cube_paths = [
        os.path.join(processed_dir, d)
        for d in os.listdir(processed_dir)
        if d.endswith(".zarr")
    ]

    new_entries = []

    for path in tqdm(cube_paths):
        cube_id = extract_cube_id(path)

        if cube_id in existing_ids:
            continue

        is_valid, reason = check_cube(
            path,
            context_len=cfg["data"]["context_length"],
            target_len=cfg["data"]["target_length"],
            ctx_min_valid_ratio=cfg["data"]["quality_check"]["ctx_min_valid_ratio"],
            ctx_required_fraction=cfg["data"]["quality_check"]["ctx_required_fraction"],
            ctx_min_valid_overall=cfg["data"]["quality_check"]["ctx_min_valid_overall"],
            tgt_min_valid_ratio=cfg["data"]["quality_check"]["tgt_min_valid_ratio"],
            tgt_min_timesteps=cfg["data"]["quality_check"]["tgt_min_timesteps"],
        )

        if not is_valid:
            new_entries.append({"cube_id": cube_id, "reason": reason})
            print(f"[BAD] {cube_id}: {reason}")

    if len(new_entries) > 0:
        df_new = pd.DataFrame(new_entries)
        df_out = pd.concat([df_ex, df_new], ignore_index=True)
        df_out = df_out.drop_duplicates(subset="cube_id")
        df_out.to_csv(exclude_csv_path, index=False)
        print(f"Added {len(new_entries)} new excluded cubes.")
    else:
        print("No new bad cubes found.")
