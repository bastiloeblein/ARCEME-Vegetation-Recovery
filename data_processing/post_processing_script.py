import os
import sys
import base64
import io
from dotenv import load_dotenv, find_dotenv
from data_processing.scripts.post_processing_checks import (
    verify_cube_chunks,
    find_good_indices,
)
from data_processing.scripts.plot_helpers_new import (
    find_cloud_free_indices,
    plot_rgb,
    plot_nan_distribution,
    plot_spatial_nan_frequency,
)
from data_processing.scripts.interpolation import (
    trim_to_first_s2_acquisition,
    interpolate_context_only,
    create_final_binary_masks,
)
from data_processing.scripts.sentinel_2_processing import get_s2_quality_masks
import xarray as xr
from data_processing.scripts.plot_helpers import plot_landcover
import matplotlib.pyplot as plt
import random
import s3fs
import fsspec
import pandas as pd
import numpy as np
import json


# --- 1. HTML REPORT FUNCTION ---
def create_html_report(
    info_dir, cube_id, report_sequence, filename="postprocessing.html"
):
    html_start = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Report {cube_id}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f0f2f5; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 20px; border-radius: 8px; }}
            pre {{ background: #222; color: #fff; padding: 10px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 100%; border: 1px solid #ccc; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Cube Report: {cube_id}</h1>
    """
    html_end = "</div></body></html>"

    content = ""
    for entry_type, value in report_sequence:
        if entry_type == "text":
            if value.strip():
                content += f"<pre>{value.strip()}</pre>"
        elif entry_type == "plot_b64":
            # Hier wird der Base64-String direkt eingebettet
            content += (
                f'<div class="plot"><img src="data:image/png;base64,{value}"></div>'
            )
        elif entry_type == "html_raw":
            # Fügt den IFrame direkt ein
            content += f'<div class="map-container">{value}</div>'

    report_path = os.path.join(info_dir, filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_start + content + html_end)


def save_plot_to_report(fig, report_sequence, stdout_buffer):
    """
    Saves the current stdout buffer and the plot as Base64 to the report sequence.
    Clears the buffer afterwards to prepare for the next step.
    """
    # 1. Retrieve current text logs from buffer and add to sequence
    text_content = stdout_buffer.getvalue()
    if text_content.strip():
        report_sequence.append(("text", text_content))

    # 2. Convert plot to Base64 string
    tmp_buffer = io.BytesIO()
    fig.savefig(tmp_buffer, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)  # Free RAM immediately

    b64_img = base64.b64encode(tmp_buffer.getvalue()).decode()
    report_sequence.append(("plot_b64", b64_img))

    # 3. Clear buffer for the next processing section
    stdout_buffer.truncate(0)
    stdout_buffer.seek(0)

    return report_sequence


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":

    load_dotenv(find_dotenv())

    # 1. Define data directories
    S3_BASE_URL = os.getenv("S3_BASE_URL")
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    CUBE_DIR = os.getenv("OUTPUT_DIR")
    S3_ENDPOINT = os.getenv("S3_ENDPOINT")
    OUTPUT_DIR = os.path.join(CUBE_DIR, "postprocessed")
    INFO_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports"
    )

    # Initialize stats dict for nan reduction calculation
    all_cubes_stats = {}

    # Pixel counts for nan reduction calculation
    global_totals = {
        "kNDVI": {
            "pre_ctx_nans": 0,
            "post_ctx_nans": 0,
            "ctx_total": 0,
            "after_nans": 0,
            "after_total": 0,
        },
        "vv": {
            "pre_ctx_nans": 0,
            "post_ctx_nans": 0,
            "ctx_total": 0,
            "after_nans": 0,
            "after_total": 0,
        },
    }

    # Define vars of interest
    S2_VARS = ["NDVI", "kNDVI", "CIRE", "IRECI", "NDWI", "NDMI", "NIRv"]
    S1_VARS = ["vv", "vh"]
    STATIC_VARS = ["ESA_LC", "COP_DEM", "is_veg"]
    ERA5_VARS = [
        "pei_30_mean",
        "pei_90_mean",
        "pei_180_mean",
        "pei_360_mean",
        "t2m_mean",
        "t2mmax_mean",
        "t2mmin_mean",
        "tp_dailymax_mean",
        "tp_dailymean_mean",
        "tp_rollingmax_mean",
    ]

    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": S3_ENDPOINT})

    # Get cubes
    cube_list = os.listdir(CUBE_DIR)
    cube_list = [file for file in cube_list if file[0] == "2"]

    for n, cube_name in enumerate(cube_list):
        # --- SETUP REPORTING ---
        stdout_buffer = io.StringIO()
        sys.stdout = stdout_buffer
        report_sequence = []

        # Verify chunking
        verify_cube_chunks(os.path.join(CUBE_DIR, cube_name))

        # Load cube
        ds = xr.open_zarr(os.path.join(CUBE_DIR, cube_name), consolidated=True)
        cube_id = ds.attrs["cube_id"]

        print(f"Processing {cube_id}")

        # Create postprocessing file in processing_info/{cube_id}
        info_dir = os.path.join(INFO_DIR, cube_id)

        try:

            # Get original cube to plot as rgb
            url = f"{S3_BASE_URL}{BUCKET_NAME}/DC__{cube_id}.zarr"
            mapper = fsspec.get_mapper(url)
            ds_plot = xr.open_zarr(mapper, consolidated=True)
            ds_plot = get_s2_quality_masks(ds_plot)
            indices_rgb = find_cloud_free_indices(ds_plot)

            #  1. RGB Snapshot
            print("RGB Snapshot")
            if indices_rgb is not None and len(indices_rgb) > 0:
                idx = int(indices_rgb[0])
                fig = plot_rgb(ds_plot, idx)
                save_plot_to_report(fig, report_sequence, stdout_buffer)
            else:
                print("❌ No cloud-free indices available to plot RGB.")

            # 2. S2 Variable Plot
            print("Visual analysis of S2 Variables")
            indices = find_good_indices(ds)

            if indices:
                selected_idx = indices[0]
                is_fallback = False
            else:
                print("⚠️ No cloud-free indices found via find_good_indices.")
                # Fallback: Selected timestep with fewest NaNs
                nan_counts_s2 = ds["kNDVI"].isnull().sum(dim=["x", "y"]).compute()
                selected_idx = int(nan_counts_s2.argmin())
                nan_percent = (
                    nan_counts_s2[selected_idx] / (ds.x.size * ds.y.size) * 100
                ).values
                print(
                    f"Selecting fallback index {selected_idx} (minimal NaNs in kNDVI: {nan_percent:.2%})"
                )
                is_fallback = True

            # Jetzt für alle Variablen diesen Index plotten
            for var in S2_VARS:
                print("#" * 10, f" {var} ", "#" * 10)
                fig = plt.figure(figsize=(8, 6))

                try:
                    title_suffix = " (Minimal NaNs Fallback)" if is_fallback else ""
                    ds[var].isel(time_sentinel_2_l2a=selected_idx).plot(
                        robust=True, cmap="viridis"
                    )
                    plt.title(f"Variable: {var} at index {selected_idx}{title_suffix}")

                    # In den Report speichern
                    save_plot_to_report(fig, report_sequence, stdout_buffer)
                except Exception as e:
                    print(f"Could not plot {var}: {e}")
                    plt.close(fig)

            # 3. S1 Variable Plots
            print("Visual analysis of S1 Variables")
            # 3.1. Calculate percentage of NaNs per timesteps
            nan_ratio = ds[S1_VARS[0]].isnull().mean(dim=["x", "y"]).compute()
            # 3.2. Find all indices where nan_ratio <0.1
            valid_indices = np.where(nan_ratio < 0.1)[0]
            if len(valid_indices) > 0:
                # 3.3 Select random valid index
                idx_s1 = int(random.choice(valid_indices))
                print(
                    f"Selected Index: {idx_s1} (NaN-Proportion: {nan_ratio[idx_s1].values:.2%})"
                )
            else:
                print("No timestep with < 10% NaNs found!")
                # Fallback: Take timestep with fewest nans
                idx_s1 = int(nan_ratio.argmin())
                print(
                    f"Using best available timestep: {idx_s1} ({nan_ratio[idx_s1].values:.2%} NaNs)"
                )

            for var in S1_VARS:
                print("#" * 10, {var}, "#" * 10)
                fig = plt.figure(figsize=(8, 6))
                ds[var].isel(time_sentinel_2_l2a=idx_s1).plot(
                    robust=True, cmap="viridis"
                )
                plt.title(f"Variable: {var} at index {idx_s1}")
                save_plot_to_report(fig, report_sequence, stdout_buffer)

            #  4. Static plots
            # Delete DEM time_dim if still present
            dim = "time_esa_worldcover"
            if dim in ds.dims:
                ds = ds.squeeze(dim, drop=True)

            print("Visual analysis of static Variables")
            for var in STATIC_VARS:
                print("#" * 10, {var}, "#" * 10)
                fig = plt.figure(figsize=(8, 6))
                static_dim = ds[var].dims[0]
                # assert len(ds[var][static_dim]) == 1 | len(ds[var][static_dim]) == 1000
                print("Dimension Length: ", len(ds[var][static_dim]))
                if len(ds[var][static_dim]) == 1:
                    ds[var].isel({static_dim: 0}).plot()
                else:
                    ds[var].plot()
                plt.title(f"Variable: {var}")
                save_plot_to_report(fig, report_sequence, stdout_buffer)

            # 5. Visual comparison
            print("Visual comparison of RGB, Landcover and Vegetation Mask")
            fig_comp, axes = plt.subplots(1, 3, figsize=(25, 10))
            # 5.1. Left plot: RGB Plot
            plot_rgb(ds_plot, indices_rgb[0], ax=axes[0])
            axes[0].set_title("RGB Snapshot")
            # 5.2. Center Plot: Landcover
            plot_landcover(ds, ax=axes[1])
            # 5.3. Right Plot: Vegetation Mask
            if "is_veg" in ds:
                ds.is_veg.plot(ax=axes[2], add_colorbar=False)
                axes[2].set_title("Vegetation Mask ")
                axes[2].set_aspect("equal")
            axes[0].axis("off")
            axes[1].axis("off")
            axes[2].axis("off")
            plt.tight_layout()
            save_plot_to_report(fig_comp, report_sequence, stdout_buffer)

            # 6. Spatial NAN Analysis
            print("Spatial NaN analysis")
            for var in S2_VARS + S1_VARS:
                print("#" * 10, {var}, "#" * 10)
                fig = plot_spatial_nan_frequency(ds, var, ds.attrs["precip_end_date"])
                save_plot_to_report(fig, report_sequence, stdout_buffer)

            #  6. ERA5 plots
            print("Visual analysis of ERA5 variables")
            for var in ERA5_VARS:
                print("#" * 10, {var}, "#" * 10)
                fig = plt.figure(figsize=(8, 6))
                ds[var].plot()
                plt.title(f"Variable: {var}")
                save_plot_to_report(fig, report_sequence, stdout_buffer)

            #  7. Interpolation & before/after comparison
            print("\n--- Interpolation Analysis ---")
            print("Number of timesteps before trimming: ", len(ds.time_sentinel_2_l2a))
            ds = trim_to_first_s2_acquisition(ds)
            print("Number of timesteps after trimming: ", len(ds.time_sentinel_2_l2a))

            vars_to_interpolate = [
                v
                for v in ds.data_vars
                if "time_sentinel_2_l2a" in ds[v].dims
                and not any(suffix in v for suffix in ["_count", "std", "_mask"])
            ]

            # Interpolate
            ds_intp = interpolate_context_only(ds, vars_to_interpolate)

            # NaN analysis
            # Define context window
            cutoff_date = pd.to_datetime(ds.attrs["precip_end_date"])
            ds_ctx = ds.sel(time_sentinel_2_l2a=slice(None, cutoff_date))
            ds_intp_ctx = ds_intp.sel(time_sentinel_2_l2a=slice(None, cutoff_date))

            # --- Quality Metrics ---
            cube_stats = {}
            for v_comp in ["kNDVI", "vv"]:
                # 1. Calculate data
                ctx_total = ds_ctx[v_comp].size
                ctx_nans_before = int(ds_ctx[v_comp].isnull().sum().compute())
                ctx_nans_after = int(ds_intp_ctx[v_comp].isnull().sum().compute())
                n_filled = ctx_nans_before - ctx_nans_after
                pct_before = ctx_nans_before / ctx_total * 100
                pct_after = ctx_nans_after / ctx_total * 100
                fill_rate = (
                    (n_filled / ctx_nans_before * 100) if ctx_nans_before > 0 else 0
                )

                # 2. Format Dashboard
                print("\n" + "═" * 60)
                print(f"📊 CONTEXT WINDOW INTERPOLATION: {v_comp}")
                print(f"📅 Period: Start to {cutoff_date.strftime('%Y-%m-%d')}")
                print("─" * 60)
                print(
                    f"  NaNs before:    {ctx_nans_before:>12,} ({pct_before:>6.2f}% of context)"
                )
                print(
                    f"  NaNs after :    {ctx_nans_after:>12,} ({pct_after:>6.2f}% of context)"
                )
                print(f"  {'-' * 50}")
                print(f"  ✅ FILLED:  {n_filled:>12,} ({fill_rate:.1f}% of gaps)")
                print("═" * 60 + "\n")

                # Period after cutoff
                ds_after = ds_intp.sel(
                    time_sentinel_2_l2a=slice(cutoff_date + pd.Timedelta(days=1), None)
                )
                after_total = ds_after[v_comp].size
                after_nans = int(ds_after[v_comp].isnull().sum().compute())

                # Save to cube-specific dict
                cube_stats[v_comp] = {
                    "ctx_nans_before": ctx_nans_before,
                    "ctx_nans_after": ctx_nans_after,
                    "ctx_total_pixels": ctx_total,
                    "after_cutoff_nans": after_nans,
                    "after_cutoff_total_pixels": after_total,
                }

                # Add to global dict
                global_totals[v_comp]["pre_ctx_nans"] += ctx_nans_before
                global_totals[v_comp]["post_ctx_nans"] += ctx_nans_after
                global_totals[v_comp]["ctx_total"] += ctx_total
                global_totals[v_comp]["after_nans"] += after_nans
                global_totals[v_comp]["after_total"] += after_total

            # Add to global dict
            all_cubes_stats[cube_id] = cube_stats

            fig = plot_nan_distribution(ds, ds_intp, "kNDVI", cutoff_date)
            save_plot_to_report(fig, report_sequence, stdout_buffer)

            fig = plot_nan_distribution(ds, ds_intp, "vv", cutoff_date)
            save_plot_to_report(fig, report_sequence, stdout_buffer)

            # Create final masks
            ds_final = create_final_binary_masks(ds_intp)

            # ----- EXPORT ----
            # Assure variable dtypes
            uint8_vars = ["ESA_LC", "is_veg", "mask_s1", "mask_s2", "target_mask"]
            for var in list(ds_final.data_vars):
                ds_final[var].encoding = {}  # remove old encodings
                if var in ["s1_final_mask", "s2_final_mask"]:
                    ds_final = ds_final.drop_vars(var)
                    continue
                if var in uint8_vars or var.endswith("_count"):
                    ds_final[var] = ds_final[var].astype("uint8")
                else:
                    ds_final[var] = ds_final[var].astype("float32")

            # Compression and chunking
            import zarr

            compressor = zarr.Blosc(
                cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE
            )
            target_chunks = {"time_sentinel_2_l2a": 1, "x": 250, "y": 250}

            final_encoding = {}
            for var in ds_final.data_vars:

                # Define chunks
                var_dims = ds_final[var].dims
                safe_chunks_tuple = tuple(
                    target_chunks[d] for d in var_dims if d in target_chunks
                )

                # Save Encoding dict for export
                final_encoding[var] = {
                    "compressor": compressor,
                    "chunks": safe_chunks_tuple,
                    "dtype": ds_final[var].dtype,
                }

                # Apply chunk on dataset
                ds_final[var] = ds_final[var].chunk(
                    {d: target_chunks[d] for d in var_dims if d in target_chunks}
                )

            # Save to disk
            save_path = os.path.join(OUTPUT_DIR, f"{cube_id}_postprocessed.zarr")
            ds_final.to_zarr(
                save_path, mode="w", consolidated=True, encoding=final_encoding
            )
            print(f"✅ SUCCESS: Postprocessed cube saved to {save_path}")

        except Exception as e:
            print(f"\nERROR: {str(e)}")
        finally:
            text_content = stdout_buffer.getvalue()
            if text_content.strip():
                report_sequence.append(("text", text_content))

            create_html_report(info_dir, cube_id, report_sequence)

            # System-Output zurücksetzen
            sys.stdout = sys.__stdout__
            stdout_buffer.close()
            plt.close("all")
            print(f"Report for {cube_id} saved to {info_dir}")

    # --- FINALE Evaluation over all cubes---
    print("\n" + "█" * 80)
    print("      GLOBAL POST-PROCESSING SUMMARY (ALL CUBES)      ")
    print("█" * 80)

    for var in ["kNDVI", "vv"]:
        g = global_totals[var]

        # Berechnungen
        total_ctx_reduction = (
            (g["pre_ctx_nans"] - g["post_ctx_nans"]) / g["pre_ctx_nans"] * 100
            if g["pre_ctx_nans"] > 0
            else 0
        )
        final_ctx_nan_pct = g["post_ctx_nans"] / g["ctx_total"] * 100
        after_nan_pct = (
            g["after_nans"] / g["after_total"] * 100 if g["after_total"] > 0 else 0
        )

        print(f"\n➔ VARIABLE: {var}")
        print("-" * 30)
        print("  CONTEXT WINDOW (Pre-Event):")
        print(f"    Total Pixels:     {g['ctx_total']:>15,}")
        print(
            f"    NaNs Before:      {g['pre_ctx_nans']:>15,} ({g['pre_ctx_nans']/g['ctx_total']*100:>5.2f}%)"
        )
        print(
            f"    NaNs After:       {g['post_ctx_nans']:>15,} ({final_ctx_nan_pct:>5.2f}%)"
        )
        print(
            f"    ✅ REDUCTION:     {total_ctx_reduction:>14.1f}% of missing values filled"
        )

        print("\n  AFTER CUTOFF (Event/Post-Event):")
        print(f"    Total Pixels:     {g['after_total']:>15,}")
        print(f"    Remaining NaNs:   {g['after_nans']:>15,} ({after_nan_pct:>5.2f}%)")

    print("\n" + "█" * 80)

    # Save statistics as json
    stats_save_path = os.path.join(OUTPUT_DIR, "global_postprocessing_stats.json")
    with open(stats_save_path, "w") as f:
        json.dump(all_cubes_stats, f, indent=4)
    print(f"Detailed stats for all cubes saved to {stats_save_path}")
