import os
import sys
import gc
import base64
import warnings
import io
import pandas as pd
import numpy as np
import json
import xarray as xr
import matplotlib.pyplot as plt
import zarr
from dotenv import load_dotenv, find_dotenv
from data_processing.scripts.sentinel_1_processing import (
    find_global_veg_clipping_values,
    clip_s1_data,
    apply_lee_to_ds,
    normalize_s1_vars,
)  # , calculate_SAR_index #, aggregate_s1_causal_nearest
from data_processing.scripts.sentinel_2_processing import (
    get_s2_quality_masks,
    get_vegetation_mask,
    apply_masking,
    clean_and_normalize_bands,
    report_permanent_nans_for_var,
    calculate_s2_index,
    filter_static_vegetation_outliers,
    integrate_veg_and_wrongly_classified_mask,
)
from data_processing.scripts.plot_helpers_new import (
    find_cloud_free_indices,
    plot_statistical_outliers,
    plot_acquisition_timelines_filtered,
)
from data_processing.scripts.era_5_processing import (
    subset_era5_spatial,
    subset_era5_time,
    aggregate_era5_metrics_new,
)  # , create_uniform_era5_features, verify_era5_alignment
from data_processing.scripts.cube_processing import add_event_metadata
from data_processing.scripts.aggregation_5_day_interval import align_all_to_5d
from data_processing.scripts.normalize_and_clip import (
    normalize_dem,
    calculate_global_era5_stats,
    normalize_era5,
)


# Ignore warnings
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in cast"
)
warnings.filterwarnings("ignore", category=xr.SerializationWarning)


# --- 2. HTML REPORT FUNCTION ---
def create_html_report(info_dir, cube_id, report_sequence, filename="processing.html"):
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


def process_era5(cubes, era5_cubes, era5_dir):

    stats_path = os.path.join(era5_dir, "global_era5_stats.json")

    print(f"Pre-processing ERA5 for {len(cubes)} cubes...")
    all_era5_series = {}
    os.makedirs(era5_dir, exist_ok=True)

    # Loop over all cubes (train + test)
    for key, ds_target in cubes.items():

        cube_features = []

        # Caching check: if already exist load it
        era5_path = os.path.join(era5_dir, f"{key}_era5.zarr")
        if os.path.exists(era5_path):
            print(f"-> Using cached ERA5 for {key}")
            era5_merged = xr.open_zarr(era5_path, consolidated=True)
        else:
            for i, era5_cube_raw in enumerate(era5_cubes):
                print(f"#### Retrieving ERA5 data for cube {key} ###")
                # 1. Temporal subset
                tmp = subset_era5_time(era5_cube_raw, ds_target)

                if tmp is None:
                    print("❌ aggregate_era5_metrics_new hat None zurückgegeben!")

                # 2. Spatial subset
                tmp, _ = subset_era5_spatial(tmp, ds_target, plot_check=False)

                # 3. Convert tp from m to mm
                vars_list = list(tmp.data_vars)
                for v in vars_list:
                    if v.startswith("tp_"):
                        # Convert from meter to milimeter
                        tmp[v] = tmp[v] * 1000
                        tmp[v].attrs["units"] = "mm"

                # 4. Reduce dimensions to 1 (only time_sentinel_2_l2a)
                tmp = tmp.squeeze(dim=["latitude", "longitude"], drop=True)

                # Little check
                v0 = vars_list[0]
                if tmp[v0].isnull().all():
                    print(
                        f"⚠️ Warning: All NaNs in {v0} for cube {key}. Check spatial extent!"
                    )

                cube_features.append(tmp)

            # Merge all ERA5 vars for this cube
            era5_merged = xr.merge(cube_features)

            # Save ERA5 data separately for cube
            era5_merged.to_zarr(era5_path, mode="w", consolidated=True)

        # Add to dict for global statistic
        all_era5_series[key] = era5_merged

        if not all_era5_series:
            raise ValueError(
                "No ERA5 data was processed. Check input cubes and time ranges."
            )

    # --- GLOBAL STATISTIC CALCULATION ---
    # CAREFUL: ONLY CALCULATE ON TRAIN (DATA LEAKAGE!)
    train_keys = [k for k, ds in cubes.items() if ds.attrs.get("split") == "train"]
    train_series = {k: all_era5_series[k] for k in train_keys}

    print(
        f"Calculating global ERA5 stats based on {len(train_series)} training cubes..."
    )

    first_key = list(all_era5_series.keys())[0]
    era5_stats_vars = list(all_era5_series[first_key].data_vars)

    # ONLY CALCULATE GLOBAL STATS BASED ON TRAIN KEYS!
    global_era5_stats = calculate_global_era5_stats(train_series, era5_stats_vars)

    # Save stats
    save_stats_to_json(global_era5_stats, stats_path)

    print("✅ Global ERA5 stats calculated and ERA5 cubes saved.")

    # Cleanup
    del all_era5_series
    gc.collect()

    return global_era5_stats


# --- 2. DIE HAUPTSCHLEIFE ---


def run_processing_pipeline(
    cubes, era5_cubes, train_dir, test_dir, era5_dir, info_base, global_s1, global_era5
):
    """
    Main Orchestrator for the Satellite Data Processing Pipeline.
    Processes each cube 1-by-1 to minimize memory footprint.
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    vv_max, vh_max = global_s1

    cube_keys = list(cubes.keys())
    for n, key in enumerate(cube_keys):
        # Get cube and split
        split = cubes[key].attrs["split"]

        # Define save_path
        target_dir = train_dir if split == "train" else test_dir
        save_path = os.path.join(target_dir, f"{key}.zarr")

        # --- CHECKPOINT: Skip if already done ---
        if os.path.exists(save_path):
            sys.__stdout__.write(f"⏭️  Skipping {key}: Already processed.\n")
            cubes.pop(key)
            continue

        ds = cubes.pop(key)
        cube_id = key

        info_dir = os.path.join(info_base, split, cube_id)
        os.makedirs(info_dir, exist_ok=True)

        stdout_buffer = io.StringIO()
        sys.stdout = stdout_buffer
        report_sequence = []

        try:
            print(
                f"{'#'*10} PROCESSING CUBE: {cube_id} ({n+1}/{len(cubes)}) {'#'*10}\n"
            )

            # --- STAGE 1: SENTINEL-2 MASKING & QUALITY CONTROL ---
            print("Step 1: Applying Quality and Vegetation Masks...")
            ds = get_s2_quality_masks(ds)
            ds = get_vegetation_mask(ds)
            ds = apply_masking(ds)
            report_permanent_nans_for_var(ds, "B01", "time_sentinel_2_l2a")

            # --- STAGE 2: BAND NORMALIZATION ---
            print("Step 2: Cleaning and Normalizing Spectral Bands...")
            ds = clean_and_normalize_bands(ds)
            report_permanent_nans_for_var(ds, "B01_normalized", "time_sentinel_2_l2a")

            ## --- STAGE 3: INDEX CALCULATION ---
            print("Step 3: Calculating Vegetation and Water Indices...")
            for idx in ["NDVI", "kNDVI", "NIRv", "NDMI", "NDWI", "IRECI", "CIRE"]:
                ds = calculate_s2_index(ds, idx)
                if idx == "kNDVI":
                    report_permanent_nans_for_var(ds, idx, "time_sentinel_2_l2a")

            # --- STAGE 4: OUTLIER FILTERING & SPATIAL VALIDATION ---
            print("Step 4: Filtering Static Outliers...")
            ds = filter_static_vegetation_outliers(ds)
            indices = find_cloud_free_indices(ds)
            if len(indices) > 0:
                fig = plot_statistical_outliers(ds, indices[0], False)
                save_plot_to_report(fig, report_sequence, stdout_buffer)

            # --- STAGE 5: DATASET CLEANUP & INTEGRATION ---
            print("Step 5: Final Mask Integration and Dropping Raw Bands...")
            ds = integrate_veg_and_wrongly_classified_mask(ds)
            all_b_bands = [
                v for v in ds.data_vars if v.startswith("B") and v[1].isalnum()
            ]
            ds = ds.drop_vars(all_b_bands)

            # --- STAGE 6: SENTINEL-1 (SAR) PROCESSING ---
            print("Step 6: Processing Sentinel-1 (Radar) Data...")
            fig = plot_acquisition_timelines_filtered(
                ds, ds.attrs["precip_end_date"], 12, False
            )
            save_plot_to_report(fig, report_sequence, stdout_buffer)

            # Integrity check for NaNs
            nan_mask_vv_before = ds.vv.isnull()
            nan_mask_vh_before = ds.vh.isnull()

            # Process S1 vars
            ds = clip_s1_data(ds, global_vv_max, global_vh_max)
            ds = apply_lee_to_ds(ds, bands=["vv", "vh"], win_size=7, cu=0.25)
            ds = normalize_s1_vars(ds, global_vv_max, global_vh_max)

            # Integrity check for NaNs
            nan_mask_vv_after = ds.vv.isnull()
            nan_mask_vh_after = ds.vh.isnull()
            assert (
                nan_mask_vv_before == nan_mask_vv_after
            ).all(), "NaN mismatch in VV!"
            assert (
                nan_mask_vh_before == nan_mask_vh_after
            ).all(), "NaN mismatch in VH!"

            print("✅ Sentinel-1 processed (Clipping, Lee-Filter, Normalization).")

            # --- STAGE 7: TEMPORAL ALIGNMENT (5-DAY INTERVALS) ---
            print("Step 7: Resampling to regular 5-day intervals...")
            ds, fig_analysis = align_all_to_5d(ds, "strict", False)
            save_plot_to_report(fig_analysis, report_sequence, stdout_buffer)
            print("✅ Temporal alignment finished.")

            # --- STAGE 8: ERA5 CLIMATE DATA MERGING ---
            print("Step 8: Merging and Standardizing ERA5 Climate Data...")
            era5_path = os.path.join(era5_dir, f"{cube_id}_era5.zarr")

            if os.path.exists(era5_path):
                # 1. Load daily data
                era5_daily = xr.open_zarr(era5_path, consolidated=True)

                # 2. Aggreagtion to 5 day raster of Sentinel data
                vars_list = list(era5_daily.data_vars)
                era5_resampled = aggregate_era5_metrics_new(era5_daily, ds, vars_list)

                # Make sure that they have same temporal resolution
                era5_resampled = era5_resampled.reindex(
                    {"time_sentinel_2_l2a": ds.time_sentinel_2_l2a},
                    method="nearest",
                    tolerance=pd.Timedelta(days=1),
                )

                # Normalize ERA5
                era5_resampled = normalize_era5(era5_resampled, global_era5_stats)

                # Delete encodings to avoid conflicts
                for v in era5_resampled.data_vars:
                    era5_resampled[v].encoding = {}

                era5_valid_chunks = {
                    dim: chunks
                    for dim, chunks in ds.chunks.items()
                    if dim in era5_resampled.dims
                }
                era5_resampled = era5_resampled.chunk(era5_valid_chunks)

                # Merge
                ds = xr.merge([ds, era5_resampled], compat="override")
                ds = ds.unify_chunks()
                print("✅ ERA5 merged and standardized using global stats.")
            else:
                print(f"⚠️ Warning: No ERA5 cache found for {cube_id}")

            # --- STAGE 9: FINAL STANDARDIZATION & CLIPPING ---
            print("Step 9: Global Standardization and Outlier Clipping (-5, 5)...")
            ds = normalize_dem(ds)
            for var in ds.data_vars:
                if var != "ESA_LC":  # Keep Land Cover labels as they are
                    ds[var] = ds[var].clip(-5, 5)

            # --- STAGE 10: EXPORT ---
            print("Step 10: Final Export Preparation")

            compressor = zarr.Blosc(
                cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE
            )

            # 1. Globale Chunk-Definition (bleibt gleich)
            master_chunks = {"x": 250, "y": 250}
            if "time_sentinel_2_l2a" in ds.dims:
                master_chunks["time_sentinel_2_l2a"] = 1

            # Build the encoding dict:
            final_encoding = {}

            for var in ds.data_vars:
                # Delete existing encoding
                ds[var].encoding = {}

                # Define chunks
                var_dims = ds[var].dims
                safe_chunks = tuple(
                    master_chunks[dim] for dim in var_dims if dim in master_chunks
                )

                # Save Encoding dict for export
                final_encoding[var] = {
                    "compressor": compressor,
                    "chunks": safe_chunks,
                    "dtype": ds[var].dtype,
                }

                # Apply chunk on dataset
                ds[var] = ds[var].chunk(
                    {
                        dim: master_chunks[dim]
                        for dim in var_dims
                        if dim in master_chunks
                    }
                )

            # Final check and export
            ds = ds.unify_chunks()
            ds.to_zarr(save_path, mode="w", consolidated=True, encoding=final_encoding)
            print(f"\n✅ SUCCESS: Cube {cube_id} saved to {save_path}")

        except Exception as e:
            # Report the error in the HTML log
            print(f"\n{'!'*20} FATAL ERROR {'!'*20}")
            print(f"Cube ID: {cube_id}\nError: {str(e)}")
            # Ensure the error is also visible in the terminal
            sys.__stdout__.write(f"❌ Error in Cube {cube_id}: {str(e)}\n")

        finally:
            # Finalize Report
            report_sequence.append(("text", stdout_buffer.getvalue()))
            create_html_report(info_dir, cube_id, report_sequence)

            # Reset System State
            sys.stdout = sys.__stdout__
            stdout_buffer.close()

            # Explicit Memory Cleanup
            ds.close()
            del ds
            gc.collect()
            sys.__stdout__.write(f"Progress: {n+1}/{len(cubes)} cubes completed.\n")


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


def save_map_to_report(m, info_dir, cube_id, report_sequence):
    """
    Saves a Folium map as HTML and adds an IFrame reference to the report.
    """
    if m is None:
        return report_sequence

    # 1. Define path for the map file (within plots subfolder)
    map_filename = f"map_{cube_id}.html"
    plot_dir = os.path.join(info_dir, "plots")

    os.makedirs(plot_dir, exist_ok=True)
    map_path = os.path.join(plot_dir, map_filename)

    # 2. Save Folium map as standalone HTML file
    m.save(map_path)

    # 3. Create IFrame code for the report to keep the map interactive
    iframe_html = f'<iframe src="plots/{map_filename}" width="100%" height="500px" style="border:none;"></iframe>'

    report_sequence.append(("html_raw", iframe_html))
    return report_sequence


def save_stats_to_json(stats, file_path):
    """
    Saves the statistics dictionary as a JSON file, converting NumPy types to floats.
    """

    def convert_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    clean_stats = convert_types(stats)

    # Write cleaned dictionary to JSON file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(clean_stats, f, indent=4)

    print(f"📂 Global stats saved to: {file_path}")


def load_global_stats(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_s1_stats(vv_max, vh_max, file_path):
    stats = {"global_vv_max": float(vv_max), "global_vh_max": float(vh_max)}
    with open(file_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"💾 Sentinel-1 Stats gespeichert in {file_path}")


def load_s1_stats(file_path):
    with open(file_path, "r") as f:
        stats = json.load(f)
    return stats["global_vv_max"], stats["global_vh_max"]


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":

    load_dotenv(find_dotenv())

    # 1. Define data directories
    S3_BASE_URL = os.getenv("S3_BASE_URL")
    ERA5_PATH = os.getenv("ERA5_DATA_PATH")
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    BASE_OUTPUT_DIR = os.getenv("OUTPUT_DIR")
    S3_ENDPOINT = os.getenv("S3_ENDPOINT")  #

    # Subfolder for train, test and era5
    TRAIN_OUTPUT = os.path.join(BASE_OUTPUT_DIR, "train")
    TEST_OUTPUT = os.path.join(BASE_OUTPUT_DIR, "test")
    ERA5_DIR = os.path.join(BASE_OUTPUT_DIR, "era5")

    INFO_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reports/final_cube_info"
    )

    pei_cube_name = "PEICube_era5land.zarr"
    t2_cube_name = "t2_ERA5land.zarr"
    tp_cube_name = "tp_ERA5land.zarr"

    # Setup S3 Filesystem
    import s3fs
    import fsspec

    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": S3_ENDPOINT})

    # 2. LOAD & FILTER METADATA (CSV)
    print("Loading and filtering metadata...")
    df_data = pd.read_csv("data_processing/data/train_test_split.csv", sep=",")

    train_ids = df_data[df_data["split"] == "train"]["DisNo."].unique().tolist()
    test_ids = df_data[df_data["split"] == "test"]["DisNo."].unique().tolist()
    all_cube_ids = train_ids + test_ids

    # 4. REMOTE DATA FETCHING
    print(
        f"Fetching {len(all_cube_ids)} cubes (Train: {len(train_ids)}, Test: {len(test_ids)})..."
    )
    cubes = {}
    for cube_id in all_cube_ids:
        try:
            url = f"{S3_BASE_URL}{BUCKET_NAME}/DC__{cube_id}.zarr"
            mapper = fsspec.get_mapper(url)
            ds = xr.open_zarr(mapper, consolidated=True)

            # Ensure correct chunking
            ds = ds.chunk(
                {
                    "x": 500,
                    "y": 500,
                    "time_sentinel_2_l2a": -1,
                    "time_sentinel_1_rtc": -1,
                }
            )

            # Metadata & DEM Pre-processing
            ds.attrs["cube_id"] = cube_id
            ds.attrs["split"] = "train" if cube_id in train_ids else "test"
            ds = add_event_metadata(ds, df_data, cube_id)

            # Collapse DEM time dimension
            ds["COP_DEM"] = ds.COP_DEM.mean(dim="time_cop_dem_glo_30_dged_cog")
            ds = ds.drop_dims("time_cop_dem_glo_30_dged_cog")

            cubes[cube_id] = ds

        except Exception as e:
            print(f"❌ Failed to load Cube {cube_id}: {e}")

    # 5. STATS CALCULATION (ONLY TRAIN CUBES!)
    print("\n" + "═" * 60)
    print("PHASE: CALCULATING GLOBAL STATS (TRAIN ONLY)")
    print("═" * 60)

    # Filter for train cubes only
    train_cubes_dict = {k: v for k, v in cubes.items() if v.attrs["split"] == "train"}

    # 5.1 S1 Stats
    s1_stats_path = os.path.join(BASE_OUTPUT_DIR, "global_s1_stats.json")
    if os.path.exists(s1_stats_path):
        print(f"-> Loading existing S1 stats from {s1_stats_path}")
        global_vv_max, global_vh_max = load_s1_stats(s1_stats_path)
    else:
        print("-> S1 stats not found. Calculating global percentiles..")
        global_vv_max, global_vh_max = find_global_veg_clipping_values(train_cubes_dict)
        save_s1_stats(global_vv_max, global_vh_max, s1_stats_path)

    print("Global percentil values for all training cubes:\n")
    print("Global vv max: ", global_vv_max)
    print("Global vh max: ", global_vh_max)

    # 5.2 ERA5 Stats
    era5_stats_path = os.path.join(BASE_OUTPUT_DIR, "global_era5_stats.json")
    # Load ERA5 Cubes (PEI, T2m, TP)
    era5_cubes = [
        xr.open_zarr(os.path.join(ERA5_PATH, name), consolidated=False)
        for name in [pei_cube_name, t2_cube_name, tp_cube_name]
    ]

    if os.path.exists(era5_stats_path):
        print(f"-> Loading existing ERA5 global stats from {era5_stats_path}")
        global_era5_stats = load_global_stats(era5_stats_path)
        # Ensuring that an era5 cube will be created for each cube
        _ = process_era5(cubes, era5_cubes, ERA5_DIR)
    else:
        print("-> ERA5 stats not found. Starting Phase 1 (Global Extraction)...")
        global_era5_stats = process_era5(train_cubes_dict, era5_cubes, ERA5_DIR)

    # 6. RUN THE PIPELINE
    if cubes:
        try:
            run_processing_pipeline(
                cubes=cubes,
                era5_cubes=era5_cubes,
                train_dir=TRAIN_OUTPUT,
                test_dir=TEST_OUTPUT,
                era5_dir=ERA5_DIR,
                info_base=INFO_DIR,
                global_s1=(global_vv_max, global_vh_max),
                global_era5=global_era5_stats,
            )
            print("\n" + "=" * 60)
            print("🚀 PIPELINE EXECUTION FINISHED")
            print("=" * 60)
        except Exception as e:
            sys.stdout = sys.__stdout__
            print(f"Pipeline crashed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            sys.stdout = sys.__stdout__
    else:
        print("No cubes loaded. Pipeline skipped.")
