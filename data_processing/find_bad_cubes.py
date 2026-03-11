import os
import xarray as xr
import pandas as pd
from tqdm import tqdm
from data_processing.scripts.sentinel_2_processing import get_s2_quality_masks
from data_processing.scripts.cube_processing import add_event_metadata
from dotenv import load_dotenv, find_dotenv
import s3fs
import fsspec

load_dotenv(find_dotenv())

# 1. Define data directories
S3_BASE_URL = os.getenv("S3_BASE_URL")
ERA5_PATH = os.getenv("ERA5_DATA_PATH")
BUCKET_NAME = os.getenv("BUCKET_NAME")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")


fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": S3_ENDPOINT})

# 2. LOAD & FILTER METADATA (CSV)
print("Loading and filtering metadata...")
df_data = pd.read_csv(
    "data_processing/data/1_max_precipitation_grid_cells.csv", sep=","
)

all_cube_ids = df_data["DisNo."].unique().tolist()

# 4. REMOTE DATA FETCHING
print(f"Fetching ALL {len(all_cube_ids)} cubes from S3...")
cubes = {}


def validate_s3_cubes(cube_ids, df_metadata, min_valid_steps=5, target_len=35):
    bad_cubes = []
    summary = []

    for cube_id in tqdm(cube_ids):
        # try:

        # Bestimme den Cutoff aus deinen Metadaten
        url = f"{S3_BASE_URL}{BUCKET_NAME}/DC__{cube_id}.zarr"
        mapper = fsspec.get_mapper(url)
        ds = xr.open_zarr(mapper, consolidated=True)

        ds = add_event_metadata(ds, df_metadata, cube_id)  # Fügt die Metadaten hinzu
        cutoff_date = pd.to_datetime(ds.attrs["precip_end_date"])

        # 2. S2 Qualitätsmasken berechnen (deine Funktion)
        ds = get_s2_quality_masks(ds)

        # Wir definieren eine "valide Beobachtung" als einen Zeitschritt,
        # in dem mindestens 20% des Cubes wolkenfrei sind
        valid_mask = ds.mask_phys_strict.mean(dim=["x", "y"]) > 0.20

        # 3. Kontext-Fenster Check
        # ctx_steps = ds.sel(time_sentinel_2_l2a=slice(None, cutoff_date))
        valid_ctx_count = int(
            valid_mask.sel(time_sentinel_2_l2a=slice(None, cutoff_date)).sum()
        )

        # 4. Target-Fenster Check (mindestens 1 klarer Blick innerhalb von 5 timesteps
        valid_tgt_count = int(
            valid_mask.sel(
                time_sentinel_2_l2a=slice(
                    cutoff_date + pd.Timedelta(days=1),
                    cutoff_date + pd.Timedelta(days=target_len),
                )
            ).sum()
        )

        if valid_ctx_count < min_valid_steps:
            reason = f"Too few ctx observations ({valid_ctx_count})"
            bad_cubes.append({"cube_id": cube_id, "reason": reason})
        elif valid_tgt_count == 0:
            reason = "No valid target observations"
            bad_cubes.append({"cube_id": cube_id, "reason": reason})
        else:
            summary.append(cube_id)

        # except Exception as e:
        #     bad_cubes.append({"cube_id": cube_id, "reason": f"Error: {str(e)}"})

    return summary, bad_cubes


good_list, bad_list = validate_s3_cubes(all_cube_ids, df_data)
pd.DataFrame(bad_list).to_csv("excluded_cubes.csv", index=False)
