import json
import os
import re
from pathlib import Path

from collections import Counter
import numpy as np
import pandas as pd

RUN = Path("model/wand_db_logs/CV_Training_SGEDConvLSTM_RGBI_2026-08-03_20-20-39")
META = Path("data_processing/data/train_test_split.csv")

meta = pd.read_csv(META)
splits = json.loads((RUN / "cv_splits.json").read_text())

# Collect all validation  IDs from CV split, and check that each is unique.
validation_ids = [
    re.search(r"2\d{3}-\d{4}-[A-Z]{3}", str(path)).group()
    for name, fold in splits.items()
    if name.startswith("fold_")
    for path in fold["val_files"]
]

counts = Counter(validation_ids)
duplicates = {cube: n for cube, n in counts.items() if n != 1}
assert not duplicates, f"Multiple validation IDs: {duplicates}"

dev_ids = set(validation_ids)
print(f"CV-Check: {len(dev_ids)} cubes, each unique.")

dev = meta[meta["DisNo."].isin(dev_ids)].reset_index(drop=True)
test = meta[
    (meta["split"] == "test")
    # & (meta["DisNo."] != "2023-0457-NER")
].reset_index(drop=True)

print(f"Development: {len(dev)} Cubes | Test: {len(test)} Cubes")

assert dev["DisNo."].nunique() == len(dev_ids), "Development-IDs missing!"
assert not dev.empty and not test.empty
assert not pd.concat([dev, test])[["latitude", "longitude"]].isna().any().any()

# Haversine-Distances
lat_t = np.radians(test["latitude"].to_numpy())[:, None]
lon_t = np.radians(test["longitude"].to_numpy())[:, None]
lat_d = np.radians(dev["latitude"].to_numpy())[None, :]
lon_d = np.radians(dev["longitude"].to_numpy())[None, :]

a = (
    np.sin((lat_t - lat_d) / 2) ** 2
    + np.cos(lat_t) * np.cos(lat_d) * np.sin((lon_t - lon_d) / 2) ** 2
)
distances = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# Get the nearest development cube for each test cube.
nearest_idx = distances.argmin(axis=1)
nearest_km = distances[np.arange(len(test)), nearest_idx]

distance_table = (
    pd.DataFrame(
        {
            "test_cube_id": test["DisNo."].to_numpy(),
            "nearest_development_cube_id": dev.iloc[nearest_idx]["DisNo."].to_numpy(),
            "distance_km": nearest_km,
        }
    )
    .sort_values("distance_km")
    .reset_index(drop=True)
)

output_dir = os.path.join("evaluation", "results", "data_split")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "test_nearest_development_distances.csv")

distance_table.to_csv(output_path, index=False, float_format="%.6f")

closest = distance_table.iloc[0]

print(f"\nDevelopment: {len(dev)} Cubes | Test: {len(test)} Cubes")
print(f"Minimum distance: {closest['distance_km']:.2f} km")
print(f"Test-Cube:        {closest['test_cube_id']}")
print(f"Development-Cube: {closest['nearest_development_cube_id']}")
print(
    f"Average distance to the nearest development cube: " f"{nearest_km.mean():.2f} km"
)
print(f"\nCSV saved to: {output_path}")
