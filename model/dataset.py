import os
import torch
import pandas as pd
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import random
import re


class ARCEME_Dataset(Dataset):
    """
    Custom Dataset for loading multi-modal Earth observation data cubes.
    """

    def __init__(
        self,
        cube_paths,
        context_length,
        target_length,
        patch_size,
        train,
        exclude_file=None,
        s2_vars=None,
        s1_vars=None,
        era5_vars=None,
        static_vars=None,
        fixed_tiles=None,
        use_augmentation=False,
    ):
        """
        Initializes the dataset with cube paths and variable configurations.

        Args:
            cube_paths (list): Paths to the .zarr data cubes.
            context_length (int): Number of input timesteps (X).
            target_length (int): Number of prediction timesteps (Y).
            patch_size (int): Spatial dimensions for cropping.
            train (bool): Enables random patching if True, else fixed tiling.
            s2_vars (list, optional): Sentinel-2 derived indices (kNDVI, NDWI, ...).
            s1_vars (list, optional): Sentinel-1 radar bands (vv, vh).
            era5_vars (list, optional): ERA5 variables.
            static_vars (list, optional): Non-temporal variables (e.g., DEM).
            fixed_tiles (list, optional): Spatial offsets for validation tiling.
            use_augmentation (bool): Whether to apply data augmentation.
        """
        self.cube_paths = self._filter_cube_paths(cube_paths, exclude_file)
        self.context_len = context_length
        self.target_len = target_length
        self.patch_size = patch_size
        self.train = train
        self.fixed_tiles = fixed_tiles
        self.use_augmentation = use_augmentation

        # Variable configuration
        self.s2_vars = s2_vars or []
        self.s1_vars = s1_vars or []
        self.era5_vars = era5_vars or []
        self.static_vars = static_vars or []

        # List of all variables for indexing/debugging (order matters!)
        self.all_vars_ordered = (
            self.s2_vars
            + self.s1_vars
            + self.era5_vars
            + ["mask_s2", "mask_s1", "ESA_OH"]
            + self.static_vars
        )

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        if self.fixed_tiles is not None:
            return len(self.fixed_tiles)

        # Upsampling for small datasets to ensure enough iterations per epoch
        if len(self.cube_paths) < 50:
            return len(self.cube_paths) * 10

        return len(self.cube_paths)

    def _check_shape(self, tensor, expected_shape, name, path):
        """
        Validates tensor shape against an expected shape tuple.
        Use None in expected_shape as wildcard for that dimension.
        """
        actual_shape = tuple(tensor.shape)
        if len(actual_shape) != len(expected_shape):
            raise ValueError(
                f"{name} rank mismatch: expected {len(expected_shape)}D {expected_shape}, "
                f"got {actual_shape}. Path: {path}"
            )

        for i, (actual_dim, expected_dim) in enumerate(
            zip(actual_shape, expected_shape)
        ):
            if expected_dim is not None and actual_dim != expected_dim:
                raise ValueError(
                    f"{name} shape mismatch at dim {i}: expected {expected_shape}, "
                    f"got {actual_shape}. Path: {path}"
                )

    def _validate_time_windows(self, ds, ds_ctx, ds_target, cutoff_date, path):
        """
        Validates context/target time slicing consistency.

        Note:
            cutoff_date must be part of the available timestamps.
            Context includes all timesteps up to and including cutoff_date,
            and target starts at the immediate next available timestep after cutoff_date.
        """
        full_time = pd.to_datetime(ds.time_sentinel_2_l2a.values)
        if cutoff_date not in full_time:
            raise ValueError(
                f"cutoff_date={cutoff_date} is not an available timestep. Path: {path}"
            )

        if len(ds_ctx.time_sentinel_2_l2a) != self.context_len:
            raise ValueError(
                f"Context time mismatch: expected {self.context_len}, "
                f"got {len(ds_ctx.time_sentinel_2_l2a)}. Path: {path}"
            )

        if len(ds_target.time_sentinel_2_l2a) != self.target_len:
            raise ValueError(
                f"Target time mismatch: expected {self.target_len}, "
                f"got {len(ds_target.time_sentinel_2_l2a)}. Path: {path}"
            )

        actual_ctx_end = pd.to_datetime(ds_ctx.time_sentinel_2_l2a[-1].values)
        if actual_ctx_end != cutoff_date:
            raise ValueError(
                f"Context end mismatch! Expected {cutoff_date}, "
                f"but got {actual_ctx_end}. Path: {path}"
            )

        next_time_after_cutoff = full_time[full_time > cutoff_date]
        if len(next_time_after_cutoff) == 0:
            raise ValueError(
                f"No timestep exists after cutoff_date={cutoff_date}. Path: {path}"
            )

        expected_target_start = pd.to_datetime(next_time_after_cutoff[0])
        actual_target_start = pd.to_datetime(ds_target.time_sentinel_2_l2a[0].values)
        if actual_target_start != expected_target_start:
            raise ValueError(
                f"Target start mismatch! Expected next timestep after cutoff "
                f"({expected_target_start}), but got {actual_target_start}. Path: {path}"
            )

        if actual_target_start <= cutoff_date:
            raise ValueError(
                f"Target must start after cutoff_date={cutoff_date}, "
                f"but got {actual_target_start}. Path: {path}"
            )

    def _filter_cube_paths(self, cube_paths, exclude_file):
        """
        Filters the list of cube paths based on an exclusion file or list.
        """
        pattern = r"2\d{3}-\d{4}-[A-Z]{3}"
        excluded_ids = set()

        # 1. Load excluded IDs from file or list
        if exclude_file is not None:
            if isinstance(exclude_file, str) and exclude_file.endswith(".csv"):
                df_ex = pd.read_csv(exclude_file)
                if "cube_id" in df_ex.columns:
                    excluded_ids = set(
                        df_ex["cube_id"].astype(str).str.strip().tolist()
                    )
            elif isinstance(exclude_file, list):
                excluded_ids = set([str(i).strip() for i in exclude_file])

        if not excluded_ids:
            return cube_paths

        # Filter
        filtered_paths = []
        for path in cube_paths:
            match = re.search(pattern, path)
            if match:
                cube_id = match.group(0)
                if cube_id not in excluded_ids:
                    filtered_paths.append(path)
            else:
                pass

        print(f"Filter applied: {len(cube_paths)} -> {len(filtered_paths)} Cubes.")

        return filtered_paths

    def __getitem__(self, idx):
        """
        Loads and processes a single data sample.

        Returns:
            tuple: (x_context, x_future_feat, y_target, target_mask, meta, baseline_sample)
        """
        # All cubes are 1000 x 1000 pixel
        h, w = 1000, 1000
        pattern = r"2\d{3}-\d{4}-[A-Z]{3}"

        # ======================================================================
        # 1. PATCHING STRATEGY
        # ======================================================================
        if not self.train and self.fixed_tiles is not None:
            # VALIDATION: Use the pre-calculated grid offsets for patching
            tile_info = self.fixed_tiles[idx]
            path = tile_info["path"]
            top = tile_info["top"]
            left = tile_info["left"]
        else:
            # TRAINING: Random cropping from a randomly (through shuffling in the dataloader) selected cube
            real_idx = idx % len(self.cube_paths)
            path = self.cube_paths[real_idx]
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)

        # ======================================================================
        # 2. DATA LOADING & SPATIAL SLICING
        # ======================================================================
        try:
            ds = xr.open_zarr(path, consolidated=True)
        except Exception:
            ds = xr.open_zarr(path, consolidated=False)

        # Get cube_id
        match = re.search(pattern, path)
        if match:
            cube_id = match.group(0)
        else:
            cube_id = "No_ID_Found"

        # Select spatial patch: (time, y, x)
        ds = ds.isel(
            y=slice(top, top + self.patch_size), x=slice(left, left + self.patch_size)
        )

        # Check if ds is expected size
        if ds.x.size != self.patch_size or ds.y.size != self.patch_size:
            raise ValueError(
                f"Patch slicing failed at {path}. Expected {self.patch_size}x{self.patch_size}, "
                f"but received {ds.x.size}x{ds.y.size} at top={top}, left={left}."
            )

        # ======================================================================
        # 3. TIME WINDOW SLICING (Context vs Target)
        # ======================================================================
        cutoff_date = pd.to_datetime(ds.attrs["precip_end_date"])

        # cutoff_date must be an available timestep in the series
        full_time = pd.to_datetime(ds.time_sentinel_2_l2a.values)
        if cutoff_date not in full_time:
            raise ValueError(
                f"cutoff_date={cutoff_date} is not an available timestep. Path: {path}"
            )

        # ds_ctx: (time_context, y, x)
        # Includes cutoff_date
        ds_ctx = ds.sel(time_sentinel_2_l2a=slice(None, cutoff_date)).tail(
            time_sentinel_2_l2a=self.context_len
        )

        # ds_target: (time_target, y, x)
        ds_target = ds.where(ds.time_sentinel_2_l2a > cutoff_date, drop=True).head(
            time_sentinel_2_l2a=self.target_len
        )

        self._validate_time_windows(
            ds=ds,
            ds_ctx=ds_ctx,
            ds_target=ds_target,
            cutoff_date=cutoff_date,
            path=path,
        )

        # Fill NaNs with 0.0
        ds_ctx = ds_ctx.fillna(0.0)
        ds_target = ds_target.fillna(0.0)  # think about not filling!

        # ======================================================================
        # 4. INPUT CONSTRUCTION (CONTEXT WINDOW)
        # ======================================================================
        # x_s2: (C_s2, T_ctx, H, W)  -- C_s2 will be in the order how I passed the list self.s2_var (Ensure target is at first position!)
        x_s2 = torch.from_numpy(ds_ctx[self.s2_vars].to_array().values).float()
        self._check_shape(
            x_s2,
            (len(self.s2_vars), self.context_len, self.patch_size, self.patch_size),
            "x_s2",
            path,
        )

        # x_s1: (C_s1, T_ctx, H, W)
        x_s1 = torch.from_numpy(ds_ctx[self.s1_vars].to_array().values).float()
        self._check_shape(
            x_s1,
            (len(self.s1_vars), self.context_len, self.patch_size, self.patch_size),
            "x_s1",
            path,
        )

        # x_era5_1d: (C_era5, T_ctx) -> Broadcasted: (C_era5, T_ctx, H, W)
        if len(self.era5_vars) > 0:
            x_era5_1d = torch.from_numpy(
                ds_ctx[self.era5_vars].to_array().values
            ).float()
            x_era5 = broadcast_era5(x_era5_1d, self.patch_size, self.patch_size)
        else:
            # Erzeuge einen leeren Tensor mit 0 Kanälen, aber passenden anderen Dimensionen
            x_era5 = torch.empty(
                (0, self.context_len, self.patch_size, self.patch_size)
            )
        self._check_shape(
            x_era5,
            (len(self.era5_vars), self.context_len, self.patch_size, self.patch_size),
            "x_era5",
            path,
        )

        # ======================================================================
        # 5. MASKING & VEGETATION LOGIC
        # ======================================================================
        # Vegetation mask: (T_ctx, H, W)
        is_veg = torch.from_numpy(ds_ctx["is_veg"].values).float()

        # S2 / S1 Mask combined with vegetation mask: (1, T_ctx, H, W)
        m_s2 = (torch.from_numpy(ds_ctx["mask_s2"].values).float() * is_veg).unsqueeze(
            0
        )
        m_s1 = (torch.from_numpy(ds_ctx["mask_s1"].values).float() * is_veg).unsqueeze(
            0
        )
        self._check_shape(
            m_s2,
            (1, self.context_len, self.patch_size, self.patch_size),
            "m_s2",
            path,
        )
        self._check_shape(
            m_s1,
            (1, self.context_len, self.patch_size, self.patch_size),
            "m_s1",
            path,
        )

        # ======================================================================
        # 6. LANDCOVER & STATIC FEATURES
        # ======================================================================
        # lc: (T_ctx, H, W) -> lc_onehot: (T_ctx, 12, H, W)
        lc = torch.from_numpy(ds_ctx["ESA_LC"].values).long()
        lc_onehot = encode_landcover(lc)
        self._check_shape(
            lc_onehot,
            (self.context_len, 12, self.patch_size, self.patch_size),
            "lc_onehot",
            path,
        )

        # x_stat_raw: (C_stat, T_ctx, H, W)
        if len(self.static_vars) > 0:
            x_stat_raw = torch.from_numpy(
                ds_ctx[self.static_vars].to_array().values
            ).float()
        else:
            x_stat_raw = torch.empty(
                (0, self.context_len, self.patch_size, self.patch_size)
            )
        self._check_shape(
            x_stat_raw,
            (len(self.static_vars), self.context_len, self.patch_size, self.patch_size),
            "x_stat_raw",
            path,
        )

        # Combine LC and Statics -> x_static: (C_lc+stat, T_ctx, H, W)
        # Note: Permute used to align channels for concatenation
        x_static = torch.cat(
            [lc_onehot, x_stat_raw.permute(1, 0, 2, 3)], dim=1
        ).permute(1, 0, 2, 3)
        self._check_shape(
            x_static,
            (
                len(self.static_vars) + lc_onehot.shape[1],
                self.context_len,
                self.patch_size,
                self.patch_size,
            ),
            "x_static",
            path,
        )

        # Final Context Stack -> x_context: (T_ctx, C_total, H, W)
        x_context = torch.cat(
            [x_s2, x_s1, x_era5, m_s2, m_s1, x_static], dim=0
        ).permute(1, 0, 2, 3)
        expected_channels = (
            len(self.s2_vars)
            + len(self.s1_vars)
            + len(self.era5_vars)
            + 2
            + len(self.static_vars)
            + lc_onehot.shape[1]
        )
        self._check_shape(
            x_context,
            (self.context_len, expected_channels, self.patch_size, self.patch_size),
            "x_context",
            path,
        )

        # ======================================================================
        # 7. FUTURE FEATURES CONSTRUCTION
        # ======================================================================
        # Climate (ERA5) -> x_fut_era5_1d: (C_era5, T_target) -> x_fut_era5: (C_era5, T_target, H, W)
        if len(self.era5_vars) > 0:
            x_fut_era5_1d = torch.from_numpy(
                ds_target[self.era5_vars].to_array().values
            ).float()
            self._check_shape(
                x_fut_era5_1d,
                (len(self.era5_vars), self.target_len),
                "x_fut_era5_1d",
                path,
            )
            x_fut_era5 = broadcast_era5(x_fut_era5_1d, self.patch_size, self.patch_size)
        else:
            x_fut_era5 = torch.empty(
                (0, self.target_len, self.patch_size, self.patch_size)
            )
        self._check_shape(
            x_fut_era5,
            (len(self.era5_vars), self.target_len, self.patch_size, self.patch_size),
            "x_fut_era5",
            path,
        )

        # Future Static (One-Hot LC + DEM + is_veg)
        lc_fut = torch.from_numpy(ds_target["ESA_LC"].values).long()
        lc_fut_onehot = encode_landcover(lc_fut).permute(
            1, 0, 2, 3
        )  # (12, T_target, H, W)
        if len(self.static_vars) > 0:
            x_stat_fut = torch.from_numpy(
                ds_target[self.static_vars].to_array().values
            ).float()
        else:
            x_stat_fut = torch.empty(
                (0, self.target_len, self.patch_size, self.patch_size)
            )

        # Combine to x_future_feat: (T_target, C_fut, H, W)
        # Concatenate: Climate (C_era5) + ESA One-Hot (12) + Statics (2)
        x_future_feat = torch.cat(
            [x_fut_era5, lc_fut_onehot, x_stat_fut], dim=0
        ).permute(1, 0, 2, 3)
        expected_channels_fut = (
            len(self.era5_vars) + lc_fut_onehot.shape[0] + len(self.static_vars)
        )
        self._check_shape(
            x_future_feat,
            (self.target_len, expected_channels_fut, self.patch_size, self.patch_size),
            "x_future_feat",
            path,
        )

        # ======================================================================
        # 8. TARGET & LOSS MASK
        # ======================================================================
        # y_target: (T_target, 1, H, W)
        y_target = torch.from_numpy(ds_target["kNDVI"].values).unsqueeze(1).float()
        self._check_shape(
            y_target,
            (self.target_len, 1, self.patch_size, self.patch_size),
            "y_target",
            path,
        )

        # target_mask: (T_target, 1, H, W)
        is_veg_target = torch.from_numpy(ds_target["is_veg"].values).float()
        target_mask = (
            torch.from_numpy(ds_target["target_mask"].values).float() * is_veg_target
        ).unsqueeze(1)
        self._check_shape(
            target_mask,
            (self.target_len, 1, self.patch_size, self.patch_size),
            "target_mask",
            path,
        )

        # ======================================================================
        # 9. BASELINE LOGIC (Last-Frame)
        # ======================================================================
        context_kndvi = ds_ctx.kNDVI.values
        mask_s2 = ds_ctx.mask_s2.values

        last_available = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        found = np.zeros((self.patch_size, self.patch_size), dtype=bool)

        # Loop through context timesteps in reverse to find the last valid kNDVI pixel
        for t in reversed(range(context_kndvi.shape[0])):
            valid = (mask_s2[t] == 1) & (~found)
            last_available[valid] = context_kndvi[t][valid]
            found[valid] = True
            if found.all():
                break

        # baseline_sample: (1, H, W)
        baseline_sample = torch.from_numpy(last_available).unsqueeze(0).float()
        self._check_shape(
            baseline_sample,
            (1, self.patch_size, self.patch_size),
            "baseline_sample",
            path,
        )

        # ======================================================================
        # 10. AUGMENTATION (Training only)
        # ======================================================================
        if self.use_augmentation and self.train:
            if torch.rand(1).item() > 0.5:  # Horizontal Flip
                x_context = torch.flip(x_context, dims=[-1])
                x_future_feat = torch.flip(x_future_feat, dims=[-1])
                y_target = torch.flip(y_target, dims=[-1])
                target_mask = torch.flip(target_mask, dims=[-1])
                baseline_sample = torch.flip(baseline_sample, dims=[-1])

            if torch.rand(1).item() > 0.5:  # Vertical Flip
                x_context = torch.flip(x_context, dims=[-2])
                x_future_feat = torch.flip(x_future_feat, dims=[-2])
                y_target = torch.flip(y_target, dims=[-2])
                target_mask = torch.flip(target_mask, dims=[-2])
                baseline_sample = torch.flip(baseline_sample, dims=[-2])

        # NAN checks for critical tensors
        if torch.isnan(x_context).any():
            raise ValueError(f"NaN values detected in x_context. Path: {path}")
        if torch.isnan(x_future_feat).any():
            raise ValueError(f"NaN values detected in x_future_feat. Path: {path}")
        if torch.isnan(y_target).any():
            raise ValueError(f"NaN values detected in y_target. Path: {path}")

        ds.close()

        # Save tiles
        meta = {"top": top, "left": left, "path": path, "cube_id": cube_id}

        return x_context, x_future_feat, y_target, target_mask, meta, baseline_sample


# --- Implementation of Leave-Time-and-Region-Out CV ---
def get_llto_splits(root_dir, csv_path="train_test_split.csv", k=3, show=False):
    """
    Implements k-fold Leave-Location-and-Time-Out (LLTO) splitting according to
    https://doi.org/10.1016/j.envsoft.2017.12.001.
    Groups data by Koppen-Geiger region (Location) and Phenological Season (Time)
    and ensures that Region+Season group are never split between train and val.

    Args:
        root_dir: Directory containing the .zarr cubes.
        csv_path: Path to the metadata CSV.
        k: Number of folds (e.g., 3, 4, or 5).

    Returns:
        List of tuples: [(train_paths, val_paths), ...] for k folds.
    """
    # 1. Load and filter only the training split
    df = pd.read_csv(csv_path)
    df = df[df["split"] == "train"].copy()

    # 2. Map file paths
    processed_files = {
        f.replace("_postprocessed.zarr", ""): os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(".zarr")
    }
    df["full_path"] = df["DisNo."].map(processed_files)

    # Drop entries not found on disk
    initial_count = len(df)
    df = df.dropna(subset=["full_path"])
    print(f"Matched {len(df)}/{initial_count} cubes from CSV to disk.")

    # 3. Create the unique 'Location-Time' Group ID
    df["llto_group"] = (
        df["koppen_geiger"].astype(str) + "_" + df["pheno_season_name"].astype(str)
    )
    unique_groups = sorted(df["llto_group"].unique())
    group_to_idx = {grp: i for i, grp in enumerate(unique_groups)}

    # 4. Initialize GroupKFold
    gkf = GroupKFold(n_splits=k)
    cv_splits = []
    vis_matrix = np.zeros((k, len(unique_groups)))

    indices = np.arange(len(df))
    groups = df["llto_group"].values

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        cv_splits.append((train_df["full_path"].tolist(), val_df["full_path"].tolist()))

        # Fill visualization matrix: 1 for Train (Blue), 2 for Val (Red)
        for grp in train_df["llto_group"].unique():
            vis_matrix[fold_idx, group_to_idx[grp]] = 1
        for grp in val_df["llto_group"].unique():
            vis_matrix[fold_idx, group_to_idx[grp]] = 2

        print(f"Fold {fold_idx}:")
        print(
            f"  > Training:   {len(train_df):3d} cubes ({(len(train_df)/len(df)*100):5.1f}%)"
        )
        print(
            f"  > Validation: {len(val_df):3d} cubes ({(len(val_df)/len(df)*100):5.1f}%)"
        )
        print(
            f"  > Excluded:   {(len(df) - len(train_df) - len(val_df)):3d} cubes ({(len(df) - len(train_df) - len(val_df))/len(df)*100:5.1f}%) [Blocked due to Loc/Time overlap]"
        )

    print(f"Successfully created {k} folds using LLTO GroupKFold.")

    if show:
        _plot_llto_strategy(vis_matrix, unique_groups, k)

    return cv_splits


def get_llto_splits_strict(
    root_dir,
    csv_path="train_test_split.csv",
    k=3,
    min_val_ratio=0.15,
    show=False,
    save_path=None,
):
    """
    Implements LLTO-CV ensuring a minimum validation size while reporting data usage.

    This strategy strictly follows the principle of leaving out both locations
    and time units to test model generalization.

    Args:
        root_dir (str): Directory containing the 1000x1000 .zarr cubes.
        csv_path (str): Metadata file path.
        k (int): Number of folds.
        min_val_ratio (float): Minimum percentage of total cubes for validation (e.g., 0.15).
        show (bool): If True, generates the strategy visualization plot.

    Returns:
        list: [(train_paths, val_paths), ...] for k folds.
    """
    # 1. Loading and Path Mapping
    df = pd.read_csv(csv_path)
    df = df[df["split"] == "train"].copy()

    processed_files = {
        f.replace("_postprocessed.zarr", ""): os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(".zarr")
    }
    df["full_path"] = df["DisNo."].map(processed_files)
    df = df.dropna(subset=["full_path"])

    total_cubes = len(df)
    min_val_cubes = int(total_cubes * min_val_ratio)

    # 2. Identify Groups
    df["llto_group"] = (
        df["koppen_geiger"].astype(str) + "_" + df["pheno_season_name"].astype(str)
    )
    unique_groups = sorted(df["llto_group"].unique())
    all_locations = df["koppen_geiger"].unique()
    all_seasons = df["pheno_season_name"].unique()

    cv_splits = []
    vis_matrix = np.zeros((k, len(unique_groups)))
    group_to_idx = {grp: i for i, grp in enumerate(unique_groups)}

    print(f"--- LLTO-CV Initialization (Total Cubes: {total_cubes}) ---")

    for fold_idx in range(k):
        found_valid_split = False
        attempts = 0

        while not found_valid_split and attempts < 200:
            attempts += 1
            # Randomly sample Location and Season combinations
            sel_locs = random.sample(
                list(all_locations), max(1, len(all_locations) // 4)
            )
            sel_seasons = random.sample(
                list(all_seasons), max(1, len(all_seasons) // 3)
            )

            # Validation: Intersection of selected Location AND Season [cite: 209]
            val_mask = df["koppen_geiger"].isin(sel_locs) & df[
                "pheno_season_name"
            ].isin(sel_seasons)
            val_df = df[val_mask]

            if len(val_df) >= min_val_cubes:
                # Training: Must share NEITHER Location NOR Season with Validation [cite: 130, 209]
                train_mask = (~df["koppen_geiger"].isin(sel_locs)) & (
                    ~df["pheno_season_name"].isin(sel_seasons)
                )
                train_df = df[train_mask]

                if (
                    len(train_df) > total_cubes * 0.1
                ):  # Ensure at least some training data remains
                    found_valid_split = True

        if not found_valid_split:
            raise RuntimeError(
                f"Could not find a valid LLTO split for fold {fold_idx} with current constraints."
            )

        # Calculate Percentages
        train_pct = (len(train_df) / total_cubes) * 100
        val_pct = (len(val_df) / total_cubes) * 100
        excluded_pct = 100 - train_pct - val_pct

        print(f"Fold {fold_idx}:")
        print(f"  > Training:   {len(train_df):3d} cubes ({train_pct:5.1f}%)")
        print(f"  > Validation: {len(val_df):3d} cubes ({val_pct:5.1f}%)")
        print(
            f"  > Excluded:   {(total_cubes - len(train_df) - len(val_df)):3d} cubes ({excluded_pct:5.1f}%) [Blocked due to Loc/Time overlap]"
        )

        cv_splits.append((train_df["full_path"].tolist(), val_df["full_path"].tolist()))

        # 3. Visualization Matrix Update
        for grp in train_df["llto_group"].unique():
            vis_matrix[fold_idx, group_to_idx[grp]] = 1
        for grp in val_df["llto_group"].unique():
            vis_matrix[fold_idx, group_to_idx[grp]] = 2

    if show:
        _plot_llto_strategy(vis_matrix, unique_groups, k, save_path=save_path)

    assert_disjoint_val_sets(cv_splits)

    return cv_splits


def _plot_llto_strategy(vis_matrix, groups, k, save_path=None):
    """Generates a target-oriented cross-validation matrix plot."""
    plt.figure(figsize=(16, 7))
    cmap = ["#bdc3c7", "#3498db", "#e74c3c"]  # Grey, Blue, Red
    sns.heatmap(
        vis_matrix,
        annot=False,
        cbar=False,
        cmap=cmap,
        vmin=0,
        vmax=2,
        linewidths=1,
        linecolor="white",
    )

    plt.title(
        f"Target-Oriented LLTO-CV Strategy ({k} Folds)\n"
        f"Testing Generalization on Unknown Locations and Time Steps"
    )
    plt.xlabel("Spatio-Temporal Groups (KoppenGeiger_Season)")
    plt.ylabel("Fold Index")

    legend_elements = [
        Patch(facecolor="#3498db", label="Used for Training"),
        Patch(facecolor="#e74c3c", label="Used for Validation"),
        Patch(facecolor="#bdc3c7", label="Excluded (Spatial/Temporal Overlap)"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.25, 1))

    if len(groups) < 40:
        plt.xticks(np.arange(len(groups)) + 0.5, groups, rotation=90, fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        out_path = os.path.join(save_path, "cv_splits_plot.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"✅ CV Split Plot gespeichert: {out_path}")
    else:
        plt.show()

    plt.close()


def broadcast_era5(era5_tensor, target_h, target_w):
    """
    Broadcasts a (C, T) era5 tensor to (C, T, H, W).
    """
    # era5_tensor: (Channels, Time)
    # Result: (Channels, Time, H, W)
    return era5_tensor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, target_h, target_w)


def get_val_tiles_auto(cube_paths, patch_size=256, dim_max=1000):
    """
    Generates a deterministic grid of tiles to cover 1000x1000 cubes.

    The grid is calculated once and applied to all provided cube paths to
    ensure consistent validation coverage across the dataset.

    Args:
        cube_paths (list): List of strings containing paths to .zarr files.
        patch_size (int): Spatial size of the patches (e.g., 256).
        dim_max (int): Total size of the cube (default 1000).

    Returns:
        list: List of dicts with 'path', 'top', and 'left' for each tile.
    """
    # 1. Calculate grid coordinates ONCE (valid for all cubes)
    num_tiles = int(np.ceil(dim_max / patch_size))

    if num_tiles > 1:
        stride = (dim_max - patch_size) // (num_tiles - 1)
    else:
        stride = 0

    # Generate the starting points for the patches
    offsets = [i * stride for i in range(num_tiles)]

    # Final safety check: Adjust the last offset to hit the edge exactly
    if len(offsets) > 0 and (offsets[-1] + patch_size) != dim_max:
        offsets[-1] = dim_max - patch_size

    # 2. Combine offsets with paths
    tiled_list = [
        {"path": path, "top": top, "left": left}
        for path in cube_paths
        for top in offsets
        for left in offsets
    ]

    print(f"Validation Strategy: Created {len(offsets)**2} tiles per cube.")
    print(
        f"Grid: {num_tiles}x{num_tiles} patches, Stride: {stride}, Total Tiles: {len(tiled_list)}"
    )

    return tiled_list


def encode_landcover(lc_tensor):
    """
    Transforms ESA Landcover into One-Hot Encoding.
    Input: (T, H, W) or (H, W)
    Output: (T, 12, H, W)
    """
    labels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

    unique_vals = torch.unique(lc_tensor)
    valid_labels = torch.tensor(labels, dtype=lc_tensor.dtype, device=lc_tensor.device)
    invalid_vals = unique_vals[~torch.isin(unique_vals, valid_labels)]
    if invalid_vals.numel() > 0:
        raise ValueError(
            f"encode_landcover received invalid ESA_LC labels: {invalid_vals.tolist()}. "
            f"Allowed labels are: {labels}"
        )

    mapping = torch.zeros(101, dtype=torch.long)
    for i, val in enumerate(labels):
        mapping[val] = i

    lc_mapped = mapping[lc_tensor]
    # One-Hot and permute to (T, C, H, W)
    lc_onehot = F.one_hot(lc_mapped, num_classes=len(labels))

    if lc_onehot.ndim == 4:  # (T, H, W, C)
        return lc_onehot.permute(0, 3, 1, 2).float()
    else:  # (H, W, C) -> (C, H, W)
        return lc_onehot.permute(2, 0, 1).float()


def assert_disjoint_val_sets(cv_splits):
    val_sets = [set(val) for _, val in cv_splits]

    # Pairwise overlap check
    for i in range(len(val_sets)):
        for j in range(i + 1, len(val_sets)):
            overlap = val_sets[i] & val_sets[j]
            if overlap:
                raise ValueError(
                    f"Val overlap between fold {i} and {j}: {len(overlap)} files"
                )

    print("OK: All validation sets are pairwise disjoint.")
