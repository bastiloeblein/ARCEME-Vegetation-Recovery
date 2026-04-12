import torch
import pandas as pd
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import torch.nn.functional as F
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
        config,
        exclude_file=None,  # usually is dealt with already during the split (kept here as safety net)
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
        self.cube_paths = self._filter_cube_paths(
            cube_paths, exclude_file
        )  # filter bad cubes
        self.context_len = context_length
        self.target_len = target_length
        self.patch_size = patch_size
        self.cfg = config
        self.train = train
        self.fixed_tiles = fixed_tiles
        self.use_augmentation = use_augmentation

        # Variable configuration
        self.s2_vars = s2_vars or []
        self.s1_vars = s1_vars or []
        self.era5_vars = era5_vars or []
        self.static_vars = static_vars or []

        # Cube Height and Width
        self.h = 1000
        self.w = 1000

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

        For training, the length is calculated to ensure full spatial coverage
        of the 1000x1000 cubes based on the patch size. For validation,
        it returns the number of pre-defined fixed tiles.
        """
        if self.fixed_tiles is not None:
            return len(self.fixed_tiles)

        # Safety check: Ensure patch size is not larger than the cube dimensions
        if self.patch_size > self.h or self.patch_size > self.w:
            raise ValueError(
                f"Patch size {self.patch_size} exceeds cube dimensions ({self.h}x{self.w})"
            )

        # Calculate number of tiles needed to cover one dimension
        # e.g., for 1000/128: ceil(7.81) = 8 tiles
        num_tiles = int(np.ceil(self.h / self.patch_size))

        # Total number of patches needed for full spatial coverage per cube
        patches_per_cube = num_tiles * num_tiles

        return len(self.cube_paths) * patches_per_cube

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

    def _filter_cube_paths(self, cube_paths, exclude_list):
        """
        Filters the list of cube paths based on an exclusion file or list.
        """
        pattern = r"2\d{3}-\d{4}-[A-Z]{3}"
        excluded_ids = set()

        # 1. Load excluded IDs from file or list
        if exclude_list is not None:
            if isinstance(exclude_list, str) and exclude_list.endswith(".csv"):
                print(f"DEBUG: Received list with bad cubes at {exclude_list}")
                df_ex = pd.read_csv(exclude_list)
                if "cube_id" in df_ex.columns:
                    excluded_ids = set(
                        df_ex["cube_id"].astype(str).str.strip().tolist()
                    )
            elif isinstance(exclude_list, (list, set)):
                excluded_ids = set([str(i).strip() for i in exclude_list])

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

        print(
            f"DATALOADER:Filter applied: {len(cube_paths)} -> {len(filtered_paths)} Cubes."
        )

        return filtered_paths

    def _get_random_patch_coords(self, h, w, ds, max_retries=15):
        """
        Finds suitable spatial coordinates (top, left) for a patch using rejection sampling.

        It validates patches based on vegetation presence and temporal data availability
        in both context and target windows, using thresholds from the configuration.

        Args:
            h, w (int): Height and width of the source cube (usually 1000, 1000).
            ds (xr.Dataset): The lazy-loaded Xarray dataset of the current cube.
            max_retries (int): Maximum number of random attempts before returning the best found patch.

        Returns:
            tuple: (top, left) coordinate
        """
        best_patch = (0, 0)
        max_score = -1

        # Extract thresholds from config
        # A timestep is "good" if it has more valid pixels than this ratio
        valid_pixel_thresh_ctx = self.cfg["data"]["quality_check"][
            "ctx_min_valid_ratio"
        ]
        valid_pixel_thresh_tgt = self.cfg["data"]["quality_check"][
            "tgt_min_valid_ratio"
        ]

        # Minimum number of "good" timesteps required
        min_ctx_steps = int(
            self.context_len
            * self.cfg["data"]["quality_check"]["ctx_required_fraction"]
        )
        min_tgt_steps = self.cfg["data"]["quality_check"]["tgt_min_timesteps"]

        cutoff_date = pd.to_datetime(ds.attrs["precip_end_date"])

        for i in range(max_retries):
            # 1. Randomly sample coordinates
            top = torch.randint(0, h - self.patch_size, (1,)).item()
            left = torch.randint(0, w - self.patch_size, (1,)).item()

            # 2. Extract masks for the spatial patch (lazy loading)
            patch_mask_s2 = ds["mask_s2"].isel(
                y=slice(top, top + self.patch_size),
                x=slice(left, left + self.patch_size),
            )
            patch_is_veg = ds["is_veg"].isel(
                y=slice(top, top + self.patch_size),
                x=slice(left, left + self.patch_size),
            )

            # 3. Combine cloud mask and vegetation mask
            # Result is 1 only where pixels are visible AND contain vegetation
            valid_mask = patch_mask_s2 * patch_is_veg

            # 4. Temporal slicing (Context and Target)
            m_ctx = (
                valid_mask.sel(time_sentinel_2_l2a=slice(None, cutoff_date))
                .tail(time_sentinel_2_l2a=self.context_len)
                .values
            )
            m_tgt = (
                valid_mask.where(
                    valid_mask.time_sentinel_2_l2a > cutoff_date, drop=True
                )
                .head(time_sentinel_2_l2a=self.target_len)
                .values
            )

            # Ratio of valid pixels per timestep: Shape (T,)
            ctx_valid_per_step = m_ctx.mean(axis=(1, 2))
            tgt_valid_per_step = m_tgt.mean(axis=(1, 2))

            # Count "good" timesteps based on pixel threshold
            n_good_ctx = np.sum(ctx_valid_per_step >= valid_pixel_thresh_ctx)
            n_good_tgt = np.sum(tgt_valid_per_step >= valid_pixel_thresh_tgt)

            # 6. Scoring and Selection
            # Score weights target steps more heavily as they are critical for recovery prediction
            current_score = (n_good_ctx * 1.0) + (n_good_tgt * 5.0)

            # Check if patch meets hard criteria from config (minimum number of good timesteps)
            if n_good_ctx >= min_ctx_steps and n_good_tgt >= min_tgt_steps:
                return top, left

            # Log only if it fails the first time
            cube_id = ds.attrs["cube_id"]
            if i == 0:
                print(
                    f"DATALOADER: Cube {cube_id}: Patch does not match criteria, retrying..."
                )
            elif i % 5 == 0:  # Log every 5th fail to avoid spam
                print(
                    f"DATALOADER: Cube {cube_id}: Still looking for a valid patch (Attempt {i+1}/15)..."
                )

            # Keep track of the best effort if criteria aren't met
            if current_score > max_score:
                max_score = current_score
                best_patch = (top, left)

        # Fallback: Return best effort if no patch met the strict criteria
        print(
            f"DEBUG: Cube {cube_id} - No patch met criteria, returning best effort with score:",
            max_score,
        )
        return best_patch

    def __getitem__(self, idx):
        """
        Loads and processes a single data sample.

        Returns:
            tuple: (x_context, x_future_feat, y_target, target_mask, meta, baseline_sample)
        """
        pattern = r"2\d{3}-\d{4}-[A-Z]{3}"

        # ======================================================================
        # 1. PATCHING STRATEGY
        # ======================================================================
        if not self.train and self.fixed_tiles is not None:
            tile_info = self.fixed_tiles[idx]
            path = tile_info["path"]
            top = tile_info["top"]
            left = tile_info["left"]

        else:
            # Select the cube path and get random patch coordinates later
            real_idx = idx % len(self.cube_paths)
            path = self.cube_paths[real_idx]

        # ======================================================================
        # 2. DATA LOADING & SPATIAL SLICING
        # ======================================================================
        try:
            ds = xr.open_zarr(path, consolidated=True)
        except Exception:
            print(
                f"WARNING: Failed to open {path} with consolidated=True. Retrying with consolidated=False."
            )
            ds = xr.open_zarr(path, consolidated=False)

        # ======================================================================
        # 3. RANDOM PATCHING WITH QUALITY CHECK (Training only)
        # ======================================================================
        if self.train:
            # Pass ds to filter function
            top, left = self._get_random_patch_coords(self.h, self.w, ds)

        # ======================================================================
        # 4. SPATIAL SLICING
        # ======================================================================
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
        # 5. TIME WINDOW SLICING (Context vs Target)
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
        # 6. INPUT CONSTRUCTION (CONTEXT WINDOW)
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

            x_era5_1d = torch.clamp(x_era5_1d, min=-3.0, max=3.0) / 3.0
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
        # 7. MASKING & VEGETATION LOGIC
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
        # 8. LANDCOVER & STATIC FEATURES
        # ======================================================================
        # lc: (T_ctx, H, W) -> lc_onehot: (T_ctx, 12, H, W)
        lc = torch.from_numpy(ds_ctx["ESA_LC"].values).long()
        lc_onehot = encode_landcover(lc)  # Shape: (T_ctx, 12, H, W)
        lc_onehot = lc_onehot.permute(1, 0, 2, 3)  # NOW: (12, T_ctx, 256, 256)
        self._check_shape(
            lc_onehot,
            (12, self.context_len, self.patch_size, self.patch_size),
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
            [lc_onehot, x_stat_raw], dim=0
        )  # Shape x_static: (12 + len(static_vars), T_ctx, 256, 256)
        self._check_shape(
            x_static,
            (
                len(self.static_vars) + lc_onehot.shape[0],
                self.context_len,
                self.patch_size,
                self.patch_size,
            ),
            "x_static",
            path,
        )

        # ======================================================================
        # FINAL CONTEXT STACK (The Symmetrical Order)
        # ======================================================================
        # In order to ensure, that the model can learn a consistent interpretation of channels between context and future features,
        # the features that are present in both (context and future) are at the same tensor positions in the channel dimension.
        # So the model can learn to interpret these channels in the same way for both context and future features.
        # The context only features are then added at the end of the channel dimension.

        # 1. Features that are present in both:
        # kNDVI, ERA5 variables, LC one-hot, Statics
        # x_s2[0:1, ...] -> (1, T_ctx, 256, 256)
        # x_era5         -> (C, T_ctx, 256, 256)
        # x_static       -> (C, T_ctx, 256, 256)
        shared_features = torch.cat([x_s2[0:1, :, :, :], x_era5, x_static], dim=0)
        # Shape: (1 (kNDVI - at first position) + C_era5 + C_static, T_ctx, 256, 256)

        # 2. Features only present in context:
        # the other S2 variables, S1 variables, Masks
        # x_s2[1:, ...]  -> (Remaining_Channels, T_ctx, 256, 256)
        context_only_features = torch.cat([x_s2[1:, :, :, :], x_s1, m_s2, m_s1], dim=0)
        # Shape: (C_s2-1 + C_s1 + 2, T_ctx, 256, 256)

        # 3. Final Stack:
        x_context = torch.cat([shared_features, context_only_features], dim=0).permute(
            1, 0, 2, 3
        )
        # Final Shape x_context: (T_ctx, C_total, 256, 256)
        expected_channels = (
            1  # kNDVI at first position,
            + len(self.era5_vars)
            + len(self.static_vars)
            + lc_onehot.shape[0]  # LC one-hot (12)
            + len(self.s2_vars)
            - 1  # Remaining S2 channels (without kNDVI)
            + len(self.s1_vars)  # S1 channels
            + 2  # Masks (S2, S1)
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
            x_fut_era5_1d = torch.clamp(x_fut_era5_1d, min=-3.0, max=3.0) / 3.0

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
        # Same order as in context, baseline/prediction of kNDVI will the be added within the model to first position
        x_future_feat = torch.cat(
            [x_fut_era5, lc_fut_onehot, x_stat_fut], dim=0
        ).permute(1, 0, 2, 3)
        expected_channels_fut = (
            len(self.era5_vars) + lc_fut_onehot.shape[0] + len(self.static_vars)
        )
        # x_future_feat = self._align_future_to_context_structure(
        #     x_fut_era5, lc_fut_onehot, x_stat_fut, path
        # )
        # # S2, S1, Weather, Masks, LC One-Hot, Statics -> C_total
        # expected_channels_fut = (
        #     len(self.s2_vars) + len(self.s1_vars) + len(self.era5_vars) +
        #     2 + 12 + len(self.static_vars)
        # )
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
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                x_context = torch.rot90(x_context, k, dims=[-2, -1])
                x_future_feat = torch.rot90(x_future_feat, k, dims=[-2, -1])
                y_target = torch.rot90(y_target, k, dims=[-2, -1])
                target_mask = torch.rot90(target_mask, k, dims=[-2, -1])
                baseline_sample = torch.rot90(baseline_sample, k, dims=[-2, -1])

        # NAN checks for critical tensors
        if torch.isnan(x_context).any():
            raise ValueError(f"NaN values detected in x_context. Path: {path}")
        if torch.isnan(x_future_feat).any():
            raise ValueError(f"NaN values detected in x_future_feat. Path: {path}")
        if torch.isnan(y_target).any():
            raise ValueError(f"NaN values detected in y_target. Path: {path}")

        ds.close()

        # Save tiles
        # Get cube_id
        match = re.search(pattern, path)
        cube_id = match.group(0) if match else "No_ID_Found"
        meta = {"top": top, "left": left, "path": path, "cube_id": cube_id}

        # ======================================================================
        # FINAL CROSS-CHECK ASSERTS (Alignment Context vs Future)
        # ======================================================================
        # Wir prüfen einen zufälligen räumlichen Pixel
        rand_y, rand_x = self.patch_size // 2, self.patch_size // 2

        # 1. Check: ERA5 Alignment
        # In x_context liegen ERA5-Variablen ab Index 1 (nach kNDVI)
        # In x_future_feat liegen ERA5-Variablen ab Index 0
        ctx_era5_sample = x_context[
            0, 1, rand_y, rand_x
        ]  # Erster Context-Zeitschritt, erste ERA5 Var
        # Wir holen den echten Wert aus dem urspünglichen x_era5 Tensor zum Vergleich
        orig_era5_sample = x_era5[0, 0, rand_y, rand_x]
        assert torch.allclose(
            ctx_era5_sample, orig_era5_sample
        ), "ERA5 im Context ist falsch positioniert!"

        # 2. Check: Static Alignment (z.B. DEM)
        # DEM ist die erste Variable in self.static_vars
        # Position in x_context: 1 (kNDVI) + len(era5) + 12 (OneHot) + 0 (DEM)
        dem_idx_ctx = 1 + len(self.era5_vars) + 12
        dem_idx_fut = len(self.era5_vars) + 12

        ctx_dem_sample = x_context[0, dem_idx_ctx, rand_y, rand_x]
        fut_dem_sample = x_future_feat[0, dem_idx_fut, rand_y, rand_x]

        assert torch.allclose(
            ctx_dem_sample, fut_dem_sample
        ), f"Statics mismatch! Context Index {dem_idx_ctx} vs Future Index {dem_idx_fut}"

        # 3. Check: LC One-Hot (Klasse 0)
        # Liegt direkt nach ERA5
        lc_idx_ctx = 1 + len(self.era5_vars)
        lc_idx_fut = len(self.era5_vars)

        ctx_lc_sample = x_context[0, lc_idx_ctx, rand_y, rand_x]
        fut_lc_sample = x_future_feat[0, lc_idx_fut, rand_y, rand_x]

        assert torch.allclose(
            ctx_lc_sample, fut_lc_sample
        ), "Landcover One-Hot Alignment fehlerhaft!"

        # 4. Check: kNDVI Consistency
        # kNDVI im Context (Index 0) muss dem letzten Frame der Baseline entsprechen
        # (wenn t = context_len - 1)
        last_ctx_kndvi = x_context[-1, 0, rand_y, rand_x]
        # In y_target ist kNDVI an Index 0 (Channel 0)
        target_kndvi_start = y_target[0, 0, rand_y, rand_x]

        # Hinweis: last_ctx_kndvi und target_kndvi_start müssen nicht gleich sein (Zeit schreitet voran),
        # aber sie sollten im gleichen Wertebereich liegen (Sinnhaftigkeitscheck)
        if not (torch.isfinite(last_ctx_kndvi) and torch.isfinite(target_kndvi_start)):
            raise ValueError(f"Non-finite kNDVI values at {path}")

        return x_context, x_future_feat, y_target, target_mask, meta, baseline_sample

    # def _align_future_to_context_structure(self, x_fut_era5, lc_fut_onehot, x_stat_fut, path):
    #     """
    #     Alignes the future feature tensor to match the channel structure of the context tensor.
    #     So the model does not have to learn to interpret different channel orders between context and future features.
    #     Unknown channels (S1, S2, Masks) are filled with zeros.
    #     """
    #     # 1. Dummies for S2 and S1 channels in the future (as they are unknown at prediction time)
    #     # Form: (Channels, T_target, H, W)
    #     x_s2_dummy = torch.zeros(
    #         (len(self.s2_vars), self.target_len, self.patch_size, self.patch_size),
    #         device=x_fut_era5.device
    #     )
    #     x_s1_dummy = torch.zeros(
    #         (len(self.s1_vars), self.target_len, self.patch_size, self.patch_size),
    #         device=x_fut_era5.device
    #     )

    #     # 2. Dummies for the masks (mask_s2, mask_s1)
    #     m_dummy = torch.zeros(
    #         (2, self.target_len, self.patch_size, self.patch_size),
    #         device=x_fut_era5.device
    #     )

    #     # 3. Prepare static features for the future (lc_onehot + statics) in the same order as context:
    #     # These will be provided for the guided prediction
    #     # lc_fut_onehot is laready (12, T_target, H, W)
    #     # x_stat_fut must be (C_stat, T_target, H, W)
    #     x_static_fut_full = torch.cat([lc_fut_onehot, x_stat_fut], dim=0)

    #     # 4. Built in the exact same channel order as context for the future features:
    #     # Context Order was: [x_s2, x_s1, x_era5, m_s2, m_s1, x_static]
    #     x_future_aligned = torch.cat(
    #         [x_s2_dummy, x_s1_dummy, x_fut_era5, m_dummy, x_static_fut_full],
    #         dim=0
    #     )

    #     # 5. Permute to (T_target, C_total, H, W) as x_context
    #     x_future_aligned = x_future_aligned.permute(1, 0, 2, 3)

    #     # ======================================================================
    #     # EXTENSIVE ASSERTS
    #     # ======================================================================
    #     # Value Checks (Dummies must be 0)
    #     if not torch.all(x_future_aligned[:, :len(self.s2_vars)] == 0):
    #          raise ValueError(f"S2 dummy in future is not zero! Path: {path}")

    #     # NaN Check
    #     if torch.isnan(x_future_aligned).any():
    #         raise ValueError(f"NaNs in aligned future features detected! Path: {path}")

    #     return x_future_aligned


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
