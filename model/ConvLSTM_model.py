import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from model.optimizers import get_opt_from_name
from torch.optim.lr_scheduler import ReduceLROnPlateau
from my_utils.losses import get_loss_from_name
from my_utils.visualization import (
    plot_full_cube_predictions,
    plot_prediction_deltas,
    verify_baseline_consistency,
)
import numpy as np
from collections import defaultdict

# delete
import matplotlib.pyplot as plt


class ConvLSTM_Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # Get copy so number of hidden_channels will not be altered within model (will lead to mismatch in later folds for SGConvLSTM)
        hidden_channels_copy = list(cfg["model"]["hidden_channels"])

        # Get model type from config:
        # SGED: Encoder-Decoder architecture (separate weights)
        # SGConvLSTM: Shared weights with dynamic padding
        model_type = cfg["model"]["model_type"]
        print(f"Building model type: {model_type}")

        # Common parameters shared by both model variants
        model_params = {
            "input_dim": cfg["model"][
                "input_channels"
            ],  #  Context variables (z.B. S1 + S2 + Wetter + Statics = 34)
            "output_dim": cfg["model"][
                "output_channels"
            ],  # Just 1 (as we only predict kNDVI)
            "hidden_dims": hidden_channels_copy,
            "num_layers": cfg["model"]["n_layers"],
            "cfg": cfg,
            "kernel_size": cfg["model"]["kernel"],
            "dilation": cfg["model"]["dilation_rate"],
            "baseline": cfg["model"]["baseline"],
            "dropout_prob": cfg["model"].get("dropout_prob", 0.0),
            "layer_norm_flag": cfg["model"].get("layer_norm", False),
        }

        # Initialize the specific model class
        if "SGED" in model_type:
            from model.ConvLSTM import SGEDConvLSTM

            # Safety check for SGED: Hidden dims list must match number of layers
            assert (
                len(model_params["hidden_dims"]) == model_params["num_layers"]
            ), f"SGED requires hidden_dims list length ({len(model_params['hidden_dims'])}) to match n_layers ({model_params['num_layers']})"
            # SGED needs an explicit decoder input dimension calculation
            decoder_input_channels = cfg["model"]["future_channels"]
            self.model = SGEDConvLSTM(
                decoder_input_dim=decoder_input_channels, **model_params
            )
        else:
            from model.ConvLSTM import SGConvLSTM

            # Safety check for SGConvLSTM:
            assert (
                len(model_params["hidden_dims"]) == model_params["num_layers"]
            ), f"SGConvLSTM requires hidden_dims list length ({len(model_params['hidden_dims'])}) to match n_layers ({model_params['num_layers']})"
            self.model = SGConvLSTM(**model_params)

        # 3. 3. Loss & Optimizer configuration
        self.lr = cfg["training"]["optimizer"]["start_learn_rate"]
        self.training_loss = get_loss_from_name(
            cfg["training"]["training_loss"]["loss_function"]
        )
        self.alpha = cfg["training"]["training_loss"]["alpha"]
        self.beta = cfg["training"]["training_loss"]["beta"]
        self.gamma = cfg["training"]["training_loss"]["gamma"]
        self.scaling_factor = cfg["training"]["scaling_factor"]

        # Internal storage for validation metrics
        self.validation_step_outputs = []

    def forward(self, x_ctx, prediction_count, non_pred_feat, baseline_sample):
        """
        Standard forward pass through the selected model.
        """
        preds, pred_deltas, baselines = self.model(
            x_ctx,
            non_pred_feat=non_pred_feat,
            prediction_count=prediction_count,
            baseline_sample=baseline_sample,
        )

        return preds, pred_deltas, baselines

    def training_step(self, batch, batch_idx):
        """
        Main training logic. Processes context data and predicts future frames.
        Batch structure: (x_ctx, x_fut, y_true, mask, meta, baseline_sample)
        """
        x_ctx, x_fut, y_true, mask, meta, baseline_sample = batch

        # Get batch size and number of prediction steps
        bs = x_ctx.shape[0]
        t_fut = y_true.size(1)

        # --- CRITICAL ASSERTS ---
        assert (
            x_ctx.dim() == 5
        ), f"Expected 5D input (B, T, C, H, W), got {x_ctx.dim()}D"
        assert x_ctx.size(0) == y_true.size(
            0
        ), "Batch size mismatch between input and target!"
        assert y_true.size(2) == 1, "Target kNDVI must have exactly 1 channel!"
        assert (
            mask.size(1) == t_fut
        ), "Mask and Target must have identical temporal length!"
        assert (
            x_fut.size(1) == t_fut
        ), "Future features time dim must match prediction count!"
        p_size = self.cfg["data"]["patch_size"]
        assert all(
            t.size(-1) == p_size and t.size(-2) == p_size
            for t in [x_ctx, x_fut, y_true, mask, baseline_sample]
        ), f"All spatial dimensions (H, W) must be {p_size}x{p_size}!"

        # Model Prediction
        y_pred, y_delta_pred, baselines = self(x_ctx, t_fut, x_fut, baseline_sample)

        # --- 1. Primary Loss ---
        # predicted kNDVI vs true kNDVI
        train_loss = self.training_loss(preds=y_pred, targets=y_true, mask=mask)

        # Initialize secondary losses with zeros to safely use them in combined_loss later
        train_delta_loss = torch.tensor(0.0).to(self.device)
        grad_loss = torch.tensor(0.0).to(self.device)

        # --- 2. Delta & Gradient Losses (OPTIONAL:Only calculate if beta OR gamma is active) ---
        if self.beta > 0 or self.gamma > 0:

            # Create mask-sequence: [mask_context_last, mask_true_1, mask_true_2...]
            # As we assume that last context frame always has valid pixels,
            # we create a mask with only 1 for first step
            mask_ctx_last = torch.ones_like(baseline_sample)
            full_mask_seq = torch.cat([mask_ctx_last.unsqueeze(1), mask], dim=1)

            # The delta mask is only 1 where NOW and BEFORE were 1, because we can only calculate a delta where we have valid pixels in both timesteps.
            # mask_delta has shape [B, T_fut, 1, H, W]
            mask_delta = full_mask_seq[:, 1:] * full_mask_seq[:, :-1]

            # Create GT sequence, starting with the last context frame
            # [baseline, gt_1, gt_2, gt_3...]
            gt_with_context = torch.cat([baseline_sample.unsqueeze(1), y_true], dim=1)

            # Now calculate the true physical differences betwwen (T and T -1)
            y_delta_true = torch.diff(gt_with_context, dim=1)
            # y_delta_true = y_true - baselines

            # 2a. Temporal Delta Loss (Beta)
            if self.beta > 0:
                train_delta_loss = self.training_loss(
                    preds=y_delta_pred * self.scaling_factor,
                    targets=y_delta_true * self.scaling_factor,
                    mask=mask_delta,
                )
                self.log(
                    "train_delta_loss",
                    train_delta_loss,
                    on_epoch=True,
                    batch_size=bs,
                    sync_dist=True,
                )

            # 2b. Spatial Gradient Loss (Gamma)
            if self.gamma > 0:
                # X-Directional (Horizontal)
                diff_x_pred = torch.diff(y_delta_pred, dim=-1)
                diff_x_true = torch.diff(y_delta_true, dim=-1)
                mask_x = mask_delta[
                    :, :, :, :, :-1
                ]  # Maske um 1 Pixel kürzen, da diff die Breite um 1 reduziert
                grad_loss_x = torch.sum(
                    torch.abs(diff_x_pred - diff_x_true) * mask_x
                ) / (mask_x.sum() + 1e-8)

                # Y-Directional (Vertical)
                diff_y_pred = torch.diff(y_delta_pred, dim=-2)
                diff_y_true = torch.diff(y_delta_true, dim=-2)
                # Korrekt für 5D [B, T, C, H, W]:
                mask_y = mask_delta[:, :, :, :-1, :]
                grad_loss_y = torch.sum(
                    torch.abs(diff_y_pred - diff_y_true) * mask_y
                ) / (mask_y.sum() + 1e-8)

                # Combine Spatial Loss
                grad_loss = grad_loss_x + grad_loss_y
                self.log(
                    "train_grad_only_loss",
                    grad_loss,
                    on_epoch=True,
                    batch_size=bs,
                    sync_dist=True,
                )

        # --- 3. Combined Final Loss ---
        # Multiply each loss by its config weight
        combined_loss = (
            (self.alpha * train_loss)
            + (self.beta * train_delta_loss)
            + (self.gamma * grad_loss)
        )

        # --- Logging ---
        self.log(
            "train_loss",
            train_loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=bs,
            sync_dist=True,
        )
        if self.gamma > 0 or self.beta > 0:
            self.log(
                "combined_loss",
                combined_loss,
                on_epoch=True,
                batch_size=bs,
                sync_dist=True,
            )
            if self.beta > 0:
                self.log(
                    "loss_ratio_delta_to_main",
                    train_delta_loss / (train_loss + 1e-8),
                    on_epoch=True,
                    batch_size=bs,
                    sync_dist=True,
                )

            if self.gamma > 0:
                self.log(
                    "loss_ratio_grad_to_main",
                    grad_loss / (train_loss + 1e-8),
                    on_epoch=True,
                    batch_size=bs,
                    sync_dist=True,
                )

        return combined_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation logic: Performs the forward pass and collects raw predictions and
        ground truth patches for later stitching and metric calculation on the CPU.
        """
        x_ctx, x_fut, y_true, mask, meta, baseline_sample = batch
        bs, t_fut = x_ctx.shape[0], y_true.shape[1]

        # Forward pass
        y_pred, y_delta_pred, baselines = self(x_ctx, t_fut, x_fut, baseline_sample)

        # Create static persistence baseline [B, T, 1, H, W]
        # baseline_sample is [B, 1, H, W] -> convert to [B, T, 1, H, W] to facilitate metric calculation with y_true and mask
        # Copies the basleine T times. So calculation of Baseline at timestep t - GT at t can be calculated
        persistence_baseline = baseline_sample.unsqueeze(1).repeat(1, t_fut, 1, 1, 1)

        # Extract metadata
        cube_ids = meta["cube_id"]
        tops = meta["top"]
        lefts = meta["left"]

        # Store raw outputs for stitching at on_validation_epoch_end.
        # Moving tensors to CPU prevents CUDA Out-Of-Memory errors.
        for i in range(bs):
            self.validation_step_outputs.append(
                {
                    "cube_id": cube_ids[i],
                    "top": tops[i].item(),
                    "left": lefts[i].item(),
                    "y_pred": y_pred[i].detach().cpu(),  # Shape: [T, 1, H, W]
                    "y_true": y_true[i].detach().cpu(),  # Shape: [T, 1, H, W]
                    "mask": mask[i].detach().cpu(),  # Shape: [T, 1, H, W]
                    "baseline": persistence_baseline[i]
                    .detach()
                    .cpu(),  # Shape: [T, 1, H, W]
                }
            )

        # --- Visualizations (Keep your existing code here) ---
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            # Log histograms every 5 epochs to visualize prediction and ground truth deltas
            # log_delta_histograms(self, y_pred, y_true, baselines, mask, batch_idx, num_samples=5)

            # Save high quality batch for visualization over the epochs
            # fix that
            # store_fixed_val_samples(self, x_ctx, x_fut, y_true, mask, meta, baseline_sample, pixels, number_of_samples=5)

            # Check if baseline consistency holds for this batch
            verify_baseline_consistency(self, meta, baseline_sample, batch_idx)

    def on_validation_epoch_end(self):
        """
        Executed at the end of the epoch. When model has seen all patches for each cube.
        Stitches patches back into full 1000x1000 cubes, averages overlapping predictions,
        and calculates pixel-weighted metrics per timestep and per cube.
        """
        if not self.validation_step_outputs:
            return

        # 1. Group patches by cube_id
        # test this in notebook!
        cubes_data = defaultdict(list)
        for output in self.validation_step_outputs:
            cubes_data[output["cube_id"]].append(output)

        patch_size = self.cfg["data"]["patch_size"]
        dim_max = 1000  # Cube spatial dimension
        # Calculate expected number of patches based on patch_size and cube dims
        expected_patches = int(np.ceil(dim_max / patch_size)) ** 2

        # Flag to save tensors during testing/evaluation
        # Only when testing
        try:
            is_sanity = self.trainer.sanity_checking
        except RuntimeError:
            is_sanity = False
        save_tensors = not is_sanity and self.cfg.get("testing", {}).get(
            "save_tensors", False
        )

        # --- Global Storage for Metrics ---
        # Timestep-wise storage (to see how error evolves over time T)
        t_fut = self.validation_step_outputs[0]["y_pred"].shape[0]
        # Global Storage for Pixel-wise (Micro) metrics across all cubes
        global_ts_stats = {
            "sq_err": torch.zeros(t_fut),
            "abs_err": torch.zeros(t_fut),
            "pixels": torch.zeros(t_fut),
            "sq_err_base": torch.zeros(t_fut),
            "abs_err_base": torch.zeros(t_fut),
            "bias": torch.zeros(t_fut),
            "bias_base": torch.zeros(t_fut),
            "y_true": torch.zeros(t_fut),
            "y_true_sq": torch.zeros(t_fut),
            "y_pred": torch.zeros(t_fut),
            "y_pred_base": torch.zeros(t_fut),
        }

        # Storage for Cube-wise (Macro) metrics
        all_cube_metrics = []

        # 2. Iterate and Stitch each Cube
        for cube_id, patches in cubes_data.items():

            # --- SAFETY CHECK ---
            if len(patches) != expected_patches:
                print(
                    f"⚠️ WARNING: Cube {cube_id} has {len(patches)} patches, expected {expected_patches}. Stitching might be incomplete!"
                )

            # Initialize empty tensors for the full 1000x1000 cube
            # Shape: [T, H, W] (squeeze the channel dimension since C=1)
            pred_cube = torch.zeros((t_fut, dim_max, dim_max), dtype=torch.float32)
            count_cube = torch.zeros((t_fut, dim_max, dim_max), dtype=torch.float32)

            true_cube = torch.zeros((t_fut, dim_max, dim_max), dtype=torch.float32)
            mask_cube = torch.zeros((t_fut, dim_max, dim_max), dtype=torch.float32)
            base_cube = torch.zeros((t_fut, dim_max, dim_max), dtype=torch.float32)

            # Stitching: Place patches into their spatial position
            for p in patches:
                top, left = p["top"], p["left"]
                bottom, right = top + patch_size, left + patch_size

                # Assert we don't go out of bounds
                assert (
                    bottom <= dim_max and right <= dim_max
                ), f"Patch bounds ({bottom}, {right}) exceed cube max ({dim_max})"

                # Squeeze channel dim for easier handling: [T, 1, H, W] -> [T, H, W]
                p_pred = p["y_pred"].squeeze(1)
                p_true = p["y_true"].squeeze(1)
                p_mask = p["mask"].squeeze(1)
                p_base = p["baseline"].squeeze(1)

                # Accumulate predictions and count overlaps
                pred_cube[:, top:bottom, left:right] += p_pred
                count_cube[:, top:bottom, left:right] += 1.0

                # Overwrite true, mask, and base (they are identical in overlapping regions, no need to average)
                true_cube[:, top:bottom, left:right] = p_true
                mask_cube[:, top:bottom, left:right] = p_mask
                base_cube[:, top:bottom, left:right] = p_base

            # Average the overlapping predictions  (clamp to prevent division by zero, but this should not occur)
            if count_cube.min() == 0:
                print(
                    f"⚠️ WARNING: Cube {cube_id} has uncovered pixels (count=0). This should not happen if patches fully cover the cube."
                )
            pred_cube = pred_cube / count_cube.clamp(min=1.0)

            # Create a boolean mask for safe indexing
            # Check if mask == 1, we use > 0.5 because of floating point inaccurcies, that can arise when sending tensors to CPU and doing the stitching with additions and divisions. So we want to consider a pixel as valid if the mask value is greater than 0.5, which effectively means it was marked as valid in the original data before any floating point issues.
            valid_mask = mask_cube > 0.5

            # --- FULL CUBE PLOTTING (WITH HIGH QUALITY FILTER) ---
            # 1. Initialize list with HQ cube ids
            if not hasattr(self, "fixed_plot_cube_ids"):
                self.fixed_plot_cube_ids = []

            # 2. Calcualte quality of cubes
            cube_quality_ratio = valid_mask.sum().item() / valid_mask.numel()
            should_plot = False

            # 3. In Epoch 0: Get quality of all cubes
            if self.current_epoch == 0:
                # 3.1 Save quality ratio for all cubes in epoch 0
                if not hasattr(self, "cube_scouting_scores"):
                    self.cube_scouting_scores = {}
                self.cube_scouting_scores[cube_id] = cube_quality_ratio
            else:
                # 3.2 PLOTTING: From Epoch 5 plot 5 best cubes
                if (
                    hasattr(self, "fixed_plot_cube_ids")
                    and cube_id in self.fixed_plot_cube_ids
                ):
                    if self.current_epoch % 5 == 0:
                        should_plot = True

            # 4. Plotting
            if should_plot:
                # Plot full cube
                plot_full_cube_predictions(
                    true_cube=true_cube,
                    pred_cube=pred_cube,
                    base_cube=base_cube,
                    mask_cube=mask_cube,
                    cube_id=cube_id,
                    epoch=self.current_epoch,
                    save_path=os.path.join(self.cfg["model"]["run_dir"], "plots"),
                    logger=self.logger,
                )

            # --- TESTING ONLY: Save Tensors to Disk ---
            # and dims [T, H, W]
            if save_tensors:
                import xarray as xr

                save_dir = os.path.join(self.cfg["model"]["run_dir"], "tensors")
                os.makedirs(save_dir, exist_ok=True)

                # Create an xarray Dataset for this cube
                ds_out = xr.Dataset(
                    {
                        "pred": (["time", "y", "x"], pred_cube.cpu().numpy()),
                        "true": (["time", "y", "x"], true_cube.cpu().numpy()),
                        "mask": (["time", "y", "x"], mask_cube.cpu().numpy()),
                        "base": (["time", "y", "x"], base_cube.cpu().numpy()),
                    }
                )
                # Als Zarr speichern
                zarr_path = os.path.join(save_dir, f"{cube_id}.zarr")
                ds_out.to_zarr(zarr_path, mode="w")

            # 3. Calculate Metrics per Timestep for THIS Cube
            # Prediction Metrics
            cube_ts_sq_err, cube_ts_abs_err, cube_ts_bias = (
                torch.zeros(t_fut),
                torch.zeros(t_fut),
                torch.zeros(t_fut),
            )
            # Baseline Metrics
            cube_ts_sq_err_base, cube_ts_abs_err_base, cube_ts_bias_base = (
                torch.zeros(t_fut),
                torch.zeros(t_fut),
                torch.zeros(t_fut),
            )
            # Valid Pixels
            cube_ts_pixels = torch.zeros(t_fut)
            # Tensors necessary for R2 and NNSE calculation
            cube_ts_y_t, cube_ts_y_t_sq = torch.zeros(t_fut), torch.zeros(t_fut)

            for t in range(t_fut):
                m_t = valid_mask[t]
                pixels_t = m_t.sum().item()

                if pixels_t > 0:
                    y_pred_t = pred_cube[t][m_t]
                    y_true_t = true_cube[t][m_t]
                    y_base_t = base_cube[t][m_t]

                    # Model errors
                    err = y_pred_t - y_true_t
                    cube_ts_sq_err[t] = (err**2).sum()
                    cube_ts_abs_err[t] = torch.abs(err).sum()
                    cube_ts_bias[t] = err.sum()

                    # Baseline errors
                    err_b = y_base_t - y_true_t
                    cube_ts_sq_err_base[t] = (err_b**2).sum()
                    cube_ts_abs_err_base[t] = torch.abs(err_b).sum()
                    cube_ts_bias_base[t] = err_b.sum()

                    # Stats for R2 and NNSE
                    cube_ts_pixels[t] = pixels_t
                    cube_ts_y_t[t] = y_true_t.sum()
                    cube_ts_y_t_sq[t] = (y_true_t**2).sum()

                    # Add to global timestep stats (Micro)
                    global_ts_stats["sq_err"][t] += cube_ts_sq_err[t]
                    global_ts_stats["abs_err"][t] += cube_ts_abs_err[t]
                    global_ts_stats["bias"][t] += cube_ts_bias[t]

                    global_ts_stats["sq_err_base"][t] += cube_ts_sq_err_base[t]
                    global_ts_stats["abs_err_base"][t] += cube_ts_abs_err_base[t]
                    global_ts_stats["bias_base"][t] += cube_ts_bias_base[t]

                    global_ts_stats["pixels"][t] += pixels_t
                    global_ts_stats["y_true"][t] += cube_ts_y_t[t]
                    global_ts_stats["y_true_sq"][t] += cube_ts_y_t_sq[t]

            # Calculate Macro Metrics for this specific Cube (aggregated over all T)
            # (Total squared error across all timesteps / Total valid pixels across all timesteps)
            # Pixel based weithing (each pixel has same weight) - if one timestep has only few pixels, it will not bias the overall metric for the cube
            total_cube_pixels = cube_ts_pixels.sum().item()
            if (
                total_cube_pixels > 0
            ):  # Could add min_pixel_threshold here to filter out cubes with too little valid data
                # Model Metrics
                cube_mse = cube_ts_sq_err.sum().item() / total_cube_pixels
                cube_mae = cube_ts_abs_err.sum().item() / total_cube_pixels
                cube_bias = cube_ts_bias.sum().item() / total_cube_pixels

                # Baseline Metrics
                cube_mse_base = cube_ts_sq_err_base.sum().item() / total_cube_pixels
                cube_mae_base = cube_ts_abs_err_base.sum().item() / total_cube_pixels
                cube_bias_base = cube_ts_bias_base.sum().item() / total_cube_pixels

                cube_skill = 1 - (cube_mse / (cube_mse_base + 1e-8))

                # Variance metrics
                # Variance/R2 per Cube
                # --- R² CALCULATION VIA VARIANCE DECOMPOSITION ---
                # R² = 1 - (SS_res / SS_tot)
                #
                # Since we cannot store all pixels, we use the "Sum of Squares Identity":
                # SS_tot = Σ(y_true - mean_y_true)²  =>  Σ(y_true²) - ( (Σ y_true)² / n )
                #
                # 1. sq_err: Residual Sum of Squares (SS_res)  (Σ (y_pred - y_true)²)
                # 2. y_true_sq_sum:   Sum of squared raw values (Σ y_true²)
                # 3. y_true_sum:      Sum of raw values (Σ y_true)
                # 4. pixels:          Total pixel count (n)
                cube_sst = cube_ts_y_t_sq.sum().item() - (
                    (cube_ts_y_t.sum().item() ** 2) / total_cube_pixels
                )
                cube_r2 = 1 - (cube_ts_sq_err.sum().item() / (cube_sst + 1e-8))
                cube_r2_base = 1 - (
                    cube_ts_sq_err_base.sum().item() / (cube_sst + 1e-8)
                )
                # Formula from Pellicer-Valero et al. (2025) to convert R² to NNSE: NNSE = 1 / (2 - R²)
                cube_nnse = 1 / (2 - cube_r2)
                cube_nnse_base = 1 / (2 - cube_r2_base)

                all_cube_metrics.append(
                    {
                        "id": cube_id,
                        "mse": cube_mse,
                        "mae": cube_mae,
                        "bias": cube_bias,
                        "r2": cube_r2,
                        "nnse": cube_nnse,
                        "mse_base": cube_mse_base,
                        "mae_base": cube_mae_base,
                        "bias_base": cube_bias_base,
                        "r2_base": cube_r2_base,
                        "nnse_base": cube_nnse_base,
                        "cube_skill": cube_skill,
                    }
                )

        # --- 4. Final Aggregation and Logging ---
        metrics_to_log = {}

        # Log Timestep-wise Micro metrics across all cubes
        # Every pixel has the same weight, so if a cube has only few pixels in timestep 3, it will not bias the overall metric for that timestep.
        log_ts = self.cfg["training"]["validation"].get("log_timestep_metrics", False)
        if log_ts:
            for t in range(t_fut):
                p_t = global_ts_stats["pixels"][t].item()
                if p_t > 0:
                    mse_t = global_ts_stats["sq_err"][t].item() / p_t
                    mse_base_t = global_ts_stats["sq_err_base"][t].item() / p_t

                    sst_t = global_ts_stats["y_true_sq"][t].item() - (
                        (global_ts_stats["y_true"][t].item() ** 2) / p_t
                    )
                    r2_t = 1 - (global_ts_stats["sq_err"][t].item() / (sst_t + 1e-8))
                    r2_base_t = 1 - (mse_base_t * p_t / (sst_t + 1e-8))

                    skill_t = 1 - (mse_t / (mse_base_t + 1e-8))

                    metrics_to_log[f"val/step_{t}/MSE_pixel"] = mse_t
                    metrics_to_log[f"val/step_{t}/MSE_pixel_base"] = mse_base_t
                    metrics_to_log[f"val/step_{t}/R2_pixel"] = r2_t
                    metrics_to_log[f"val/step_{t}/R2_pixel_base"] = r2_base_t
                    metrics_to_log[f"val/step_{t}/NNSE_pixel"] = 1 / (2 - r2_t)
                    metrics_to_log[f"val/step_{t}/NNSE_pixel_base"] = 1 / (
                        2 - r2_base_t
                    )
                    metrics_to_log[f"val/step_{t}/Skill_Score_pixel"] = skill_t

        # Grand Mean Micro (Pixel-weighted over ALL cubes and ALL timesteps)
        # Each valid pixel has the same weight
        # Answers the question: "How well performs the model on an average pixel across all of earth and all timesteps"
        # Careful: Bias towards cubes with many valid pixels in later timesteps, as they will dominate the metric if they have more valid pixels than other cubes (e.g. tropical regions with many invalid pixels could be underrepresented, while desert areas with many valid pixels could dominate the metric)
        g_pix = global_ts_stats["pixels"].sum().item()
        if g_pix > 0:
            g_mse = global_ts_stats["sq_err"].sum().item() / g_pix
            g_mae = global_ts_stats["abs_err"].sum().item() / g_pix
            g_bias = global_ts_stats["bias"].sum().item() / g_pix

            g_mse_base = global_ts_stats["sq_err_base"].sum().item() / g_pix
            g_mae_base = global_ts_stats["abs_err_base"].sum().item() / g_pix
            g_bias_base = global_ts_stats["bias_base"].sum().item() / g_pix

            g_skill = 1 - (g_mse / (g_mse_base + 1e-8))

            g_sst = global_ts_stats["y_true_sq"].sum().item() - (
                (global_ts_stats["y_true"].sum().item() ** 2) / g_pix
            )
            g_r2 = 1 - (global_ts_stats["sq_err"].sum().item() / (g_sst + 1e-8))
            g_r2_base = 1 - (
                global_ts_stats["sq_err_base"].sum().item() / (g_sst + 1e-8)
            )

            metrics_to_log["val/grand_mean_micro/MSE"] = g_mse
            metrics_to_log["val/grand_mean_micro/MSE_base"] = g_mse_base
            metrics_to_log["val/grand_mean_micro/MAE"] = g_mae
            metrics_to_log["val/grand_mean_micro/MAE_base"] = g_mae_base
            metrics_to_log["val/grand_mean_micro/Bias"] = g_bias
            metrics_to_log["val/grand_mean_micro/Bias_base"] = g_bias_base
            metrics_to_log["val/grand_mean_micro/R2"] = g_r2
            metrics_to_log["val/grand_mean_micro/R2_base"] = g_r2_base
            metrics_to_log["val/grand_mean_micro/NNSE"] = 1 / (2 - g_r2)
            metrics_to_log["val/grand_mean_micro/NNSE_base"] = 1 / (2 - g_r2_base)
            metrics_to_log["val/grand_mean_micro/Skill_Score"] = g_skill

        # Grand Mean Macro (Cube-weighted average)
        # Here each cube has the same weight (independend of how many valid pixels it has), so cubes with few pixels are not dominated by cubes with many pixels,
        # Answers the question: "How well perfomrs the model on an average region on earth"
        # Tropical regions with many invalid pixels could be underrepresented in the micro metrics (as the have fewer valid pixels than dessert areas)
        if all_cube_metrics:
            n_c = len(all_cube_metrics)
            metrics_to_log["val/grand_mean_macro/MSE"] = (
                sum(c["mse"] for c in all_cube_metrics) / n_c
            )
            metrics_to_log["val/grand_mean_macro/MSE_base"] = (
                sum(c["mse_base"] for c in all_cube_metrics) / n_c
            )
            metrics_to_log["val/grand_mean_macro/MAE"] = (
                sum(c["mae"] for c in all_cube_metrics) / n_c
            )
            metrics_to_log["val/grand_mean_macro/Bias"] = (
                sum(c["bias"] for c in all_cube_metrics) / n_c
            )
            metrics_to_log["val/grand_mean_macro/R2"] = (
                sum(c["r2"] for c in all_cube_metrics) / n_c
            )
            metrics_to_log["val/grand_mean_macro/R2_base"] = (
                sum(c["r2_base"] for c in all_cube_metrics) / n_c
            )
            metrics_to_log["val/grand_mean_macro/NNSE"] = (
                sum(c["nnse"] for c in all_cube_metrics) / n_c
            )
            metrics_to_log["val/grand_mean_macro/NNSE_base"] = (
                sum(c["nnse_base"] for c in all_cube_metrics) / n_c
            )
            metrics_to_log["val/grand_mean_macro/Skill_Score"] = (
                sum(c["cube_skill"] for c in all_cube_metrics) / n_c
            )

        self.log_dict(metrics_to_log, sync_dist=True)
        self.validation_step_outputs.clear()

        # --- Plotting - After Epoch 0: Select 5 best cubes ---
        if self.current_epoch == 0 and hasattr(self, "cube_scouting_scores"):
            # Sort descending by quality score
            sorted_cubes = sorted(
                self.cube_scouting_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            # Extract Top 5 cube IDs for fixed plotting in future epochs
            self.fixed_plot_cube_ids = [cid for cid, _ in sorted_cubes[:5]]

            # Free memory
            del self.cube_scouting_scores

        # --- TESTING ONLY: Save Metrics ---
        has_trainer = getattr(self, "_trainer", None) is not None
        save_metrics = (
            not is_sanity and self.cfg.get("testing", {}).get("save_metrics", False)
            if has_trainer
            else False
        )

        if save_metrics:
            import pandas as pd
            import json

            metrics_dir = os.path.join(self.cfg["model"]["run_dir"], "metrics")
            os.makedirs(metrics_dir, exist_ok=True)

            # 1. Save cube-wise Metrics as CSV
            if all_cube_metrics:
                df_cubes = pd.DataFrame(all_cube_metrics)
                csv_path = os.path.join(metrics_dir, "cube_metrics.csv")
                df_cubes.to_csv(csv_path, index=False)

            # 2. Save aggregated Grand Means and Timestep Metrics as JSON
            json_path = os.path.join(metrics_dir, "global_metrics.json")
            with open(json_path, "w") as f:
                json.dump(metrics_to_log, f, indent=4)

    def log_fixed_validation_samples(self, log_interval=5):
        """
        Logs qualitative prediction visualizations for fixed validation samples.
        Executed every few epochs to monitor model behavior over time.
        """
        if self.logger is not None:
            if not hasattr(self, "fixed_val_batches"):
                return

            if self.current_epoch % log_interval != 0:
                return

            for i, val_batch in enumerate(self.fixed_val_batches):

                # Move batch to device
                batch_cuda = [
                    t.to(self.device) if torch.is_tensor(t) else t for t in val_batch
                ]

                x_ctx, x_fut, y_true, mask, meta, baseline_sample = batch_cuda

                # Optional: visualize hidden states
                with torch.no_grad():
                    self.visualize_and_log_hidden(batch_cuda)

                    y_pred, y_delta_pred, baselines = self(
                        x_ctx, y_true.size(1), x_fut, baseline_sample
                    )

                # Create visualization
                fig = plot_prediction_deltas(
                    y_true,
                    y_pred,
                    y_delta_pred,
                    baselines,
                    mask,
                    batch_idx=i,
                    epoch=self.current_epoch,
                    save_path="plots",
                )

                cube_id = meta["cube_id"][0]

                # Logging
                if isinstance(self.logger, WandbLogger):
                    self.logger.experiment.log(
                        {
                            f"Fixed_Samples/Patch_{i}_Cube_{cube_id}": fig,
                            "epoch": self.current_epoch,
                        }
                    )
                else:
                    self.logger.experiment.add_figure(
                        f"Visuals/Cube_{cube_id}_Deltas_Epoch_{self.current_epoch}",
                        fig,
                        global_step=self.current_epoch,
                    )

                # Prevent memory leaks
                plt.close(fig)

    def configure_optimizers(self):
        """
        Setup optimizer and learning rate scheduler.
        """
        optimizer = get_opt_from_name(
            self.cfg["training"]["optimizer"]["name"],
            params=self.parameters(),
            lr=self.cfg["training"]["optimizer"]["start_learn_rate"],
        )

        # Monitor metric that should defines validation performance
        # monitor_key = f"{self.cfg['training']['validation']['monitor']['split']}_{self.cfg['training']['validation']['monitor']['metric']}"
        monitor_key = self.cfg["training"]["validation"]["monitor"]["metric"]
        # Plateau scheduler
        plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=self.cfg["training"]["validation"]["monitor_mode"],
            factor=self.cfg["training"]["optimizer"]["lr_factor"],
            patience=self.cfg["training"]["optimizer"]["patience"],
            threshold=self.cfg["training"]["optimizer"]["lr_threshold"],
        )

        # # Get warmup parameters from config
        # warmup_cfg = self.cfg["training"]["optimizer"].get("warmup", {})
        # if warmup_cfg.get("enabled", False):
        #     warmup_epochs = warmup_cfg.get("epochs", 3)
        #     # Warmup scheduler: for first epochs climbs from 0.01 to 100% of the target LR
        #     warmup_scheduler = LinearLR(
        #         optimizer,
        #         start_factor=0.01,  # Startet bei 1% der Ziel-LR
        #         end_factor=1.0,
        #         total_iters=warmup_epochs,
        #     )

        #     # Combine warmup and plateau schedulers sequentially
        #     # combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     #     optimizer,
        #     #     schedulers=[warmup_scheduler, plateau_scheduler],
        #     #     milestones=[warmup_epochs]
        #     # )

        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, plateau_scheduler]),
        #             "monitor": monitor_key,
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": plateau_scheduler,
                "monitor": monitor_key,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def visualize_hidden_states(self, x_ctx, x_fut, baseline_sample):
        self.model.eval()
        with torch.no_grad():

            activations = {}

            def get_activation(name):
                def hook(model, input, output):
                    # output bei ConvLSTM ist oft (hidden_state, cell_state)
                    if isinstance(output, tuple):
                        activations[name] = output[0].detach()
                    else:
                        activations[name] = output.detach()

                return hook

            # Dynamische Layer-Auswahl
            if (
                hasattr(self.model, "encoder_cells")
                and len(self.model.encoder_cells) > 0
            ):
                target_layer = self.model.encoder_cells[-2]
            elif hasattr(self.model, "cell_list") and len(self.model.cell_list) > 0:
                target_layer = self.model.cell_list[-2]
            else:
                # Fallback auf das Modell selbst, falls keine Liste gefunden wird
                target_layer = self.model

            handle = target_layer.register_forward_hook(get_activation("last_hidden"))

            # Forward Pass
            _ = self(x_ctx, x_fut.size(1), x_fut, baseline_sample)
            handle.remove()

            hidden = activations.get("last_hidden")
            if hidden is None:
                print("WARNING: Could not capture hidden states.")
                return None

            # hidden shape ist (B, C, H, W)
            total_channels = hidden.shape[1]
            # Wir wollen genau 8 Plots (für dein 2x4 Layout)
            num_to_plot = min(8, total_channels)

            fig, axes = plt.subplots(2, 4, figsize=(15, 8))
            axes_flat = axes.flatten()

            # num_layers_available = hidden.shape[1] if torch.is_tensor(hidden) else len(hidden)

            # Visualisierung der ersten 8 Hidden Channels des ersten Batch-Samples
            for i in range(8):
                ax = axes_flat[i]

                # Nur plotten, wenn der Channel existiert
                if i < num_to_plot:
                    h_img = hidden[0, i].cpu().numpy()
                    im = ax.imshow(h_img, cmap="viridis")
                    ax.set_title(f"Channel {i}\nMax: {h_img.max():.4f}")
                    plt.colorbar(im, ax=ax)
                else:
                    # Deaktiviere Achsen für leere Plots (verhindert den Fehler)
                    ax.axis("off")
                    ax.set_title("N/A")

            plt.tight_layout()

            os.makedirs("plots", exist_ok=True)
            fig.savefig(f"plots/hidden_states_epoch_{self.current_epoch}.png")
            plt.close(fig)

            return fig

    def visualize_and_log_hidden(self, batch):

        if self.logger is not None:
            x_ctx, x_fut, y_true, mask, meta, baseline_sample = batch

            # Deine Funktion von oben aufrufen
            fig = self.visualize_hidden_states(x_ctx, x_fut, baseline_sample)

            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(
                    {
                        "Debug/Hidden_States": fig,
                        "epoch": self.current_epoch,
                        "global_step": self.global_step,
                    }
                )
            else:
                self.logger.experiment.add_figure(
                    f"Debug/Hidden_States_Epoch_{self.current_epoch}",
                    fig,
                    global_step=self.current_epoch,
                )
            plt.close(fig)

    def on_after_backward(self):
        if self.logger is not None:
            # Diese Methode wird nach jedem Gradienten-Schritt aufgerufen
            if (
                self.global_step % 50 == 0
            ):  # Alle 10 Schritte loggen, um TB nicht zu fluten
                # In deiner on_after_backward Methode:
                grad_dict = {}
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        # Nur den Mittelwert der absoluten Gradienten loggen
                        grad_dict[f"grad_norms/{name}"] = param.grad.abs().mean().item()
                self.logger.experiment.log(grad_dict, commit=False)

    def on_before_optimizer_step(self, optimizer):

        if self.logger is not None:
            if self.global_step % 50 == 0:
                grad_dict = {}
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        grad_dict[f"grad_clipped/{name}"] = (
                            param.grad.abs().mean().item()
                        )
                self.logger.experiment.log(grad_dict, commit=False)
                # if isinstance(self.logger, WandbLogger):
                #     for name, param in self.named_parameters():
                #         if param.grad is not None:
                #             self.logger.experiment.log(
                #                 {
                #                     f"gradients/{name}": wandb.Histogram(
                #                         param.grad.cpu().detach().numpy()
                #                     )
                #                 },
                #                 commit=False,
                #             )
                # else:
                #     for name, param in self.named_parameters():
                #         if param.grad is not None:
                #             self.logger.experiment.add_histogram(
                #                 f"Gradients/{name}", param.grad, self.global_step
                #             )
                #             self.logger.experiment.add_histogram(
                #                 f"Weights/{name}", param.data, self.global_step
                #             )
