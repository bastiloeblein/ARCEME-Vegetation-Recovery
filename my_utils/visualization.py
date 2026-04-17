import matplotlib.pyplot as plt
import numpy as np
import os
import wandb
import torch


def plot_full_cube_predictions(
    true_cube,
    pred_cube,
    base_cube,
    mask_cube,
    is_veg_cube,
    cube_id,
    epoch,
    save_path=None,
    logger=None,
):
    """
    Plots a comparison between Ground Truth and Prediction including Deltas for FULL CUBES.
    Inputs are expected to be Tensors or Numpy arrays of shape (T, H, W).
    """
    # Detach and convert to numpy if they are tensors
    if hasattr(true_cube, "cpu"):
        y_true_raw = true_cube.detach().cpu().numpy()
        y_pred_raw = pred_cube.detach().cpu().numpy()
        baselines_raw = base_cube.detach().cpu().numpy()
        mask_np = mask_cube.detach().cpu().numpy()
    else:
        y_true_raw, y_pred_raw, baselines_raw, mask_np = (
            true_cube,
            pred_cube,
            base_cube,
            mask_cube,
        )

    # Ensure is_veg is numpy
    if hasattr(is_veg_cube, "cpu"):
        is_veg_np = is_veg_cube.detach().cpu().numpy()
    else:
        is_veg_np = is_veg_cube

    t_steps = y_true_raw.shape[0]

    # Create the static Vegetation mask (1000x1000)
    m_veg = is_veg_np > 0.5

    # Initialize figure
    fig, axes = plt.subplots(6, t_steps, figsize=(t_steps * 3, 18))
    if t_steps == 1:
        axes = axes.reshape(6, 1)

    for t in range(t_steps):
        # Masking logic
        m = mask_np[t] > 0.5  # Valid pixels mask for this timestep

        # --- 1. Masking for Plots ---
        # GT: Show only where we have valid data (cloud-free & veg)
        gt_plot = np.where(~m, np.nan, y_true_raw[t])

        # Prediction: Show ALL vegetated pixels (masking of non-veg pixels)
        pred_plot = np.where(~m_veg, np.nan, y_pred_raw[t])

        # --- 2. Delta Calculations ---
        # GT Delta vs Baseline
        d_gt_vs_base = np.where(~m, np.nan, y_true_raw[t] - baselines_raw[t])

        # GT Delta Physical (Real change in nature: T - (T-1))
        if t == 0:
            d_gt_phys = np.where(~m, np.nan, y_true_raw[0] - baselines_raw[0])
            m_delta = m
        else:
            m_prev = mask_np[t - 1] > 0.5
            m_delta = m & m_prev
            d_gt_phys = np.where(~m_delta, np.nan, y_true_raw[t] - y_true_raw[t - 1])

        # Pred Delta (Prediction - Baseline): Show for all vegetation pixels, but mask non-veg areas
        pred_delta_plot = np.where(~m_veg, np.nan, y_pred_raw[t] - baselines_raw[t])

        # Error (Prediction - Ground Truth): Only possible where GT is valid
        error_plot = np.where(~m, np.nan, y_pred_raw[t] - y_true_raw[t])

        # --- 3. Calculate FAIR Means ---
        # Use boolean indexing to extract only the relevant valid pixels for the mean
        avg_d_phys = (
            (y_true_raw[t] - (y_true_raw[t - 1] if t > 0 else baselines_raw[0]))[
                m_delta
            ].mean()
            if m_delta.any()
            else 0
        )
        avg_d_vs_base = (y_true_raw[t] - baselines_raw[t])[m].mean() if m.any() else 0
        avg_d_error = (y_pred_raw[t] - y_true_raw[t])[m].mean() if m.any() else 0

        # Fair Prediction Average: Calculated over ALL vegetation pixels!
        avg_d_pred = (
            (y_pred_raw[t] - baselines_raw[t])[m_veg].mean() if m_veg.any() else 0
        )

        # --- 4. Plotting ---
        axes[0, t].imshow(gt_plot, vmin=0, vmax=1, cmap="YlGn")
        axes[0, t].set_title(f"GT T{t}")

        axes[1, t].imshow(pred_plot, vmin=0, vmax=1, cmap="YlGn")
        axes[1, t].set_title(f"Pred T{t}")

        axes[2, t].imshow(d_gt_vs_base, vmin=-0.2, vmax=0.2, cmap="RdYlGn")
        axes[2, t].set_title(f"GT Delta (vs Base)\n(avg: {avg_d_vs_base:.3f})")

        axes[3, t].imshow(d_gt_phys, vmin=-0.2, vmax=0.2, cmap="RdYlGn")
        axes[3, t].set_title(f"GT Delta Phys\n(avg: {avg_d_phys:.3f})")

        axes[4, t].imshow(pred_delta_plot, vmin=-0.2, vmax=0.2, cmap="RdYlGn")
        axes[4, t].set_title(f"Pred Delta\n(avg: {avg_d_pred:.3f})")

        im_err = axes[5, t].imshow(error_plot, vmin=-0.2, vmax=0.2, cmap="coolwarm")
        axes[5, t].set_title(f"Error (Pred-GT)\n(avg: {avg_d_error:.3f})")

    # Remove axis ticks
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.subplots_adjust(
        right=0.88, left=0.05, top=0.9, bottom=0.1, wspace=0.1, hspace=0.3
    )

    # Colorbars
    cbar_ax_abs = fig.add_axes([0.88, 0.72, 0.015, 0.20])
    fig.colorbar(axes[0, 0].get_images()[0], cax=cbar_ax_abs, label="kNDVI Abs")

    cbar_ax_delta = fig.add_axes([0.88, 0.28, 0.015, 0.35])
    fig.colorbar(axes[2, 0].get_images()[0], cax=cbar_ax_delta, label="Delta")

    cbar_ax_err = fig.add_axes([0.88, 0.10, 0.015, 0.10])
    fig.colorbar(im_err, cax=cbar_ax_err, label="Error")

    # Save to disk if requested
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/epoch_{epoch}_{cube_id}.png", dpi=150)

    # Log directly to W&B if logger is provided
    if logger is not None:
        try:
            logger.experiment.log(
                {
                    f"Fixed_Samples/Full_Cube_{cube_id}": fig,
                    "epoch": epoch,
                }
            )
        except Exception as e:
            print(f"Failed to log image to W&B: {e}")

    plt.close(fig)


def plot_prediction_deltas(
    y_true, y_pred, y_delta_pred, baselines, mask, batch_idx, epoch, save_path=None
):
    """
    Plots a comparison between Ground Truth and Prediction including Deltas.
    y_true, y_pred, baselines: Tensors of shape (B, T, 1, H, W)
    mask: Tensor of shape (B, T, 1, H, W), where 0 = masked/invalid, 1 = valid
    """
    # Use the first sample in the batch
    b = 0
    t_steps = y_true.size(1)

    # Detach and convert to numpy

    y_true_raw = y_true.detach().cpu().numpy()
    y_pred_raw = y_pred.detach().cpu().numpy()
    y_delta_pred_raw = y_delta_pred.detach().cpu().numpy()
    baselines_raw = baselines.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    # Initialize figure - Now 6 rows to fit everything
    fig, axes = plt.subplots(6, t_steps, figsize=(t_steps * 3, 18))
    if t_steps == 1:
        axes = axes.reshape(6, 1)

    for t in range(t_steps):

        # Masking logic
        m = mask_np[b, t, 0] == 1  # Valid pixels mask for this sample and timestep
        # GT only where valid
        gt_plot = np.where(~m, np.nan, y_true_raw[b, t, 0])
        # Prediction full (as requested)
        pred_plot = y_pred_raw[b, t, 0]

        # 2. Delta Calculations
        # GT Delta vs Baseline (Model Target)
        d_gt_vs_base = np.where(
            ~m, np.nan, y_true_raw[b, t, 0] - baselines_raw[b, t, 0]
        )

        # GT Delta Physical (Real change in nature: T - (T-1))
        if t == 0:
            # For the first step, we use the very first baseline (context)
            d_gt_phys = np.where(
                ~m, np.nan, y_true_raw[b, 0, 0] - baselines_raw[b, 0, 0]
            )
        else:
            # Mask for previous timestep
            m_prev = mask_np[b, t - 1, 0] == 1

            # Pixel must be valid in BOTH timesteps
            m_delta = m & m_prev

            # Previous GT is already masked, so we use raw values for calculation but mask result
            d_gt_phys = np.where(
                ~m_delta, np.nan, y_true_raw[b, t, 0] - y_true_raw[b, t - 1, 0]
            )

        pred_delta_plot = y_delta_pred_raw[b, t, 0]
        error_plot = np.where(~m, np.nan, y_pred_raw[b, t, 0] - y_true_raw[b, t, 0])

        # Calculate Mean Delta for valid pixels
        avg_d_phys = d_gt_phys[m].mean() if m.any() else 0
        avg_d_vs_base = d_gt_vs_base[m].mean() if m.any() else 0
        avg_d_pred = pred_delta_plot[m].mean() if m.any() else 0
        avg_d_error = error_plot[m].mean() if m.any() else 0

        # --- Plotting ---
        # Row 0: GT (Masked)
        axes[0, t].imshow(gt_plot, vmin=0, vmax=1, cmap="YlGn")
        axes[0, t].set_title(f"GT T{t}")

        # Row 1: Pred (Full)
        axes[1, t].imshow(pred_plot, vmin=0, vmax=1, cmap="YlGn")
        axes[1, t].set_title(f"Pred T{t}")

        # Row 2: GT Delta vs Baseline
        axes[2, t].imshow(d_gt_vs_base, vmin=-0.2, vmax=0.2, cmap="RdYlGn")
        axes[2, t].set_title(f"GT Delta (vs Base) \n (avg: {avg_d_vs_base:.3f})")

        # Row 3: GT Delta Physical (T - T-1)
        axes[3, t].imshow(d_gt_phys, vmin=-0.2, vmax=0.2, cmap="RdYlGn")
        axes[3, t].set_title(f"GT Delta Phys \n (avg: {avg_d_phys:.3f})")

        # Row 4: Pred Delta
        axes[4, t].imshow(pred_delta_plot, vmin=-0.2, vmax=0.2, cmap="RdYlGn")
        axes[4, t].set_title(f"Pred Delta \n (avg: {avg_d_pred:.3f})")

        # Row 5: Error Map (Pred - GT)
        im_err = axes[5, t].imshow(error_plot, vmin=-0.2, vmax=0.2, cmap="coolwarm")
        axes[5, t].set_title(f"Error (Pred-GT) \n (avg: {avg_d_error:.3f})")

    # Remove axis ticks for a cleaner look
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

    # Adjust layout and add colorbars
    fig.subplots_adjust(
        right=0.88, left=0.05, top=0.9, bottom=0.1, wspace=0.1, hspace=0.3
    )

    # Absolute kNDVI colorbar (Row 1 & 2)
    cbar_ax_abs = fig.add_axes([0.88, 0.72, 0.015, 0.20])
    fig.colorbar(axes[0, 0].get_images()[0], cax=cbar_ax_abs, label="kNDVI Abs")

    cbar_ax_delta = fig.add_axes([0.88, 0.28, 0.015, 0.35])
    fig.colorbar(axes[2, 0].get_images()[0], cax=cbar_ax_delta, label="Delta")

    cbar_ax_err = fig.add_axes([0.88, 0.10, 0.015, 0.10])
    fig.colorbar(im_err, cax=cbar_ax_err, label="Error")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/epoch_{epoch}_batch_{batch_idx}.png")
        plt.close()

    return fig


def log_delta_histograms(
    self, y_pred, y_true, baselines, mask, batch_idx, num_samples=5
):
    """
    Plots histograms of predicted vs. ground truth deltas for each timestep.
    Only logs every 5 epochs for the first batch.
    """
    if self.logger is None:
        return

    # Only every 5 epochs and for the first batch to avoid cluttering the logs
    if self.current_epoch % 5 != 0 or batch_idx != 0:
        return

    t_fut = y_true.size(1)
    bs_to_plot = min(y_pred.size(0), num_samples)

    for i in range(bs_to_plot):
        for t in range(t_fut):
            # Calculate ground truth delta: For t=0, use baseline
            if t == 0:
                # Difference from last known image (context) to the first target
                gt_delta_phys = y_true[i, 0] - baselines[i, 0]
                pred_delta_phys = y_pred[i, 0] - baselines[i, 0]
            else:
                # Difference between previous and current target timesteps
                gt_delta_phys = y_true[i, t] - y_true[i, t - 1]
                pred_delta_phys = y_pred[i, t] - y_pred[i, t - 1]

            # Mask for this specific patch and timestep
            m = mask[i, t].bool()

            if m.any():
                t_flat = gt_delta_phys[m].detach().cpu().numpy()
                p_flat = pred_delta_phys[m].detach().cpu().numpy()

                self.logger.experiment.log(
                    {
                        f"hist/step_{t}_GT_phys_delta": wandb.Histogram(t_flat),
                        f"hist/step_{t}_Pred_phys_delta": wandb.Histogram(p_flat),
                    },
                    commit=False,
                )


def store_fixed_val_samples(
    self,
    x_ctx,
    x_fut,
    y_true,
    mask,
    meta,
    baseline_sample,
    valid_pixels,
    number_of_samples=5,
):
    """
    Selects and stores a few high-quality samples during the first epoch
    to be used for consistent visualization across the entire training.
    """
    # We only need to collect samples during the first epoch
    if self.current_epoch != 0:
        return

    # Initialize storage if it doesn't exist
    if not hasattr(self, "fixed_val_batches"):
        self.fixed_val_batches = []
        self.seen_cube_ids = set()

    if len(self.fixed_val_batches) < number_of_samples:
        bs = x_ctx.shape[0]
        for i in range(bs):
            cid = meta["cube_id"][i]
            total_pixels = mask[i].numel()
            num_valid = valid_pixels[i].sum().item()
            quality_ratio = num_valid / total_pixels

            # Store only new cubes that meet the quality threshold (more than 65% valid pixels)
            if cid not in self.seen_cube_ids and quality_ratio > 0.65:
                # We move data to CPU and clone to prevent memory leaks and keep them for later epochs
                safe_sample = [
                    x_ctx[i : i + 1].detach().cpu().clone(),  # Context
                    x_fut[i : i + 1].detach().cpu().clone(),  # Future features
                    y_true[i : i + 1].detach().cpu().clone(),  # Ground Truth
                    mask[i : i + 1].detach().cpu().clone(),  # Valid Mask
                    {
                        k: [v[i]] for k, v in meta.items()
                    },  # Meta data as single-entry dict
                    baseline_sample[i : i + 1]
                    .detach()
                    .cpu()
                    .clone(),  # Persistence anchor
                ]

                self.fixed_val_batches.append(safe_sample)
                self.seen_cube_ids.add(cid)

                print(
                    f"DEBUG: Stored fixed visualization sample {len(self.fixed_val_batches)} for Cube ID: {cid}"
                )

                # Stop if we have collected enough samples
                if len(self.fixed_val_batches) >= number_of_samples:
                    break


def verify_baseline_consistency(self, meta, baseline_sample, batch_idx):
    """
    Ensures that the data loaded in the current epoch matches the anchor samples
    from Epoch 0. Prevents evaluation on drifted or incorrectly augmented data.
    """
    # Only possible if we have stored anchors and are past the first epoch
    if not hasattr(self, "fixed_val_batches") or self.current_epoch == 0:
        return

    bs = baseline_sample.shape[0]

    for i in range(bs):
        current_id = meta["cube_id"][i]
        current_top = meta["top"][i]
        current_left = meta["left"][i]

        for anchor_sample in self.fixed_val_batches:
            # anchor_sample structure: [x_ctx, x_fut, y_true, mask, meta, baseline_sample]
            anchor_meta = anchor_sample[4]

            # Find the matching patch within the anchor batch
            for j in range(len(anchor_meta["cube_id"])):
                if (
                    current_id == anchor_meta["cube_id"][j]
                    and current_top == anchor_meta["top"][j]
                    and current_left == anchor_meta["left"][j]
                ):

                    # Match found! Compare current data with stored anchor
                    stored_baseline = anchor_sample[-1][j].to(self.device)
                    current_baseline = baseline_sample[i]

                    diff = torch.abs(current_baseline - stored_baseline)
                    max_diff = diff.max().item()

                    # Trigger error handling only if drift is detected
                    if max_diff > 1e-5:
                        self._log_drift_to_wandb(
                            current_id,
                            diff,
                            stored_baseline,
                            current_baseline,
                            max_diff,
                        )

                        raise AssertionError(
                            f"DATA DRIFT DETECTED: Patch {current_id} (Top: {current_top}, Left: {current_left}) "
                            f"has changed since Epoch 0. Max Diff: {max_diff:.6f}. Check WandB for details!"
                        )


def _log_drift_to_wandb(self, cube_id, diff, stored, current, max_diff):
    """Helper to log debug images only when a drift occurs."""
    self.logger.experiment.log(
        {
            f"debug/drift_map_{cube_id}": wandb.Image(
                diff.squeeze().cpu().numpy(),
                caption=f"Drift Map (Max: {max_diff:.6f}) Ep {self.current_epoch}",
            ),
            f"debug/baseline_comparison_{cube_id}": [
                wandb.Image(stored.squeeze().cpu().numpy(), caption="Stored (Epoch 0)"),
                wandb.Image(
                    current.squeeze().cpu().numpy(),
                    caption=f"Current (Epoch {self.current_epoch})",
                ),
            ],
        }
    )
