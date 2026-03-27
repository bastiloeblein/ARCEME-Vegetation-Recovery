import matplotlib.pyplot as plt
import numpy as np
import os


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
    delta_true_raw = (y_true - baselines).detach().cpu().numpy()
    delta_pred_raw = (y_delta_pred).detach().cpu().numpy()
    y_true_raw = y_true.detach().cpu().numpy()
    y_pred_raw = y_pred.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()  # (B, T, 1, H, W)

    # Prepare arrays for plotting by applying the mask
    # Set masked pixels (mask == 0) to NaN
    y_true_plot = np.where(mask_np == 0, np.nan, y_true_raw)[b, :, 0]
    y_pred_plot = y_pred_raw[b, :, 0]
    delta_true_plot = np.where(mask_np == 0, np.nan, delta_true_raw)[b, :, 0]
    delta_pred_plot = delta_pred_raw[b, :, 0]

    # Initialize figure
    fig, axes = plt.subplots(4, t_steps, figsize=(t_steps * 3, 12))

    # If only 1 timestep, axes becomes 1D, force it back to 2D for consistency
    if t_steps == 1:
        axes = axes.reshape(4, 1)

    for t in range(t_steps):
        # Row 1: Ground Truth kNDVI (Masked pixels = white)
        axes[0, t].imshow(y_true_plot[t], vmin=0, vmax=1, cmap="YlGn")
        axes[0, t].set_title(f"GT T{t+1}")

        # Row 2: Predicted kNDVI
        axes[1, t].imshow(y_pred_plot[t], vmin=0, vmax=1, cmap="YlGn")
        axes[1, t].set_title(f"Pred T{t+1}")

        # Row 3: True Delta (Masked pixels = white)
        axes[2, t].imshow(delta_true_plot[t], vmin=-0.2, vmax=0.2, cmap="RdYlGn")
        axes[2, t].set_title(f"True Delta T{t}")

        # Row 4: Predicted Delta
        im_delta_pred = axes[3, t].imshow(
            delta_pred_plot[t], vmin=-0.2, vmax=0.2, cmap="RdYlGn"
        )
        axes[3, t].set_title(f"Pred Delta T{t}")

    # Remove axis ticks for a cleaner look
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust layout and add colorbars
    fig.subplots_adjust(
        right=0.88, left=0.05, top=0.9, bottom=0.1, wspace=0.1, hspace=0.3
    )

    # Absolute kNDVI colorbar (Row 1 & 2)
    cbar_ax_abs = fig.add_axes([0.91, 0.55, 0.015, 0.35])
    fig.colorbar(axes[0, 0].get_images()[0], cax=cbar_ax_abs, label="kNDVI Abs")

    # Delta kNDVI colorbar (Row 3 & 4)
    cbar_ax_delta = fig.add_axes([0.91, 0.1, 0.015, 0.35])
    fig.colorbar(im_delta_pred, cax=cbar_ax_delta, label="kNDVI Delta")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/epoch_{epoch}_batch_{batch_idx}.png")
        plt.close()

    return fig
