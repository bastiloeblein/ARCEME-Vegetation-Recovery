import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_prediction_deltas(y_true, y_pred, y_delta_pred, baselines, mask, batch_idx, epoch, save_path=None):
    """
    Plottet den Vergleich zwischen Ground Truth und Prediction inklusive der Deltas.
    y_true, y_pred, baselines: Tensors of shape (B, T, 1, H, W)
    """
    # Wir plotten nur das erste Sample aus dem Batch (Index 0)
    b = 0 
    t_steps = y_true.size(1)
    
    # Deltas berechnen
    delta_true = (y_true - baselines).detach().cpu().numpy()
    delta_pred = (y_delta_pred).detach().cpu().numpy() 
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy() # (B, T, 1, H, W)

    fig, axes = plt.subplots(4, t_steps, figsize=(t_steps * 3, 12))
    
    for t in range(t_steps):

        # Get mask for timestep
        m = mask_np[b, t, 0]

        # Zeile 1: Ground Truth kNDVI
        axes[0, t].imshow(y_true_np[b, t, 0], vmin=0, vmax=1, cmap='YlGn')
        axes[0, t].set_title(f"GT T{t+1}")
        
        # Zeile 2: Prediction kNDVI
        axes[1, t].imshow(y_pred_np[b, t, 0], vmin=0, vmax=1, cmap='YlGn')
        axes[1, t].set_title(f"Pred T{t+1}")

        # Zeile 3: True Delta + Masken Overlay
        # Wir nutzen eine Maske, um bewölkte Bereiche (m=0) halbtransparent zu machen
        dt = delta_true[b, t, 0]
        im2 = axes[2, t].imshow(dt, vmin=-0.2, vmax=0.2, cmap='RdYlGn')
        # Overlay: Wo maske == 0, zeichne weiß mit 0.5 alpha
        axes[2, t].imshow(np.where(m == 0, 1, np.nan), cmap='gray_r', vmin=0, vmax=1, alpha=0.5)
        axes[2, t].set_title(f"True Delta T{t}")

        # Zeile 4: Pred Delta
        im3 = axes[3, t].imshow(delta_pred[b, t, 0], vmin=-0.2, vmax=0.2, cmap='RdYlGn')
        axes[3, t].set_title(f"Pred Delta T{t}")

    # Achsen verschönern
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # Colorbars ohne tight_layout Konflikt
    fig.subplots_adjust(right=0.88, left=0.05, top=0.9, bottom=0.1, wspace=0.1, hspace=0.3)
    cbar_ax_abs = fig.add_axes([0.91, 0.55, 0.015, 0.35])
    fig.colorbar(axes[0,0].get_images()[0], cax=cbar_ax_abs, label='kNDVI Abs')
    
    cbar_ax_delta = fig.add_axes([0.91, 0.1, 0.015, 0.35])
    fig.colorbar(im3, cax=cbar_ax_delta, label='kNDVI Delta')
    
    if save_path:
        plt.savefig(f"{save_path}/epoch_{epoch}_batch_{batch_idx}.png")
        plt.close()

    return fig