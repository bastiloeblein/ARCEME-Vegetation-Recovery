import os
import glob 
import matplotlib.pyplot as plt
import numpy as np
import re

def get_best_model_path(log_dir="../tb_logs"):

    checkpoint_files = glob.glob(os.path.join(log_dir, "**", "*.ckpt"), recursive=True)
    
    if not checkpoint_files:
        print(f"❌ ERROR: Keine .ckpt Dateien in {log_dir} gefunden!")
        return None
        
    best_loss = float('inf')
    best_path = None
    
    for ckpt in checkpoint_files:
        try:
            # Wir suchen nach dem val_loss im Dateinamen
            # Beispiel: epoch=2-val_loss=0.04.ckpt
            if "val_loss=" in ckpt:
                # Extrahiere die Zahl nach 'val_loss='
                loss_part = ckpt.split("val_loss=")[1]
                loss_str = loss_part.replace(".ckpt", "")
                loss = float(loss_str)
                
                if loss < best_loss:
                    best_loss = loss
                    best_path = ckpt
        except Exception as e:
            print(f"⚠️ Konnte Loss für {ckpt} nicht lesen: {e}")
            continue

    return best_path

def plot_patch_comparison(ctx_np, true_np, pred_np, mask_np, z_ctx, z_tgt, meta, mse_steps, baseline_np):
    # Wir erhöhen auf 6 Zeilen
    fig, axes = plt.subplots(6, 8, figsize=(22, 16)) 
    fig.patch.set_facecolor('white')
    
    y_min, x_min = int(meta['top'][0]), int(meta['left'][0])

    for i in range(8):
        if i < 3: # --- CONTEXT BEREICH ---
            idx = -3 + i
            axes[0, i].imshow(ctx_np[idx], vmin=0, vmax=0.8, cmap='Greens')
            axes[0, i].set_title(f"Ctx {idx}")
            
            # Neu: Die Baseline unter den Kontext plotten (als Referenz)
            if i == 2: # Nur einmal in der Spalte anzeigen
                axes[1, i].imshow(baseline_np, vmin=0, vmax=0.8, cmap='Greens')
                axes[1, i].set_title("Scharfe Baseline")
            else:
                axes[1, i].axis('off')
                
            axes[3, i].imshow(z_ctx[idx], vmin=0, vmax=0.8, cmap='Greens')
            axes[3, i].set_title("Zarr Raw Ctx")
            for r in [2, 4, 5]: axes[r, i].axis('off')

        else: # --- TARGET BEREICH ---
            idx = i - 3
            current_mask = (mask_np[idx] == 0)
            
            # Maskierte Arrays
            m_true = np.ma.masked_where(current_mask, true_np[idx])
            m_pred = np.ma.masked_where(current_mask, pred_np[idx])
            m_zarr = np.ma.masked_where(current_mask, z_tgt[idx])
            m_diff = np.ma.masked_where(current_mask, (true_np[idx] - pred_np[idx]))

            # Zeile 1: GT
            axes[0, i].imshow(m_true, vmin=0, vmax=0.8, cmap='Greens')
            axes[0, i].set_title(f"GT +{idx+1}")
            
            # Zeile 2: Prediction
            axes[1, i].imshow(m_pred, vmin=0, vmax=0.8, cmap='Greens')
            axes[1, i].set_title("Pred")
            
            # Zeile 3: Das Delta (Was hat das Modell zum Baseline-Bild hinzugefügt?)
            # y_pred - baseline
            delta_map = pred_np[idx] - baseline_np
            m_delta = np.ma.masked_where(current_mask, delta_map)
            im_delta = axes[2, i].imshow(m_delta, vmin=-0.1, vmax=0.1, cmap='PuOr')
            axes[2, i].set_title("Predicted Delta")

            # Zeile 4: Zarr Vergleich
            axes[3, i].imshow(m_zarr, vmin=0, vmax=0.8, cmap='Greens')
            axes[3, i].set_title("Zarr Tgt")
            
            # Zeile 5: Error Map
            im_err = axes[4, i].imshow(m_diff, vmin=-0.2, vmax=0.2, cmap='bwr')
            axes[4, i].set_title(f"MSE: {mse_steps[idx]:.5f}")
            
            # Zeile 6: Maske
            axes[5, i].imshow(mask_np[idx], cmap='gray')
            axes[5, i].set_title("Mask")

    for ax in axes.flatten(): ax.axis('off')
    
    # Colorbars für Delta und Error
    plt.colorbar(im_delta, ax=axes[2, :], orientation='horizontal', fraction=0.02, pad=0.1, label="Delta (Pred - Baseline)")
    plt.colorbar(im_err, ax=axes[4, :], orientation='horizontal', fraction=0.02, pad=0.1, label="Error (True - Pred)")
    
    plt.suptitle(f"Evaluation mit Baseline-Check | Cube: {meta['path'][0].split('/')[-1]}", fontsize=16)
    plt.tight_layout()
    plt.show()
    
# def plot_patch_comparison(ctx_np, true_np, pred_np, mask_np, zarr_ctx, zarr_tgt, meta, mse_steps, baseline_np):
#     fig, axes = plt.subplots(6, 8, figsize=(22, 16))
#     fig.patch.set_facecolor('white')
    
#     y_min, x_min = int(meta['top'][0]), int(meta['left'][0])

#     for i in range(8):
#         if i < 3: # --- CONTEXT ---
#             idx = -3 + i
#             axes[0, i].imshow(ctx_np[idx], vmin=0, vmax=0.8, cmap='Greens')
#             axes[0, i].set_title(f"Batch Ctx {idx}")
#             axes[2, i].imshow(zarr_ctx[idx], vmin=0, vmax=0.8, cmap='Greens')
#             axes[2, i].set_title("Zarr Raw Ctx")
#             for r in [1, 3, 4]: axes[r, i].axis('off')
#         else: # --- TARGET ---
#             idx = i - 3
#             current_mask = (mask_np[idx] == 0)
            
#             # Maskierte Arrays für saubere Anzeige
#             m_true = np.ma.masked_where(current_mask, true_np[idx])
#             m_pred = np.ma.masked_where(current_mask, pred_np[idx])
#             m_zarr = np.ma.masked_where(current_mask, zarr_tgt[idx])
#             m_diff = np.ma.masked_where(current_mask, (true_np[idx] - pred_np[idx]))

#             axes[0, i].imshow(m_true, vmin=0, vmax=0.8, cmap='Greens')
#             axes[0, i].set_title(f"GT +{idx+1}")
#             axes[1, i].imshow(m_pred, vmin=0, vmax=0.8, cmap='Greens')
#             axes[1, i].set_title("Pred")
#             axes[2, i].imshow(m_zarr, vmin=0, vmax=0.8, cmap='Greens')
#             axes[2, i].set_title("Zarr Tgt")
            
#             im_err = axes[3, i].imshow(m_diff, vmin=-0.2, vmax=0.2, cmap='bwr')
#             axes[3, i].set_title(f"MSE: {mse_steps[idx]:.5f}")
            
#             axes[4, i].imshow(mask_np[idx], cmap='gray')
#             axes[4, i].set_title("Mask")

#     for ax in axes.flatten(): ax.axis('off')
#     plt.colorbar(im_err, ax=axes[3, :], orientation='horizontal', fraction=0.02, pad=0.1, label="Diff (True - Pred)")
#     plt.suptitle(f"Cube: {meta['path'][0].split('/')[-1]} | Patch: y={y_min}, x={x_min}", fontsize=16)
#     plt.show()