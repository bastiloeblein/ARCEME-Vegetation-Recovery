import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
from pathlib import Path
from model.dataset import get_val_tiles_auto, ARCEME_Dataset


def get_ckpt_and_hparams(ckpt_path):
    ckpt_path = Path(ckpt_path)

    # Checkpoint laden
    data = torch.load(ckpt_path, map_location="cpu")

    callbacks = data.get("callbacks", {})

    best_model_path = None
    best_score = None

    # ModelCheckpoint Callback finden
    for k, v in callbacks.items():
        if "ModelCheckpoint" in k:
            best_model_path = v.get("best_model_path")
            best_score = v.get("best_model_score")
            break

    # Fallback falls nichts gefunden
    if best_model_path is None:
        best_model_path = str(ckpt_path)

    best_model_path = Path(best_model_path)

    # version folder finden
    fold_dir = ckpt_path.parents[1]

    # suche version_X
    versions = list(fold_dir.glob("version_*"))
    versions.sort()
    if versions:
        hparams_path = versions[-1] / "hparams.yaml"  # letzte version
    else:
        hparams_path = fold_dir / "hparams.yaml"  # fallback direkt im fold

    return {
        "checkpoint": best_model_path,
        "best_score": float(best_score) if best_score is not None else None,
        "hparams": hparams_path if hparams_path.exists() else None,
    }


def get_best_model_path(log_dir="../tb_logs"):

    checkpoint_files = glob.glob(os.path.join(log_dir, "**", "*.ckpt"), recursive=True)

    if not checkpoint_files:
        print(f"❌ ERROR: Keine .ckpt Dateien in {log_dir} gefunden!")
        return None

    best_loss = float("inf")
    best_path = None

    for ckpt in checkpoint_files:
        try:
            match = re.search(r"val_loss=([0-9].+)\.ckpt", ckpt)
            if match:
                loss = float(match.group(1))

                if loss < best_loss:
                    best_loss = loss
                    best_path = ckpt
        except Exception as e:
            print(f"⚠️ Konnte Loss für {ckpt} nicht lesen: {e}")
            continue

    return best_path


def plot_patch_comparison(
    ctx_np, true_np, pred_np, mask_np, z_ctx, z_tgt, meta, mse_steps, baseline_np
):
    # Wir erhöhen auf 6 Zeilen
    fig, axes = plt.subplots(6, 8, figsize=(22, 16))
    fig.patch.set_facecolor("white")

    # y_min, x_min = int(meta["top"][0]), int(meta["left"][0])

    for i in range(8):
        if i < 3:  # --- CONTEXT BEREICH ---
            idx = -3 + i
            axes[0, i].imshow(ctx_np[idx], vmin=0, vmax=0.8, cmap="Greens")
            axes[0, i].set_title(f"Ctx {idx}")

            # Neu: Die Baseline unter den Kontext plotten (als Referenz)
            if i == 2:  # Nur einmal in der Spalte anzeigen
                axes[1, i].imshow(baseline_np, vmin=0, vmax=0.8, cmap="Greens")
                axes[1, i].set_title("Scharfe Baseline")
            else:
                axes[1, i].axis("off")

            axes[3, i].imshow(z_ctx[idx], vmin=0, vmax=0.8, cmap="Greens")
            axes[3, i].set_title("Zarr Raw Ctx")
            for r in [2, 4, 5]:
                axes[r, i].axis("off")

        else:  # --- TARGET BEREICH ---
            idx = i - 3
            current_mask = mask_np[idx] == 0

            # Maskierte Arrays
            m_true = np.ma.masked_where(current_mask, true_np[idx])
            m_pred = np.ma.masked_where(current_mask, pred_np[idx])
            m_zarr = np.ma.masked_where(current_mask, z_tgt[idx])
            m_diff = np.ma.masked_where(current_mask, (true_np[idx] - pred_np[idx]))

            # Zeile 1: GT
            axes[0, i].imshow(m_true, vmin=0, vmax=0.8, cmap="Greens")
            axes[0, i].set_title(f"GT +{idx+1}")

            # Zeile 2: Prediction
            axes[1, i].imshow(m_pred, vmin=0, vmax=0.8, cmap="Greens")
            axes[1, i].set_title("Pred")

            # Zeile 3: Das Delta (Was hat das Modell zum Baseline-Bild hinzugefügt?)
            # y_pred - baseline
            delta_map = pred_np[idx] - baseline_np
            m_delta = np.ma.masked_where(current_mask, delta_map)
            im_delta = axes[2, i].imshow(m_delta, vmin=-0.1, vmax=0.1, cmap="PuOr")
            axes[2, i].set_title("Predicted Delta")

            # Zeile 4: Zarr Vergleich
            axes[3, i].imshow(m_zarr, vmin=0, vmax=0.8, cmap="Greens")
            axes[3, i].set_title("Zarr Tgt")

            # Zeile 5: Error Map
            im_err = axes[4, i].imshow(m_diff, vmin=-0.2, vmax=0.2, cmap="bwr")
            axes[4, i].set_title(f"MSE: {mse_steps[idx]:.5f}")

            # Zeile 6: Maske
            axes[5, i].imshow(mask_np[idx], cmap="gray")
            axes[5, i].set_title("Mask")

    for ax in axes.flatten():
        ax.axis("off")

    # Colorbars für Delta und Error
    plt.colorbar(
        im_delta,
        ax=axes[2, :],
        orientation="horizontal",
        fraction=0.02,
        pad=0.1,
        label="Delta (Pred - Baseline)",
    )
    plt.colorbar(
        im_err,
        ax=axes[4, :],
        orientation="horizontal",
        fraction=0.02,
        pad=0.1,
        label="Error (True - Pred)",
    )

    plt.suptitle(
        f"Evaluation mit Baseline-Check | Cube: {meta['path'][0].split('/')[-1]}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()


def prepare_test_loader_from_cfg(cube_files, cfg, n_cubes=4, patches_per_cube=16):
    """
    Sets up the evaluation pipeline using the experiment configuration.

    Args:
        cube_files: List of available .zarr paths.
        cfg: The full configuration dictionary (first_model).
        n_cubes: Number of full cubes to process.
        patches_per_cube: Number of tiles to extract (e.g., 16 for full 1000x1000 coverage).
    """

    # 1. Extract params from cfg
    data_cfg = cfg["data"]
    v_cfg = data_cfg["variables"]

    p_size = data_cfg["patch_size"]
    c_len = data_cfg["context_length"]
    t_len = data_cfg["target_length"]
    # b_size = cfg["training"]["batch_size"]

    # 2. Select subset of files
    selected_files = cube_files[:n_cubes]
    print(
        f"📋 Evaluating {len(selected_files)} cubes: {[f.split('/')[-1] for f in selected_files]}"
    )

    # 3. Generate tile coordinates (Deterministic for stitching)
    all_possible_tiles = get_val_tiles_auto(
        selected_files, patch_size=p_size, dim_max=1000
    )

    # 4. Filter tiles for specific count per cube
    test_tiles = []
    potential_per_cube = len(all_possible_tiles) // len(selected_files)

    for c_idx in range(len(selected_files)):
        start_idx = c_idx * potential_per_cube
        test_tiles.extend(all_possible_tiles[start_idx : start_idx + patches_per_cube])

    print(
        f"✅ Created {len(test_tiles)} tiles ({patches_per_cube} per cube, size {p_size})"
    )

    # 5. Initialize Dataset
    test_ds = ARCEME_Dataset(
        selected_files,
        context_length=c_len,
        target_length=t_len,
        patch_size=p_size,
        train=False,
        s2_vars=v_cfg["s2"],
        s1_vars=v_cfg["s1"],
        era5_vars=v_cfg["era5"],
        static_vars=v_cfg["static"],
        fixed_tiles=test_tiles,
    )

    # 6. Initialize DataLoader (Strictly shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )

    print(f"🚀 Loader ready. Total batches: {len(test_loader)}")
    return test_loader, selected_files
