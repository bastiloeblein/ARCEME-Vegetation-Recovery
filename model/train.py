import yaml
import torch
import os
import random
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from model.ConvLSTM_model import ConvLSTM_Model
from model.dataset import ARCEME_Dataset, get_llto_splits, get_llto_splits_strict, get_val_tiles_auto
from model.utils import print_channel_info, get_cloud_stats_zarr

# --- Reproducibility ---
# Set seed (only for reproducibility between patch approaches - can be delted later)
L.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("medium")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set DataLoader worker seed for reproducibility
# -> numpy/random are deterministic
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- Load Config ---
MODEL_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODEL_DIR.parent

with open(MODEL_DIR / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# --- Constants & Paths ---
PROCESSED_DIR = "/scratch/sloeblein/postprocessed"
CSV_PATH = str(ROOT_DIR / "data_processing" / "data" / "train_test_split.csv")
K_FOLDS = 3  # for CV (test for 4 and 5)

# Create folder for each run
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = str(MODEL_DIR / "tb_logs" / f"run_{RUN_TIMESTAMP}")
os.makedirs(RUN_DIR, exist_ok=True)
print(f"📁 Run Directory: {RUN_DIR}")

# --- Dynamic Channel Calculation ---
v_cfg = cfg["data"]["variables"]
num_s2 = len(v_cfg["s2"])
num_s1 = len(v_cfg["s1"])
num_era5 = len(v_cfg["era5"])
num_masks = 2  # mask_s1 and mask_s2
num_lc = 12  # ESA Landcover One-Hot encoded
num_stat = len(v_cfg["static"])

# Input channels (Context): All features combined
# S2 + S1 + ERA5 + Masks + LC_OneHot + Statics
total_in_channels = num_s2 + num_s1 + num_era5 + num_masks + num_lc + num_stat

# Future channels (Decoder): Features known in the future
# ERA5 + LC_OneHot + Statics
total_fut_channels = num_era5 + num_lc + num_stat

# Pass calculated channels into the model config
cfg["model"]["input_channels"] = total_in_channels
cfg["model"]["future_channels"] = total_fut_channels

print("--- Channel Configuration ---")
print(f"Input Channels (Context): {total_in_channels}")
print(f"Future Channels (Known):  {total_fut_channels}")
print("-----------------------------")

# Save config for this run
config_log_path = os.path.join(RUN_DIR, "config_used.yaml")
with open(config_log_path, "w") as f:
    yaml.dump(cfg, f)
print(f"✅ Config saved: {config_log_path}")

# --- Create CV Splits ---
# all_folds = get_llto_splits(PROCESSED_DIR, CSV_PATH, k=K_FOLDS, show=True)
all_folds = get_llto_splits_strict(PROCESSED_DIR, CSV_PATH, k=K_FOLDS, show=True, save_path=RUN_DIR) 

# --- Log Split Info ---
split_info = {}
for fold_idx, (train_files, val_files) in enumerate(all_folds):
    split_info[f"fold_{fold_idx}"] = {
        "train_files": [str(f) for f in train_files],
        "val_files": [str(f) for f in val_files],
        "num_train": len(train_files),
        "num_val": len(val_files),
    }
# Save as json
split_log_path = os.path.join(RUN_DIR, "cv_splits.json")
with open(split_log_path, "w") as f:
    json.dump(split_info, f, indent=2)
print(f"✅ CV Split saved: {split_log_path}")


# --- Training Loop ---
fold_results = []

for fold_idx, (train_files, val_files) in enumerate(all_folds):
    print("\n" + "=" * 50)
    print(f"🚀 STARTING FOLD {fold_idx} (LLTO-CV)")
    print("=" * 50)

    print("📊 Scanning Cloud Cover (Zarr)...")
    train_clouds = get_cloud_stats_zarr(train_files)
    val_clouds = get_cloud_stats_zarr(val_files)
    
    print(f"☁️ Training Clouds:   {train_clouds:.2f}%")
    print(f"☁️ Validation Clouds: {val_clouds:.2f}%")

    # Get fixed patches for validation
    val_tiles = get_val_tiles_auto(val_files, patch_size=cfg["data"]["patch_size"])

    # Create Datasets
    train_ds = ARCEME_Dataset(
        train_files,
        context_length=cfg["data"]["context_length"],
        target_length=cfg["data"]["target_length"],
        patch_size=cfg["data"]["patch_size"],
        train=True,
        s2_vars=v_cfg["s2"],
        s1_vars=v_cfg["s1"],
        era5_vars=v_cfg["era5"],
        static_vars=v_cfg["static"],
        use_augmentation=cfg["training"]["use_augmentation"],
    )
    val_ds = ARCEME_Dataset(
        val_files,
        context_length=cfg["data"]["context_length"],
        target_length=cfg["data"]["target_length"],
        patch_size=cfg["data"]["patch_size"],
        train=False,
        s2_vars=v_cfg["s2"],
        s1_vars=v_cfg["s1"],
        era5_vars=v_cfg["era5"],
        static_vars=v_cfg["static"],
        fixed_tiles=val_tiles,
    )

    # Create a generator for DataLoader seeding
    g = torch.Generator()
    g.manual_seed(42)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,  # can be deleted
    ) 
    val_loader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=cfg["training"]["batch_size"], 
        shuffle=False, 
        num_workers=4
    )

    # Model intialization
    # New model in each fold
    model = ConvLSTM_Model(cfg)
    model.fold_idx = fold_idx

    # Logger & Callbacks (Saves best model of a fold)
    logger = TensorBoardLogger(RUN_DIR, name=f"fold_{fold_idx}", default_hp_metric=False)
    
    # Log Split + Config also in TensorBoard 
    logger.experiment.add_text(
        "split_info",
        f"**Train files ({len(train_files)}):**\n" + "\n".join([f"- {f}" for f in train_files]) +
        f"\n\n**Val files ({len(val_files)}):**\n" + "\n".join([f"- {f}" for f in val_files]),
        global_step=0,
    )
    logger.experiment.add_text(
        "config",
        f"```yaml\n{yaml.dump(cfg)}\n```",
        global_step=0,
    )
    
    # Only save the best model of each fold based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(RUN_DIR, f"fold_{fold_idx}", "checkpoints"),
        monitor="val_loss",
        filename="best-model-epoch={epoch:02d}-val_loss={val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    # Early stopping if validation loss doesn't improve for 10 epochs
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    # Trainer
    trainer = Trainer(
        accumulate_grad_batches=8, # Simuliert Batch Size 32 bei realer BS 4
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=cfg["training"]["max_epochs"],
        accelerator=cfg["training"]["accelerator"],
        devices=cfg["training"]["devices"],
        precision=cfg["training"]["precision"],
        logger=logger,
        callbacks=[checkpoint_callback, early_stop],
    )

    # Overview
    print_channel_info(v_cfg["s2"], v_cfg["s1"], v_cfg["era5"], v_cfg["static"])
    trainer.fit(model, train_loader, val_loader)

    # --- Collect fold results for final summary ---
    best_val_loss = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else float("nan")
    best_epoch = trainer.current_epoch
    best_ckpt = checkpoint_callback.best_model_path

    fold_result = {
        "fold": fold_idx,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_checkpoint": best_ckpt,
        "val_files": [str(f) for f in val_files],
    }
    fold_results.append(fold_result)
    print(f"\n✅ Fold {fold_idx} done | Best Val Loss: {best_val_loss:.4f} | Checkpoint: {best_ckpt}")

# --- Final Summary ---
summary = {
    "folds": fold_results,
    "best_fold": min(fold_results, key=lambda x: x["best_val_loss"]),
    "mean_val_loss": float(np.mean([r["best_val_loss"] for r in fold_results])),
    "std_val_loss": float(np.std([r["best_val_loss"] for r in fold_results])),
}

summary_path = os.path.join(RUN_DIR, "cv_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 50)
print("🏁 CROSS-VALIDATION COMPLETE")
print(f"   Mean Val Loss: {summary['mean_val_loss']:.4f} ± {summary['std_val_loss']:.4f}")
print(f"   Best Fold:     {summary['best_fold']['fold']} (Val Loss: {summary['best_fold']['best_val_loss']:.4f})")
print(f"   Best Model:    {summary['best_fold']['best_checkpoint']}")
print(f"   Summary:       {summary_path}")
print("=" * 50)

