import yaml
import torch
import os
import random
import json
import sys
import wandb
import numpy as np
from pathlib import Path
from datetime import datetime
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from my_utils.check_cubes import generate_exclusion_list
import multiprocessing as mp

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
    print("Starting from:", ROOT_DIR)
from model.ConvLSTM_model import ConvLSTM_Model
from model.dataset import (
    ARCEME_Dataset,
    get_llto_splits,
    get_val_tiles_auto,
)
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

with open(MODEL_DIR / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# --- Constants & Paths ---
# PROCESSED_DIR = "/scratch/sloeblein/postprocessed"
PROCESSED_DIR = "/net/projects/arceme/vegetation_recovery_prediction/postprocessed"
CSV_PATH = str(ROOT_DIR / "data_processing" / "data" / "train_test_split.csv")
EXCLUDE_CSV_PATH = str(ROOT_DIR / "data_processing" / "data" / "excluded_cubes.csv")
K_FOLDS = 3  # for CV (test for 4 and 5)


def main():

    # 1. Generate list of cubes to exclude based on current config quality settings
    df_excluded_cubes = generate_exclusion_list(
        processed_dir=PROCESSED_DIR, exclude_csv_path=EXCLUDE_CSV_PATH, cfg=cfg
    )
    excluded_cube_ids = set(df_excluded_cubes["cube_id"].astype(str).tolist())

    # 2. Get number of used cubes
    all_zarrs = [d for d in os.listdir(PROCESSED_DIR) if d.endswith(".zarr")]
    num_total = len(all_zarrs)
    num_excluded = len(excluded_cube_ids)
    num_used = num_total - num_excluded

    # 3. Write to config
    cfg["data"]["stats"] = {
        "total_cubes_found": num_total,
        "excluded_cubes": num_excluded,
        "used_cubes": num_used,
    }

    print(
        f"📊 Dataset Stats: Total: {num_total} | Excluded: {num_excluded} | Used: {num_used}"
    )

    # Create folder for each run
    RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RUN_DIR = str(MODEL_DIR / "wand_db_logs" / f"run_{RUN_TIMESTAMP}")
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"📁 Run Directory: {RUN_DIR}")
    cfg["model"]["run_dir"] = RUN_DIR

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
    all_folds = get_llto_splits(
        PROCESSED_DIR, CSV_PATH, k=K_FOLDS, show=True, exclude_list=excluded_cube_ids
    )
    # all_folds = get_llto_splits_strict(PROCESSED_DIR, CSV_PATH, k=K_FOLDS, show=True, save_path=RUN_DIR,exclude_list=excluded_cube_ids)

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
            config=cfg,
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
            config=cfg,
            s2_vars=v_cfg["s2"],
            s1_vars=v_cfg["s1"],
            era5_vars=v_cfg["era5"],
            static_vars=v_cfg["static"],
            fixed_tiles=val_tiles,
            use_augmentation=False,
        )

        # Create a generator for DataLoader seeding
        g = torch.Generator()
        g.manual_seed(42)

        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=cfg["data"]["data_loader"]["num_workers"],
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,  # can be deleted
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["data_loader"]["num_workers"],
            pin_memory=True,
        )

        print(
            f"DEBUG: Data loaded - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
        )
        print(f"DEBUG: Training Dataset Length: {len(train_ds)} samples")
        print(f"DEBUG: Steps per Epoch: {len(train_loader)}")

        # Model intialization
        # New model in each fold
        model = ConvLSTM_Model(cfg)
        model.fold_idx = fold_idx

        print(f"DEBUG: Model loaded on device: {model.device}")

        # Logger & Callbacks (Saves best model of a fold)
        logger = TensorBoardLogger(
            RUN_DIR, name=f"fold_{fold_idx}", default_hp_metric=False
        )
        logger.experiment.add_text(
            "config",
            f"```yaml\n{yaml.dump(cfg)}\n```",
            global_step=0,
        )

        # Initialisiere den W&B Logger
        wandb_logger = WandbLogger(
            project="ARCEME_kNDVI_Prediction",  # Name deines Projekts
            name=f"{cfg['experiment_name']}_fold_{fold_idx}",
            group=cfg["experiment_name"],
            job_type="train",
            config=cfg,  # Speichert deine komplette YAML-Config automatisch!
        )

        # --- LOG EXCLUSION LIST AS ARTIFACT (only for the first fold to avoid redundancy) ---
        if fold_idx == 0:
            exclusion_artifact = wandb.Artifact(
                name=f"exclusion_list_{cfg['experiment_name']}",
                type="dataset_metadata",
                description="Cubes excluded based on quality thresholds",
            )
            exclusion_artifact.add_file(EXCLUDE_CSV_PATH)
            wandb_logger.experiment.log_artifact(exclusion_artifact)

        # Only save the best model of each fold based on validation loss
        # monitor_key = f"{cfg['training']['validation']['monitor']['split']}_{cfg['training']['validation']['monitor']['metric']}"
        monitor_key = cfg["training"]["validation"]["monitor"]["metric"]
        filename = f"best-model-{{epoch:02d}}-{{{monitor_key}:.6f}}"

        print(f"DEBUG: Monitor Key for Checkpointing: {monitor_key}")

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(RUN_DIR, f"fold_{fold_idx}", "checkpoints"),
            monitor=monitor_key,
            filename=filename,
            save_top_k=3,
            mode=cfg["training"]["validation"]["monitor_mode"],
        )

        # Early stopping if validation loss doesn't improve for 10 epochs
        early_stop = EarlyStopping(
            monitor=monitor_key,
            patience=cfg["training"]["optimizer"]["patience"],
            mode=cfg["training"]["validation"]["monitor_mode"],
        )

        clip_val = (
            cfg["training"]["gradient_clipping"]["value"]
            if cfg["training"]["gradient_clipping"]["enabled"]
            else None
        )
        clip_algo = (
            cfg["training"]["gradient_clipping"]["algorithm"]
            if cfg["training"]["gradient_clipping"]["enabled"]
            else "norm"
        )

        # Trainer
        trainer = Trainer(
            accumulate_grad_batches=cfg["training"][
                "accumulate_grad_batches"
            ],  # Simuliert Batch Size 32 bei realer BS 4
            log_every_n_steps=1,
            gradient_clip_val=clip_val,  # Nutzt jetzt die 0.5 aus deiner Config
            gradient_clip_algorithm=clip_algo,
            check_val_every_n_epoch=1,
            max_epochs=cfg["training"]["max_epochs"],
            accelerator=cfg["training"]["accelerator"],
            devices=cfg["training"]["devices"],
            precision=cfg["training"]["precision"],
            logger=wandb_logger,
            # logger=logger,
            callbacks=[checkpoint_callback, early_stop],
        )

        # Overview
        print_channel_info(v_cfg["s2"], v_cfg["s1"], v_cfg["era5"], v_cfg["static"])
        trainer.fit(model, train_loader, val_loader)

        # Get the final learning rate after training
        final_lr = model.optimizers().param_groups[0]["lr"]

        # --- Collect fold results for final summary ---
        # Get metrics from best model checkpoint
        val_results = trainer.validate(model, dataloaders=val_loader, ckpt_path="best")[
            0
        ]
        best_score = val_results.get(
            f"{cfg['training']['validation']['monitor']['split']}_{cfg['training']['validation']['monitor']['metric']}",
            float("nan"),
        )
        # best_score = (
        #     checkpoint_callback.best_model_score.item()
        #     if checkpoint_callback.best_model_score
        #     else float("nan")
        # )
        fold_result = {
            "fold": fold_idx,
            "best_score": best_score,
            "best_epoch": checkpoint_callback.best_model_path.split("epoch=")[-1].split(
                "-"
            )[
                0
            ],  # Sicherer als current_epoch
            "best_checkpoint": checkpoint_callback.best_model_path,
            "metrics": val_results,
            "val_files": [str(f) for f in val_files],
            "final_lr": final_lr,
        }
        fold_results.append(fold_result)

        print(f"\n✅ Fold {fold_idx} done | Best Val Score: {best_score:.4f}")

        del model
        del trainer
        del train_ds, val_ds
        del train_loader, val_loader
        torch.cuda.empty_cache()
        wandb.finish()

    # --- Final Summary ---
    if cfg["training"]["validation"]["monitor_mode"] == "min":
        best_fold = min(fold_results, key=lambda x: x["best_score"])
    else:
        best_fold = max(fold_results, key=lambda x: x["best_score"])

    summary = {
        "folds": fold_results,
        "best_fold": best_fold,
        "mean_val_loss": float(np.mean([r["best_score"] for r in fold_results])),
        "std_val_loss": float(np.std([r["best_score"] for r in fold_results])),
    }

    summary_path = os.path.join(RUN_DIR, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n📊 CV Summary saved: {summary_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    main()
