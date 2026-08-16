import os
import sys
import json
import yaml
import random
import re
from my_utils.warmup import ConfigWarmupCallback
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from model.ConvLSTM_model import ConvLSTM_Model
from model.dataset import ARCEME_Dataset, get_val_tiles_auto
from model.cv_splits import (
    create_spacetime_folds,
)
from model.utils import print_channel_info
from my_utils.check_cubes import generate_exclusion_list


# Deterministic workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ARCEMEPipeline:
    def __init__(self, config, mode="train", run_dir=None, ablation_split_path=None):
        self.mode = mode  # "train" or "eval"
        self.cfg = config
        self.v_cfg = self.cfg["data"]["variables"]

        # --- Environment & Reproducibility ---
        self.global_seed = self.cfg["training"]["seed"]
        L.seed_everything(self.global_seed, workers=True)
        torch.set_float32_matmul_precision(
            "medium"
        )  # reduces precision and improves speed
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True"  # memory management
        )

        # --- Paths & Directories ---
        self.train_test_split_csv = str(ROOT_DIR / self.cfg["data"]["train_test_split"])
        # self.exclude_csv_path = str(
        #     ROOT_DIR / "data_processing" / "data" / "excluded_cubes.csv"
        # )  # maybe better save this to run_dir

        if self.mode == "train":
            self.processed_dir = self.cfg["data"]["train_data_dir"]
        elif self.mode == "eval":
            self.processed_dir = self.cfg["data"]["test_data_dir"]

        # Run Directory Logic (New vs. Resume)
        if run_dir is not None:
            # Resuming an existing run or evaluating
            self.run_dir = run_dir
            print(f"📁 Using existing Run Directory: {self.run_dir}")
        else:
            # Fresh training run
            run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            exp_name = self.cfg.get("experiment_name", "unnamed_experiment")
            run_folder_name = f"{exp_name}_{run_timestamp}"

            folder_prefix = "test_logs" if self.mode == "eval" else "wand_db_logs"

            self.run_dir = str(
                Path(__file__).resolve().parent / folder_prefix / run_folder_name
            )
            os.makedirs(self.run_dir, exist_ok=True)
            print(f"📁 Created new Run Directory: {self.run_dir}")

            # =================================================================
            # Ablation Split Import (Import predefined cv_splits)
            # =================================================================
            if ablation_split_path is not None:
                if os.path.exists(ablation_split_path):
                    import shutil

                    # Copies existing cv_split into run_dir
                    target_split_path = os.path.join(self.run_dir, "cv_splits.json")
                    shutil.copy2(ablation_split_path, target_split_path)

                    print(
                        f"🧬 ABLATION STUDY: Predefined splits from {ablation_split_path} imported!"
                    )

                else:
                    raise FileNotFoundError(
                        f"❌ ERROR: Path does not exist!\n"
                        f"Passed: {ablation_split_path}\n"
                        f"Check path in terminal!"
                    )
            # =================================================================
            # Save config for this run
            with open(os.path.join(self.run_dir, "config_used.yaml"), "w") as f:
                yaml.dump(self.cfg, f)

        self.cfg["model"]["run_dir"] = self.run_dir

        # Save excluded cubes csv to run_dir
        self.exclude_csv_path = os.path.join(self.run_dir, "excluded_cubes.csv")

        # --- CV Params ---
        if self.cfg["cross_validation"]["enabled"]:
            self.k_folds = self.cfg["cross_validation"]["k_folds"]
            self.cv_type = self.cfg["cross_validation"]["type"]
        else:
            self.k_folds = 1
            self.cv_type = "none"

        # final refit without validation loader.
        self.final_refit_cfg = self.cfg["training"].get("final_refit", {})
        self.is_final_refit = bool(self.final_refit_cfg.get("enabled", False))

        self._calculate_dynamic_channels()

    def _calculate_dynamic_channels(self):
        """
        Calculate the number of input and future channels based on the variables in the config.
        """
        # Inputs are: S2 + S1 + ERA5 + 3 Mask Channels (S1, S2, additional unique kNDVI mask)+ 12 One-Hot Encoded LC Channels + Static Variables
        total_in = (
            len(self.v_cfg["s2"])
            + len(self.v_cfg["s1"])
            + len(self.v_cfg["era5"])
            + (3 if len(self.v_cfg["s1"]) > 0 else 2)  # Masks (S1, kNDVI, S2 rest)
            + 12
            + len(self.v_cfg["static"])
        )
        # Guided input gets: placeholder_kNDVI Channel + Future ERA5 + 12 One-Hot Encoded LC Channels + Static Variables
        total_fut = 1 + len(self.v_cfg["era5"]) + 12 + len(self.v_cfg["static"])

        self.cfg["model"]["input_channels"] = total_in
        self.cfg["model"]["future_channels"] = total_fut

        print(f"--- Channels Setup | Context: {total_in} | Future: {total_fut} ---")

    def prepare_data(self):
        """ "
        Evaluates cubes against quality thresholds defined in config.
        Returns LLTO splits (if train) or a list of valid paths (if test).
        """
        if self.mode == "eval":
            # Just return all valid cubes in the test directory
            # Or do we also drop excluded cubes in eval? Maybe not, because we want to know how well the model does on the test set as it is, and not artificially inflate performance by dropping hard cubes?
            # Maybe PREFILTER!!
            all_zarrs = [
                d for d in os.listdir(self.processed_dir) if d.endswith(".zarr")
            ]
            return [os.path.join(self.processed_dir, z) for z in all_zarrs]

        # Reuse the exact manifest when a final refit is restarted.
        # Unlike CV, this manifest has one all-training-cubes list and no folds.
        final_refit_manifest_path = os.path.join(
            self.run_dir, "final_refit_manifest.json"
        )
        if self.is_final_refit and os.path.exists(final_refit_manifest_path):
            print(
                "♻️ Found final_refit_manifest.json. Reusing the exact final-refit "
                "training cubes."
            )
            with open(final_refit_manifest_path, "r") as f:
                return json.load(f)["train_files"]

        # --- Train Mode: Check if splits already exist (Resume Case) ---
        print(f"\n🔍 Scanning {self.mode.upper()} directory: {self.processed_dir}")
        split_log_path = os.path.join(self.run_dir, "cv_splits.json")
        if os.path.exists(split_log_path):
            print(
                "♻️ Found existing cv_splits.json. Loading to guarantee exact same folds for resume..."
            )
            with open(split_log_path, "r") as f:
                split_info = json.load(f)

            all_folds = []
            for fold_idx in range(self.k_folds):
                f_data = split_info[f"fold_{fold_idx}"]
                all_folds.append((f_data["train_files"], f_data["val_files"]))
            return all_folds

        # --- Fresh Train Mode: Create Splits ---
        print(f"\n🔍 Scanning TRAIN directory: {self.processed_dir}")
        df_excluded_cubes = generate_exclusion_list(
            processed_dir=self.processed_dir,
            exclude_csv_path=self.exclude_csv_path,
            cfg=self.cfg,
        )
        excluded_ids = set(df_excluded_cubes["cube_id"].astype(str).tolist())

        all_zarrs = [d for d in os.listdir(self.processed_dir) if d.endswith(".zarr")]
        valid_zarrs_paths = [
            os.path.join(self.processed_dir, z)
            for z in all_zarrs
            if not any(ex in z for ex in excluded_ids)
        ]

        self.cfg["data"]["stats"] = {
            "total_cubes": len(all_zarrs),
            "excluded_cubes": len(excluded_ids),
            "used_cubes": len(valid_zarrs_paths),
        }
        print(
            f"📊 Stats: Total: {len(all_zarrs)} | Excluded: {len(excluded_ids)} | Valid: {len(valid_zarrs_paths)}"
        )

        if self.is_final_refit:
            # Store complete, quality-filtered training set used for the final training run. 
            final_refit_manifest = {
                "train_files": [str(path) for path in valid_zarrs_paths],
                "num_train": len(valid_zarrs_paths),
                "seed": self.global_seed,
            }
            with open(final_refit_manifest_path, "w") as f:
                json.dump(final_refit_manifest, f, indent=2)
            print(f"✅ Final-refit manifest saved: {final_refit_manifest_path}")
            return valid_zarrs_paths

        # Create Splits based on defined strategy
        if self.cv_type == "llto":
            print("\n✂️ Creating LLTO Cross-Validation Splits...")
            cv_data = create_spacetime_folds(
                valid_zarrs_paths,
                self.train_test_split_csv,
                spacevar="koppen_geiger",
                timevar=None,
                k=self.k_folds,
                seed=self.global_seed,
                show=True,
                save_path=self.run_dir,
            )
        elif self.cv_type == "llto_strict":
            print("\n✂️ Creating LLTO-Strict Cross-Validation Splits...")
            cv_data = create_spacetime_folds(
                valid_zarrs_paths,
                self.train_test_split_csv,
                spacevar="koppen_geiger",
                timevar="pheno_season_name",
                k=self.k_folds,
                seed=self.global_seed,
                show=True,
                save_path=self.run_dir,
            )
        elif self.cv_type == "llto_holdout":
            print("\n✂️ Creating Clean Spatial Hold-Out Split (85% Train / 15% Val)...")
            cv_data = create_spacetime_folds(
                valid_zarrs_paths,
                self.train_test_split_csv,
                spacevar="koppen_geiger",
                timevar=None,
                k=1,
                seed=self.global_seed,
                show=True,
                save_path=self.run_dir,
                val_size_if_k1=0.15,
            )
            self.k_folds = 1

        else:
            raise ValueError(f"Unknown CV type: {self.cv_type}")

        all_folds = [(f["train_files"], f["val_files"]) for f in cv_data["folds"]]

        # --- Log Split Info ---
        split_info = {}
        for fold_idx, (train_files, val_files) in enumerate(all_folds):
            split_info[f"fold_{fold_idx}"] = {
                "train_files": [str(f) for f in train_files],
                "val_files": [str(f) for f in val_files],
                "num_train": len(train_files),
                "num_val": len(val_files),
            }

        with open(split_log_path, "w") as f:
            json.dump(split_info, f, indent=2)
        print(f"✅ CV Split saved: {split_log_path}")

        return all_folds

    def get_train_dataloader(self, train_files, fold_idx=0):
        """Create the augmented random-crop training loader without validation data."""
        # Loader for the final refit: All eligible training cubes are used and no validation data is held back
        train_ds = ARCEME_Dataset(
            train_files,
            context_length=self.cfg["data"]["context_length"],
            target_length=self.cfg["data"]["target_length"],
            patch_size=self.cfg["data"]["patch_size"],
            train=True,
            config=self.cfg,
            s2_vars=self.v_cfg["s2"],
            s1_vars=self.v_cfg["s1"],
            era5_vars=self.v_cfg["era5"],
            static_vars=self.v_cfg["static"],
            fixed_tiles=None,
            use_augmentation=self.cfg["training"]["use_augmentation"],
        )

        g = torch.Generator()
        g.manual_seed(self.global_seed + fold_idx)

        return DataLoader(
            train_ds,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=self.cfg["data"]["data_loader"]["num_workers"],
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def get_dataloaders(self, train_files, val_files, fold_idx):
        # Define fixed patches for validation
        val_tiles = get_val_tiles_auto(
            val_files, patch_size=self.cfg["data"]["patch_size"]
        )

        # Create Datasets
        val_ds = ARCEME_Dataset(
            val_files,
            context_length=self.cfg["data"]["context_length"],
            target_length=self.cfg["data"]["target_length"],
            patch_size=self.cfg["data"]["patch_size"],
            train=False,
            config=self.cfg,
            s2_vars=self.v_cfg["s2"],
            s1_vars=self.v_cfg["s1"],
            era5_vars=self.v_cfg["era5"],
            static_vars=self.v_cfg["static"],
            fixed_tiles=val_tiles,
            use_augmentation=False,
        )

        train_loader = self.get_train_dataloader(train_files, fold_idx)
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=self.cfg["data"]["data_loader"]["num_workers"],
            pin_memory=True,
        )

        return train_loader, val_loader

    def get_checkpoint_path(self, fold_idx, type="last"):
        """Returns the path to the last or best checkpoint for a given fold."""
        ckpt_dir = os.path.join(self.run_dir, f"fold_{fold_idx}", "checkpoints")
        if not os.path.exists(ckpt_dir):
            return None

        if type == "last":
            path = os.path.join(ckpt_dir, "last.ckpt")
            return path if os.path.exists(path) else None

        elif type == "best":

            # 1. Recursive Search: Checks for all .ckpt files in all subfolders
            all_ckpts = list(Path(ckpt_dir).rglob("*.ckpt"))

            # ignoring last.ckpt
            candidates = [str(p) for p in all_ckpts if p.name != "last.ckpt"]

            if not candidates:
                return None

            # 2. Parsing Logic
            parsed_candidates = []
            for filepath in candidates:
                # Extract filename (e.g. "NNSE_pixel=0.756.ckpt")
                filename = os.path.basename(filepath)

                try:
                    # Split at '=' and remove extension -> "0.756"
                    score_str = filename.split("=")[-1].replace(".ckpt", "")
                    score = float(score_str)
                    parsed_candidates.append((score, filepath))
                except ValueError:
                    continue

            if not parsed_candidates:
                return None

            # 3. Sort and return the best
            is_min = self.cfg["training"]["validation"]["monitor_mode"] == "min"
            parsed_candidates.sort(key=lambda x: x[0], reverse=not is_min)

            return parsed_candidates[0][1]

        return None

    def get_best_overall_checkpoint(self):
        """Scans the run_dir to find the absolutely best checkpoint across all completed folds."""
        # If final refit: get the final checkpoint after pre-specified number of epochs
        final_refit_summary_path = os.path.join(
            self.run_dir, "final_refit_summary.json"
        )
        if os.path.exists(final_refit_summary_path):
            with open(final_refit_summary_path, "r") as f:
                final_refit_summary = json.load(f)
            final_checkpoint = final_refit_summary.get("final_checkpoint")
            if final_checkpoint and os.path.exists(final_checkpoint):
                return final_checkpoint

        # Check if the summary file exists (means training finished cleanly)
        summary_path = os.path.join(self.run_dir, "cv_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)
                return summary["best_fold"]["best_checkpoint"]

        # If no summary exists (maybe training aborted), scan folders manually
        print(
            "⚠️ No cv_summary.json found. Manually scanning folders for the best model..."
        )
        best_overall_score = (
            float("inf")
            if self.cfg["training"]["validation"]["monitor_mode"] == "min"
            else float("-inf")
        )
        best_overall_ckpt = None
        is_min = self.cfg["training"]["validation"]["monitor_mode"] == "min"

        for fold_idx in range(self.k_folds):
            ckpt = self.get_checkpoint_path(fold_idx, type="best")
            if not ckpt:
                continue

            # Extract score from filename: "best-model-epoch=05-val_loss=0.034.ckpt"
            score_str = ckpt.split("=")[-1].replace(".ckpt", "")
            try:
                score = float(score_str)
                if (is_min and score < best_overall_score) or (
                    not is_min and score > best_overall_score
                ):
                    best_overall_score = score
                    best_overall_ckpt = ckpt
            except ValueError:
                continue

        return best_overall_ckpt

    def run_cv(self, start_fold=0, resume_from_type="last"):
        """Executes the CV loop. Handles resuming automatically."""
        if self.is_final_refit:
            return self.run_final_refit(resume_from_type=resume_from_type)

        # Get valid paths and cv splits
        all_folds = self.prepare_data()

        fold_results = []
        v_cfg = self.cfg["data"]["variables"]

        for fold_idx in range(start_fold, self.k_folds):
            train_files, val_files = all_folds[fold_idx]

            print("\n" + "=" * 50)
            print(f"🚀 STARTING FOLD {fold_idx} (LLTO-CV)")
            print(f"With {len(train_files)} train and {len(val_files)} val cubes.")
            print("=" * 50)

            # Check for existing checkpoint to resume
            resume_ckpt = self.get_checkpoint_path(
                fold_idx, type=resume_from_type
            )  # Returns None if no checkpoint exists
            if resume_ckpt:
                print(
                    f"♻️ Resuming Fold {fold_idx} from {resume_from_type} checkpoint: {resume_ckpt}"
                )

            train_loader, val_loader = self.get_dataloaders(
                train_files, val_files, fold_idx
            )

            # --- Model & Logger ---
            model = ConvLSTM_Model(self.cfg)
            model.fold_idx = fold_idx

            wandb_logger = WandbLogger(
                project=self.cfg.get("wandb", {}).get(
                    "project", "ARCEME_kNDVI_Prediction"
                ), 
                name=f"{self.cfg['experiment_name']}_fold_{fold_idx}",
                group=self.cfg["experiment_name"],
                # model_type  =self.cfg["model"]["model_type"],
                job_type="train",
                config=self.cfg,
                resume="allow",  # Allows resuming the same run if crashed - could crash... s. documentation
            )

            # maybe delete this and fix by saving the excluded cubes in the run_dir
            if fold_idx == start_fold and not resume_ckpt:
                if os.path.exists(self.exclude_csv_path):
                    # Log artifact only on first fresh fold
                    exclusion_artifact = wandb.Artifact(
                        name=f"exclusion_list_{self.cfg['experiment_name']}",
                        type="dataset_metadata",
                    )
                    exclusion_artifact.add_file(self.exclude_csv_path)
                    wandb_logger.experiment.log_artifact(exclusion_artifact)
                else:
                    print(f"⚠️ Warning: {self.exclude_csv_path} not  found gefunden.")

            # --- Callbacks ---
            ckpt_dir = os.path.join(self.run_dir, f"fold_{fold_idx}", "checkpoints")
            monitor_key = self.cfg["training"]["validation"]["monitor"]["metric"]
            monitor_mode = self.cfg["training"]["validation"]["monitor_mode"]
            filename = f"best-model-{{epoch:02d}}-{{{monitor_key}:.6f}}"

            checkpoint_callback = ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor=monitor_key,
                filename=filename,
                save_top_k=3,
                mode=monitor_mode,
                save_last=True,
            )

            early_stop = EarlyStopping(
                monitor=monitor_key,
                patience=self.cfg["training"]["optimizer"]["patience"],
                mode=monitor_mode,
                min_delta=0.00,  # Val criteria has to improve by at least this much to reset patience counter
                strict=True,  # might fail (checkt ob metric überhaupt da ist)
            )

            # Define Warmup Callback
            if (
                self.cfg["training"]["optimizer"]
                .get("warmup", {})
                .get("enabled", False)
            ):
                warmup_callback = ConfigWarmupCallback(self.cfg)
                callbacks = [warmup_callback, checkpoint_callback, early_stop]
            else:
                callbacks = [checkpoint_callback, early_stop]

            # --- Trainer ---
            trainer = Trainer(
                accumulate_grad_batches=self.cfg["training"]["accumulate_grad_batches"],
                log_every_n_steps=10,
                gradient_clip_val=(
                    self.cfg["training"]["gradient_clipping"]["value"]
                    if self.cfg["training"]["gradient_clipping"]["enabled"]
                    else None
                ),
                gradient_clip_algorithm=(
                    self.cfg["training"]["gradient_clipping"]["algorithm"]
                    if self.cfg["training"]["gradient_clipping"]["enabled"]
                    else "norm"
                ),
                check_val_every_n_epoch=1,
                max_epochs=self.cfg["training"]["max_epochs"],
                accelerator=self.cfg["training"]["accelerator"],
                devices=self.cfg["training"]["devices"],
                precision=self.cfg["training"]["precision"],
                logger=wandb_logger,
                callbacks=callbacks,
                enable_model_summary=True,  # zeigt die architektur und die anzahl der parameter an, könnte hilfreich sein
            )

            print_channel_info(v_cfg["s2"], v_cfg["s1"], v_cfg["era5"], v_cfg["static"], expected_input_channels=self.cfg["model"]["input_channels"],)

            # --- Start Training ---
            trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

            # --- VALIDATE & LOG ---
            val_results = trainer.validate(
                model, dataloaders=val_loader, ckpt_path="best"
            )[0]
            best_score = val_results.get(monitor_key, float("nan"))

            # Save stats of best model for this fold
            fold_results.append(
                {
                    "fold": fold_idx,
                    "best_score": best_score,
                    "best_checkpoint": checkpoint_callback.best_model_path,
                    "best_epoch": self._get_checkpoint_epoch(
                        checkpoint_callback.best_model_path   # here the epoch is saved, to derive median best cv epoch for the final refit
                    ),
                    "metrics": val_results,
                }
            )
            print(f"\n✅ Fold {fold_idx} done | Best Val Score: {best_score:.4f}")

            # Cleanup RAM/VRAM before next fold
            del model, trainer, train_loader, val_loader
            torch.cuda.empty_cache()
            wandb.finish()

        # --- Final Summary ---
        if fold_results:
            is_min = self.cfg["training"]["validation"]["monitor_mode"] == "min"
            best_fold = (
                min(fold_results, key=lambda x: x["best_score"])
                if is_min
                else max(fold_results, key=lambda x: x["best_score"])
            )
            # not sure if thats the right way to do it
            summary = {
                "folds": fold_results,
                "best_fold": best_fold,
                "mean_val_score": float(
                    np.mean([r["best_score"] for r in fold_results])
                ),
                "std_val_score": float(np.std([r["best_score"] for r in fold_results])),
            }
            best_epochs = [
                result["best_epoch"]
                for result in fold_results
                if result["best_epoch"] is not None
            ]
            if best_epochs:
                # Epoch indices are zero-based; the number for final refit is therefore: epoch + 1.
                summary["recommended_final_refit_epochs"] = int(
                    round(float(np.median(best_epochs)))
                ) + 1
            summary_path = os.path.join(self.run_dir, "cv_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n📊 CV Summary saved: {summary_path}")

    @staticmethod
    def _get_checkpoint_epoch(checkpoint_path):
        """Extract Lightning's zero-based epoch from a standard checkpoint name."""
        if not checkpoint_path:
            return None
        match = re.search(r"epoch=(\d+)", checkpoint_path)
        return int(match.group(1)) if match else None

    def run_final_refit(self, resume_from_type="last"):
        """Fit one selected configuration on every eligible training cube.

        Model selection must already be complete from CV. Consequently this
        method intentionally has no validation loader, early stopping, or
        validation-dependent learning-rate scheduler.
        """
        if self.mode != "train":
            raise RuntimeError("Final refit is only available in training mode.")

        epochs = self.final_refit_cfg.get("epochs")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError(
                "training.final_refit.epochs must be a positive integer. Set it "
                "from the CV summary's recommended_final_refit_epochs before "
                "starting the final refit."
            )

        final_summary_path = os.path.join(self.run_dir, "final_refit_summary.json")
        if os.path.exists(final_summary_path):
            raise RuntimeError(
                "This final-refit run is already complete. Start a new run for a "
                "new refit, or evaluate its final checkpoint instead."
            )

        # A final refit has no validation score and therefore only a
        # ``last.ckpt``.  Resuming it must use that checkpoint explicitly.
        if resume_from_type != "last":
            raise ValueError(
                "Final refit can only resume from its last checkpoint; use "
                "--resume_type last."
            )
        final_ckpt_dir = os.path.join(self.run_dir, "final_refit", "checkpoints")
        resume_ckpt = os.path.join(final_ckpt_dir, "last.ckpt")
        if not os.path.exists(resume_ckpt):
            resume_ckpt = None

        train_files = self.prepare_data()
        if not train_files:
            raise RuntimeError("No eligible training cubes are available for final refit.")

        print("\n" + "=" * 60)
        print("🚀 STARTING FINAL REFIT (all eligible training cubes, no validation)")
        print(f"With {len(train_files)} training cubes for exactly {epochs} epochs.")
        print("=" * 60)

        train_loader = self.get_train_dataloader(train_files)
        model = ConvLSTM_Model(self.cfg)

        wandb_logger = WandbLogger(
            project=self.cfg.get("wandb", {}).get(
                "project", "ARCEME_kNDVI_Prediction"
            ),
            name=f"{self.cfg['experiment_name']}_final_refit",
            group=self.cfg["experiment_name"],
            job_type="final_refit",
            config=self.cfg,
            resume="allow",
        )

        if os.path.exists(self.exclude_csv_path):
            exclusion_artifact = wandb.Artifact(
                name=f"exclusion_list_{self.cfg['experiment_name']}_final_refit",
                type="dataset_metadata",
            )
            exclusion_artifact.add_file(self.exclude_csv_path)
            wandb_logger.experiment.log_artifact(exclusion_artifact)

        # There is no best validation score to log (as there is no validation data), so save the last epoch only
        final_checkpoint_callback = ModelCheckpoint(
            dirpath=final_ckpt_dir,
            filename="final-model-{epoch:03d}",
            save_top_k=0,
            save_last=True,
        )
        callbacks = [final_checkpoint_callback]
        if (
            self.cfg["training"]["optimizer"].get("warmup", {}).get("enabled", False)
        ):
            callbacks.insert(0, ConfigWarmupCallback(self.cfg))

        trainer = Trainer(
            accumulate_grad_batches=self.cfg["training"]["accumulate_grad_batches"],
            log_every_n_steps=10,
            gradient_clip_val=(
                self.cfg["training"]["gradient_clipping"]["value"]
                if self.cfg["training"]["gradient_clipping"]["enabled"]
                else None
            ),
            gradient_clip_algorithm=(
                self.cfg["training"]["gradient_clipping"]["algorithm"]
                if self.cfg["training"]["gradient_clipping"]["enabled"]
                else "norm"
            ),
            max_epochs=epochs,
            accelerator=self.cfg["training"]["accelerator"],
            devices=self.cfg["training"]["devices"],
            precision=self.cfg["training"]["precision"],
            logger=wandb_logger,
            callbacks=callbacks,
            enable_model_summary=True,
        )

        print_channel_info(
            self.v_cfg["s2"],
            self.v_cfg["s1"],
            self.v_cfg["era5"],
            self.v_cfg["static"],
            expected_input_channels=self.cfg["model"]["input_channels"],
        )
        if resume_ckpt:
            print(f"♻️ Resuming final refit from: {resume_ckpt}")
        trainer.fit(model, train_loader, ckpt_path=resume_ckpt)

        final_checkpoint = final_checkpoint_callback.last_model_path
        if not final_checkpoint or not os.path.exists(final_checkpoint):
            raise RuntimeError("Final refit finished but no last checkpoint was saved.")

        final_summary = {
            "final_checkpoint": final_checkpoint,
            "num_training_cubes": len(train_files),
            "epochs": epochs,
            "seed": self.global_seed,
            "source_cv_run": self.final_refit_cfg.get("source_cv_run"),
            "resumed_from_checkpoint": resume_ckpt,
            "early_stopping": False,
            "validation_used": False,
        }
        summary_path = os.path.join(self.run_dir, "final_refit_summary.json")
        with open(summary_path, "w") as f:
            json.dump(final_summary, f, indent=2)

        print(f"\n✅ Final refit complete. Checkpoint: {final_checkpoint}")
        print(f"📊 Final-refit summary saved: {summary_path}")

        del model, trainer, train_loader
        torch.cuda.empty_cache()
        wandb.finish()
        return final_summary

    def evaluate(
        self,
        ckpt_path,
        test_files=None,
        output_dir=None,
        plot_samples=True,
    ):
        """Evaluate a checkpoint and optionally redirect saved artefacts.

        A separate ``output_dir`` prevents out-of-fold exports from
        overwriting one another. ``plot_samples=False`` avoids loading
        validation cubes from the configured holdout directory.
        """
        print(f"\n🔍 Evaluating model from: {ckpt_path}")
        if test_files is None:
            test_files = self.prepare_data()

        evaluation_output_dir = os.path.abspath(output_dir or self.run_dir)
        os.makedirs(evaluation_output_dir, exist_ok=True)
        # The Lightning module reads the artefact destination from cfg in
        # on_validation_epoch_end.
        self.cfg["model"]["run_dir"] = evaluation_output_dir

        model = ConvLSTM_Model.load_from_checkpoint(ckpt_path, cfg=self.cfg)
        model.eval()

        model.is_testing_mode = bool(plot_samples)

        self.cfg["testing"]["save_tensors"] = True
        self.cfg["testing"]["save_metrics"] = True

        v_cfg = self.cfg["data"]["variables"]

        # Define fixed patches for validation
        test_tiles = get_val_tiles_auto(
            test_files, patch_size=self.cfg["data"]["patch_size"]
        )

        test_ds = ARCEME_Dataset(
            test_files,
            train=False,
            config=self.cfg,
            fixed_tiles=test_tiles,
            s2_vars=v_cfg["s2"],
            s1_vars=v_cfg["s1"],
            era5_vars=v_cfg["era5"],
            static_vars=v_cfg["static"],
            use_augmentation=False,
            context_length=self.cfg["data"]["context_length"],
            target_length=self.cfg["data"]["target_length"],
            patch_size=self.cfg["data"]["patch_size"],
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=self.cfg["data"]["data_loader"]["num_workers"],
            pin_memory=True,
        )

        trainer = Trainer(
            accelerator=self.cfg["training"]["accelerator"], devices=1, logger=False
        )

        with torch.inference_mode():  # or with torch.no_grad():
            results = trainer.validate(model, dataloaders=test_loader)

        return results

    def load_model(self, ckpt_path):
        """
        Loads the trained PyTorch Lightning model from a checkpoint.
        Useful for manual inference, notebook debugging, and plotting.
        """
        print(f"🔄 Loading model weights from: {ckpt_path}")

        # Load model
        model = ConvLSTM_Model.load_from_checkpoint(ckpt_path, cfg=self.cfg)

        # Eval mode
        model.eval()

        return model
