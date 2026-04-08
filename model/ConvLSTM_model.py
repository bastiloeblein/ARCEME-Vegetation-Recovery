import os
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from model.optimizers import get_opt_from_name
from torch.optim.lr_scheduler import ReduceLROnPlateau
from my_utils.losses import get_loss_from_name
from my_utils.visualization import plot_prediction_deltas

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
            ), f"SGED requires hidden_dims list length ({len(model_params['hidden_dims'])}) to match n_layers ({model_params['num_layers']})"  # SGED needs additional parameter  decoder_input_dim
            # SGED needs an explicit decoder input dimension calculation
            decoder_input_channels = cfg["model"]["future_channels"] + 1
            self.model = SGEDConvLSTM(
                decoder_input_dim=decoder_input_channels, **model_params
            )
        else:
            from model.ConvLSTM import SGConvLSTM

            self.model = SGConvLSTM(**model_params)

        # 3. 3. Loss & Optimizer configuration
        self.lr = cfg["training"]["optimizer"]["start_learn_rate"]
        self.training_loss = get_loss_from_name(
            cfg["training"]["training_loss"]["loss_function"]
        )
        self.alpha = cfg["training"]["training_loss"]["alpha"]
        self.beta = cfg["training"]["training_loss"]["beta"]

        # Just a try:
        self.scaling_factor = cfg["training"]["scaling_factor"]

        # self.validation_loss = get_loss_from_name(self.cfg["training"]["validation_loss"])

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

        # Model Prediction
        y_pred, y_delta_pred, baselines = self(
            x_ctx, y_true.size(1), x_fut, baseline_sample
        )

        # Loss Calculation
        # Primary loss: predicted kNDVI vs true kNDVI
        train_loss = self.training_loss(preds=y_pred, targets=y_true, mask=mask)

        # Secondary loss (optional): predicted delta vs true delta
        if self.beta > 0:
            y_delta_true = y_true - baselines

            # test!!
            # delta_loss = get_loss_from_name("l2")
            train_delta_loss = self.training_loss(
                preds=y_delta_pred * self.scaling_factor,
                targets=y_delta_true * self.scaling_factor,
                mask=mask,
            )

            # --- Spatial Gradient Loss (Struktur-Erzwinger) ---
            # X-Richtung (Horizontal)
            diff_x_pred = torch.diff(y_delta_pred, dim=-1)
            diff_x_true = torch.diff(y_delta_true, dim=-1)
            mask_x = mask[
                :, :, :, :, :-1
            ]  # Maske um 1 Pixel kürzen, da diff die Breite um 1 reduziert
            grad_loss_x = torch.sum(torch.abs(diff_x_pred - diff_x_true) * mask_x) / (
                mask_x.sum() + 1e-8
            )

            # Y-Richtung (Vertikal)
            diff_y_pred = torch.diff(y_delta_pred, dim=-2)
            diff_y_true = torch.diff(y_delta_true, dim=-2)
            # Korrekt für 5D [B, T, C, H, W]:
            mask_y = mask[:, :, :, :-1, :]
            grad_loss_y = torch.sum(torch.abs(diff_y_pred - diff_y_true) * mask_y) / (
                mask_y.sum() + 1e-8
            )

            # Kombinieren
            grad_loss = grad_loss_x + grad_loss_y
            train_delta_loss += 1 * grad_loss
        else:
            train_delta_loss = torch.tensor(0.0).to(self.device)

        # Combined loss weighted by alpha and beta
        combined_loss = self.alpha * train_loss + self.beta * train_delta_loss

        self.log(
            "train_loss",
            train_loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=bs,
            sync_dist=True,
        )
        self.log(
            "train_delta_loss",
            train_delta_loss,
            on_epoch=True,
            batch_size=bs,
            sync_dist=True,
        )
        self.log(
            "combined_loss", combined_loss, on_epoch=True, batch_size=bs, sync_dist=True
        )

        return combined_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation logic: same as training but collects metrics for epoch end.
        """
        x_ctx, x_fut, y_true, mask, meta, baseline_sample = batch
        bs = x_ctx.shape[0]

        # Forward pass
        y_pred, y_delta_pred, baselines = self(
            x_ctx, y_true.size(1), x_fut, baseline_sample
        )

        ### --- PERSISTENCE BASELINE METRICS ---
        persistence_pred = baselines.expand_as(y_true)
        diff_base = torch.abs(persistence_pred - y_true) * mask
        abs_err_base = diff_base.view(bs, -1).sum(dim=1).detach()
        sq_err_base = (diff_base**2).view(bs, -1).sum(dim=1).detach()
        y_pred_base_sum = (persistence_pred * mask).view(bs, -1).sum(dim=1).detach()

        # Basic metric collection (ignoring masked pixels)
        diff = torch.abs(y_pred - y_true) * mask
        abs_err = (
            diff.view(bs, -1).sum(dim=1).detach()
        )  # Reduction per sample in batch (Result has shape [B]) -> if 4 batches in step -> 1 value for each batch -> all errors in the batches are summed
        sq_err = (
            (diff**2).view(bs, -1).sum(dim=1).detach()
        )  # Same principle -> only here squared sum
        valid_pixels = mask.view(bs, -1).sum(dim=1).detach()  # Number of valid pixels

        # Sum of predictions and targets (GT) for calculation of bias
        y_pred_sum = (y_pred * mask).view(bs, -1).sum(dim=1).detach()
        y_true_sum = (y_true * mask).view(bs, -1).sum(dim=1).detach()

        # Store outputs for metric calculation at on_validation_epoch_end
        # Assign each batch to a cube_id -> so cube wise metric calculation is possible  at the end of each epoch
        ids = meta["cube_id"]
        for i in range(bs):
            if valid_pixels[i] > 0:
                self.validation_step_outputs.append(
                    {
                        "cube_id": ids[i],
                        "abs_err": abs_err[i],
                        "sq_err": sq_err[i],
                        "y_pred_sum": y_pred_sum[i],
                        "y_true_sum": y_true_sum[i],
                        "pixels": valid_pixels[i],
                        "y_true_raw": y_true[i][mask[i].bool()].detach().cpu(),
                        # Baseline stats (NEW)
                        "abs_err_base": abs_err_base[i],
                        "sq_err_base": sq_err_base[i],
                        "y_pred_base_sum": y_pred_base_sum[i],
                    }
                )

        # # Log histograms: (maybe delete)
        # if batch_idx < 3: # Logge Histogramme nur für die ersten 3 Batches
        #     # Maskiere die Daten, damit nur echte Vegetationspixel im Histogramm landen
        #     valid_mask = mask.bool()
        #     preds_filtered = y_pred[valid_mask].detach().cpu().numpy()
        #     trues_filtered = y_true[valid_mask].detach().cpu().numpy()

        #     self.logger.experiment.log({
        #         f"val/hist_pred_batch_{batch_idx}": wandb.Histogram(preds_filtered),
        #         f"val/hist_true_batch_{batch_idx}": wandb.Histogram(trues_filtered),
        #     }, commit=False)

        # Save patches for visualization of predictions over the epochs
        if self.current_epoch == 0:
            if not hasattr(self, "fixed_val_batches"):
                self.fixed_val_batches = []
                self.seen_cube_ids = set()
            if len(self.fixed_val_batches) < 3:
                for i in range(bs):
                    cid = meta["cube_id"][i]
                    total_pixels = mask[i].numel()
                    quality_ratio = valid_pixels[i] / total_pixels

                    # only new cubes are considered with minimum quality
                    if cid not in self.seen_cube_ids and quality_ratio > 0.6:

                        safe_batch = [
                            x_ctx.detach().cpu().clone(),
                            x_fut.detach().cpu().clone(),
                            y_true.detach().cpu().clone(),
                            mask.detach().cpu().clone(),
                            meta,  # No detach needed for dicts!
                            baseline_sample.detach().cpu().clone(),
                        ]
                        self.fixed_val_batches.append(safe_batch)
                        self.seen_cube_ids.add(cid)
                        print(
                            f"DEBUG: Fixed Patch {len(self.fixed_val_batches)} saved for ID {meta['cube_id'][0]}"
                        )

                        if len(self.fixed_val_batches) >= 3:
                            break

        # --- BASELINE CONSISTENCY ASSERT ---
        # This checks if the INITIAL baseline from the data loader is identical
        # to what we stored.
        # --- BASELINE CONSISTENCY CHECK & DEBUG ---
        if hasattr(self, "fixed_val_batches") and self.current_epoch > 0:
            for i in range(bs):
                current_id = meta["cube_id"][i]

                for anchor_batch in self.fixed_val_batches:
                    anchor_meta = anchor_batch[4]

                    # Wir loopen durch den Anker-Batch, um den passenden Patch zu finden
                    for j in range(len(anchor_meta["cube_id"])):
                        if (
                            meta["cube_id"][i] == anchor_meta["cube_id"][j]
                            and meta["top"][i] == anchor_meta["top"][j]
                            and meta["left"][i] == anchor_meta["left"][j]
                        ):

                            # Patch gefunden!
                            stored_baseline = anchor_batch[-1][j].to(self.device)
                            current_baseline = baseline_sample[i]

                            diff = torch.abs(current_baseline - stored_baseline)
                            max_diff = diff.max().item()

                            # --- DEBUG PRINTS (Korrigiert: j statt idx_in_anchor) ---
                            if (
                                max_diff > 1e-5 or batch_idx == 0
                            ):  # Print nur beim Fehler oder einmal am Anfang
                                print(
                                    f"\n--- DATA DRIFT DEBUG (Epoch {self.current_epoch}) ---"
                                )
                                print(f"Cube ID: {current_id}")
                                print(
                                    f"Stored Meta:  Top={anchor_meta['top'][j]}, Left={anchor_meta['left'][j]}"
                                )
                                print(
                                    f"Current Meta: Top={meta['top'][i]}, Left={meta['left'][i]}"
                                )
                                print(f"Max Diff in Baseline: {max_diff:.6f}")

                            # --- WANDB VISUALISIERUNG BEI DRIFT ---
                            if max_diff > 1e-5:
                                diff_img = diff.squeeze().cpu().numpy()
                                self.logger.experiment.log(
                                    {
                                        f"debug/drift_map_{current_id}": wandb.Image(
                                            diff_img,
                                            caption=f"Diff Map (Max: {max_diff:.4f}) Epoch {self.current_epoch}",
                                        ),
                                        f"debug/baseline_comparison_{current_id}": [
                                            wandb.Image(
                                                stored_baseline.squeeze().cpu().numpy(),
                                                caption="Stored (Epoch 0)",
                                            ),
                                            wandb.Image(
                                                current_baseline.squeeze()
                                                .cpu()
                                                .numpy(),
                                                caption=f"Current (Epoch {self.current_epoch})",
                                            ),
                                        ],
                                    }
                                )

                                # Der ultimative Stopp
                                assert max_diff < 1e-5, (
                                    f"Baseline Mismatch für {current_id}! "
                                    f"Diff: {max_diff:.4f}. Check WandB!"
                                )
        # if hasattr(self, "fixed_val_batches") and self.current_epoch > 0:
        #     for i in range(bs):
        #         if meta["cube_id"][i] == self.fixed_val_batches[0][4]["cube_id"][i]: # ID-Vergleich
        #              stored_baseline = self.fixed_val_batches[0][-1][i].to(self.device)
        #              current_baseline = baseline_sample[i]
        #              diff = torch.abs(current_baseline - stored_baseline).max()
        #              assert diff < 1e-5, f"Baseline mismatch! Data drift at epoch {self.current_epoch}. Max diff: {diff}"

        # stored_baseline = self.fixed_val_batches[0][-1].to(self.device)
        # # Compare first sample of current batch with stored first sample
        # diff = torch.abs(baseline_sample[0] - stored_baseline[0]).max()
        # # If this fails, your DataLoader shuffles the validation set differently each epoch!
        # assert diff < 1e-5, f"Baseline mismatch! Data drift at epoch {self.current_epoch}. Max diff: {diff}"

    def on_validation_epoch_end(self):
        """
        Is executed at the end of each epoch. So when the model has seen all data.
        Aggregate validation metrics across all batches and calculate per-cube statistics.
        """
        if not self.validation_step_outputs:
            return

        # 1. Grouping by Cube IDs
        # Aggregate errors and pixel counts for each unique cube to get unbiased overall cube metrics
        cubes = {}
        for out in self.validation_step_outputs:
            cid = out["cube_id"]
            if cid not in cubes:
                cubes[cid] = {
                    "abs_err": 0,
                    "sq_err": 0,
                    "abs_err_base": 0,
                    "sq_err_base": 0,
                    "y_pred_base_sum": 0,
                    "pixels": 0,
                    "y_pred_sum": 0,
                    "y_true_sum": 0,
                    "y_true_list": [],
                }
            for key in [
                "abs_err",
                "sq_err",
                "y_pred_sum",
                "abs_err_base",
                "sq_err_base",
                "y_pred_base_sum",
                "pixels",
                "y_true_sum",
            ]:
                cubes[cid][key] += out[key]
            cubes[cid]["y_true_list"].append(out["y_true_raw"])

        # 2. Calculate metrics per cube
        # Using lists to store grand mean components
        gm_metrics = {
            k: []
            for k in [
                "l1",
                "mse",
                "bias",
                "r2",
                "nnse",
                "l1_base",
                "mse_base",
                "bias_base",
                "r2_base",
                "nnse_base",
            ]
        }

        # Define minimum pixel threshold to ensure statistical significance per cube
        min_pixel_threshold = self.cfg["training"]["validation"].get(
            "min_pixel_threshold", 10000
        )  # TODO: This should be handled before.

        eps = 1e-8
        for cid, data in cubes.items():
            # Skip cubes with insufficient valid data to avoid noisy outliers
            if data["pixels"] < min_pixel_threshold:
                continue

            # Concatenate all pixels of the cube for variance
            y_all = torch.cat(data["y_true_list"])
            variance = torch.var(y_all) + eps

            # --- MODEL METRICS ---
            mse_m = data["sq_err"] / data["pixels"]
            r2_m = 1 - (mse_m / variance)
            gm_metrics["l1"].append(data["abs_err"] / data["pixels"])
            gm_metrics["mse"].append(mse_m)
            gm_metrics["bias"].append(
                (data["y_pred_sum"] - data["y_true_sum"]) / data["pixels"]
            )
            gm_metrics["r2"].append(r2_m)
            gm_metrics["nnse"].append(1 / (2 - r2_m))

            # --- BASELINE METRICS (NEW) ---
            mse_b = data["sq_err_base"] / data["pixels"]
            r2_b = 1 - (mse_b / variance)
            gm_metrics["l1_base"].append(data["abs_err_base"] / data["pixels"])
            gm_metrics["mse_base"].append(mse_b)
            gm_metrics["bias_base"].append(
                (data["y_pred_base_sum"] - data["y_true_sum"]) / data["pixels"]
            )
            gm_metrics["r2_base"].append(r2_b)
            gm_metrics["nnse_base"].append(1 / (2 - r2_b))

        # Log Grand Means
        for k, values in gm_metrics.items():
            if values:
                metric_name = f"val_gm_{k}"
                self.log(metric_name, torch.stack(values).mean(), sync_dist=True)

        # Log current learning rate from optimizer
        opt = self.optimizers()
        current_lr = opt.param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=True, on_epoch=True)

        ## kann auch raus nur für test
        # --- Visualizations ---
        # Log visual samples every X epochs
        log_interval = 10
        if self.current_epoch % log_interval == 0 and hasattr(
            self, "fixed_val_batches"
        ):

            for i, val_batch in enumerate(self.fixed_val_batches):

                batch_cuda = []
                for t in val_batch:
                    if torch.is_tensor(t):
                        batch_cuda.append(t.to(self.device))
                    else:
                        batch_cuda.append(t)

                x_ctx, x_fut, y_true, mask, meta, baseline_sample = batch_cuda

                self.visualize_and_log_hidden(batch_cuda)

                with torch.no_grad():
                    y_pred, y_delta_pred, baselines = self(
                        x_ctx, y_true.size(1), x_fut, baseline_sample
                    )

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

                plt.close(fig)

        # --- Memory Cleanup ---
        self.validation_step_outputs.clear()

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
        monitor_key = f"{self.cfg['training']['validation']['monitor']['split']}_{self.cfg['training']['validation']['monitor']['metric']}"

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
        # Diese Methode wird nach jedem Gradienten-Schritt aufgerufen
        if self.global_step % 50 == 0:  # Alle 10 Schritte loggen, um TB nicht zu fluten
            # In deiner on_after_backward Methode:
            grad_dict = {}
            for name, param in self.named_parameters():
                if param.grad is not None:
                    # Nur den Mittelwert der absoluten Gradienten loggen
                    grad_dict[f"grad_norms/{name}"] = param.grad.abs().mean().item()
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
