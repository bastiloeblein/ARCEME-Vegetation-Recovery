import os
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from model.optimizers import get_opt_from_name
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
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
        }

        # Initialize the specific model class
        if model_type == "SGED":
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
            self.scaling_factor = 10.0
            train_delta_loss = self.training_loss(
                preds=y_delta_pred * self.scaling_factor,
                targets=y_delta_true * self.scaling_factor,
                mask=mask,
            )

            # train_delta_loss = self.training_loss(preds=y_delta_pred, targets=y_delta_true, mask=mask)
        else:
            train_delta_loss = torch.tensor(0.0).to(self.device)

        # Combined loss weighted by alpha and beta
        combined_loss = self.alpha * train_loss + self.beta * train_delta_loss

        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True, batch_size=bs)
        self.log("train_delta_loss", train_delta_loss, on_epoch=True, batch_size=bs)
        self.log("combined_loss", combined_loss, on_epoch=True, batch_size=bs)

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
                    }
                )

        ###### Kann raus#########################
        if batch_idx == 2:
            self.example_val_batch = batch
        #########################################

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
                    "pixels": 0,
                    "y_pred_sum": 0,
                    "y_true_sum": 0,
                    "y_true_list": [],
                }

            cubes[cid]["abs_err"] += out["abs_err"]
            cubes[cid]["sq_err"] += out["sq_err"]
            cubes[cid]["pixels"] += out["pixels"]
            cubes[cid]["y_pred_sum"] += out["y_pred_sum"]
            cubes[cid]["y_true_sum"] += out["y_true_sum"]
            cubes[cid]["y_true_list"].append(out["y_true_raw"])

        # 2. Calculate metrics per cube
        # Using lists to store grand mean components
        cube_metrics = {"l1": [], "mse": [], "nnse": [], "r2": [], "bias": []}

        # Define minimum pixel threshold to ensure statistical significance per cube
        min_pixel_threshold = self.cfg["training"]["validation"].get(
            "min_pixel_threshold", 10000
        )  # TODO: This should be handled in data loader!!

        for cid, data in cubes.items():
            # Skip cubes with insufficient valid data to avoid noisy outliers
            if data["pixels"] < min_pixel_threshold:
                continue

            # A) Basic Pixel-Weighted Metrics
            l1_cube = data["abs_err"] / data["pixels"]
            mse_cube = data["sq_err"] / data["pixels"]

            # B) Mean Bias Error (MBE)
            # Positive = Overestimation, Negative = Underestimation
            bias_cube = (data["y_pred_sum"] - data["y_true_sum"]) / data["pixels"]

            # C) Variance for R2 & NNSE
            # Concatenate all true values for this cube to calculate variance
            y_all = torch.cat(data["y_true_list"])

            # Ensure there are enough data points for variance calculation
            if y_all.numel() > 1:
                variance = torch.var(y_all)
            else:
                variance = torch.tensor(0.0)

            # D) R2 Score & Normalized Nash-Sutcliffe Efficiency (NNSE)
            # Add a small epsilon to prevent division by zero if variance is zero
            # R2 = 1 - (MSE / Variance)
            eps = 1e-8
            r2_cube = 1 - (mse_cube / (variance + eps))

            # NNSE n (s. Pellicer-Valero)
            nnse_cube = 1 / (2 - r2_cube)

            # Now put all individual cube metrics in dict
            cube_metrics["l1"].append(l1_cube)
            cube_metrics["mse"].append(mse_cube)
            cube_metrics["bias"].append(bias_cube)
            cube_metrics["r2"].append(r2_cube)
            cube_metrics["nnse"].append(nnse_cube)

        # 3. Grand Mean Calculation (Averaging across cubes)
        # Each cube carries the same weight in the final score
        if len(cube_metrics["l1"]) > 0:
            # Convert lists to tensors for averaging
            gm_l1 = torch.stack(cube_metrics["l1"]).mean()
            gm_mse = torch.stack(cube_metrics["mse"]).mean()
            gm_bias = torch.stack(cube_metrics["bias"]).mean()
            gm_r2 = torch.stack(cube_metrics["r2"]).mean()
            gm_nnse = torch.stack(cube_metrics["nnse"]).mean()

            # 4. Final Logging to Progress Bar and Logger
            self.log("val_gm_loss", gm_l1, prog_bar=True, sync_dist=True)
            self.log("val_gm_mse", gm_mse, sync_dist=True)
            self.log("val_gm_bias", gm_bias, sync_dist=True)
            self.log("val_gm_r2", gm_r2, sync_dist=True)
            self.log("val_gm_nnse", gm_nnse, prog_bar=True, sync_dist=True)

        # 5. Log current learning rate from optimizer
        opt = self.optimizers()
        current_lr = opt.param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=True, on_epoch=True)

        ## kann auch raus nur für test
        # --- Visualizations ---
        # Log visual samples every X epochs
        log_every_n_epochs = 5
        if self.current_epoch % log_every_n_epochs == 0 and hasattr(
            self, "example_val_batch"
        ):
            self.visualize_and_log_hidden(self.example_val_batch)

            x_ctx, x_fut, y_true, mask, meta, baseline_sample = self.example_val_batch
            y_pred, y_delta_pred, baselines = self(
                x_ctx, y_true.size(1), x_fut, baseline_sample
            )

            # Plot deltas
            fig_deltas = plot_prediction_deltas(
                y_true,
                y_pred,
                y_delta_pred,
                baselines,
                mask,
                0,
                self.current_epoch,
                save_path="plots",
            )
            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(
                    {
                        "Visuals/Deltas": fig_deltas,
                        "epoch": self.current_epoch,
                        "global_step": self.global_step,
                    }
                )
            else:
                self.logger.experiment.add_figure(
                    f"Visuals/Deltas_Epoch_{self.current_epoch}",
                    fig_deltas,
                    global_step=self.current_epoch,
                )
            plt.close(fig_deltas)

        # --- Memory Cleanup ---
        # Clear the list to free memory for the next validation epoch
        self.validation_step_outputs.clear()
        if hasattr(self, "example_val_batch"):
            del self.example_val_batch

    def configure_optimizers(self):
        """
        Setup optimizer and learning rate scheduler.
        """
        optimizer = get_opt_from_name(
            self.cfg["training"]["optimizer"]["name"],
            params=self.parameters(),
            lr=self.cfg["training"]["optimizer"]["start_learn_rate"],
        )

        # Get config parameters
        warmup_cfg = self.cfg["training"]["optimizer"].get("warmup", {})
        do_warmup = warmup_cfg.get("enabled", False)
        warmup_epochs = warmup_cfg.get("epochs", 3)

        # Monitor metric that should defines validation performance
        monitor_key = f"{self.cfg['training']['validation']['monitor']['split']}_{self.cfg['training']['validation']['monitor']['metric']}"

        # Create main scheduler
        main_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=self.cfg["training"]["validation"]["monitor_mode"],
            factor=self.cfg["training"]["optimizer"]["lr_factor"],
            patience=self.cfg["training"]["optimizer"]["patience"],
            threshold=self.cfg["training"]["optimizer"]["lr_threshold"],
        )

        if do_warmup:
            # Warmup scheduler: for first epochs climbs from 0.01 to 100% of the target LR
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,  # Startet bei 1% der Ziel-LR
                end_factor=1.0,
                total_iters=warmup_epochs,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": warmup_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": main_scheduler, "monitor": monitor_key},
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

        fold_idx = getattr(self, "fold_idx", 0)

        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(
                {
                    f"Debug/Hidden_States_Fold_{fold_idx}": fig,
                    "epoch": self.current_epoch,
                    "global_step": self.global_step,
                }
            )
        else:
            self.logger.experiment.add_figure(
                f"Debug/Hidden_States_Fold_{fold_idx}_{self.current_epoch}",
                fig,
                global_step=self.current_epoch,
            )
        plt.close(fig)

    def on_after_backward(self):
        # Diese Methode wird nach jedem Gradienten-Schritt aufgerufen
        if self.global_step % 10 == 0:  # Alle 10 Schritte loggen, um TB nicht zu fluten
            # In deiner on_after_backward Methode:
            if isinstance(self.logger, WandbLogger):
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        self.logger.experiment.log(
                            {
                                f"gradients/{name}": wandb.Histogram(
                                    param.grad.cpu().detach().numpy()
                                )
                            },
                            commit=False,
                        )
            else:
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        self.logger.experiment.add_histogram(
                            f"Gradients/{name}", param.grad, self.global_step
                        )
                        self.logger.experiment.add_histogram(
                            f"Weights/{name}", param.data, self.global_step
                        )
