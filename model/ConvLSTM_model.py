import os
from model.optimizers import get_opt_from_name
import pytorch_lightning as pl
from model.ConvLSTM import ConvLSTM
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from my_utils.losses import MaskedMSELoss, MaskedL1Loss
from my_utils.visualization import plot_prediction_deltas

# delete
import matplotlib.pyplot as plt

class ConvLSTM_Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # 1. Dynamische Kanäle aus der Config ziehen
        # Vergangenheit (z.B. S1 + S2 + Wetter + Statics = 34)
        input_channels = cfg["model"]["input_channels"]

        # Zukunft (z.B. Wetter + Statics = 24)
        # Wir addieren +1, weil die Vorhersage (kNDVI) im Decoder dazugeklebt wird
        decoder_input_channels = cfg["model"]["future_channels"] + 1

        # Hier rufen wir dein ConvLSTM aus Schritt 2 auf
        # Wir ziehen alle Werte direkt aus der cfg-Datei
        self.model = ConvLSTM(
            input_dim=input_channels,  # 34
            decoder_input_dim=decoder_input_channels,
            output_dim=cfg["model"]["output_channels"],  # 1
            hidden_dims=cfg["model"]["hidden_channels"],  # z.B. [64, 64]
            num_layers=cfg["model"]["n_layers"],  # z.B. 2
            kernel_size=cfg["model"]["kernel"],  # 3
            dilation=cfg["model"]["dilation_rate"],  # 1
            baseline=cfg["model"]["baseline"],  # "last_frame"
        )

        # 3. Loss & Metriken
        # Wir nutzen MSE, aber 'none', damit wir die Maske (Vegetation) anwenden können
        self.mse_criterion = MaskedMSELoss()
        self.l1_criterion = MaskedL1Loss()
        self.lr = cfg["training"]["start_learn_rate"]

    def forward(self, x_ctx, prediction_count, non_pred_feat, baseline_sample):

        preds, pred_deltas, baselines = self.model(
            x_ctx, non_pred_feat=non_pred_feat, prediction_count=prediction_count, baseline_sample=baseline_sample
        )

        return preds, pred_deltas, baselines

    def training_step(self, batch, batch_idx):
        """
        Ein Batch kommt aus deinem Dataset.py und hat (B, T, C, H, W)
        """
        x_ctx, x_fut, y_true, mask, meta, baseline_sample = batch
        bs = x_ctx.shape[0]
        # x_ctx: Context           (B, T_ctx, C_in, 256, 256)
        # x_fut: Climate in target (B, T_target, C_fut, 256, 256)
        # y_true: GT kNDVI         (B, T_target, 1 (kNDVI), 256, 256)
        # mask: Vegetation mask    (B, T_target, 1, 256, 256)

        # --- ASSERT 1: Batch Konsistenz ---
        assert x_ctx.size(0) == y_true.size(0), "Batch Size Mismatch zwischen Input und Target!"
        assert y_true.size(2) == 1, "Target kNDVI sollte nur 1 Kanal haben!"
        assert mask.size(1) == y_true.size(1), "Masken und Target sollten die gleiche Anzahl an Zeitschritten haben!"

        # --- 1. GLOBAL PIXEL AVAILABILITY & SKIP-CHECK ---
        total_pixels = mask.numel()
        valid_pixel_sum = mask.sum()
        valid_ratio = valid_pixel_sum / (total_pixels + 1e-8)

        # Log valid_pixel_ratio
        self.log("Train/Pixel_Availability", valid_ratio, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

        # Weniger als 5% der pixel -> skip
        if valid_ratio < 0.05: 
            self.log("Train/Mean_skipped_batches", 1.0,  on_step=False, on_epoch=True, reduce_fx="mean", batch_size=bs)
            return None
        else:
            self.log("Train/Mean_skipped_batches", 0.0, on_step=False, on_epoch=True, reduce_fx="mean", batch_size=bs)

        # --- 2. FORWARD PASS ---
        y_pred, y_delta_pred, baselines = self(x_ctx, y_true.size(1), x_fut, baseline_sample)

        # --- 3. LOSS CALCULATION ---
        loss, delta_loss = self.compute_timestep_filtered_loss(y_pred, y_true, y_delta_pred, baselines, mask)
        
        # --- 4. LOGGING ---
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=bs)
        self.log("train_delta_loss", delta_loss, on_epoch=True, batch_size=bs)

        for t in range(y_true.size(1)):
            step_vis = mask[:, t].sum() / (mask[:, t].numel() + 1e-8)
            self.log(f"Train/T{t}/Pixel_Availability", step_vis, on_step=False, on_epoch=True, batch_size=bs)

        return loss


    def validation_step(self, batch, batch_idx):
        x_ctx, x_fut, y_true, mask, meta, baseline_sample = batch
        bs = x_ctx.shape[0]
        # Availability check - we validate on all batches!!
        valid_ratio = mask.sum() / (mask.numel() + 1e-8)
        self.log("Validation/Pixel_Availability", valid_ratio, on_epoch=True, prog_bar=True, batch_size=bs)

        y_pred, y_delta_pred, baselines = self(x_ctx, y_true.size(1), x_fut, baseline_sample)

        loss, delta_loss = self.compute_timestep_filtered_loss(y_pred, y_true, y_delta_pred, baselines, mask)
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=bs)
        self.log("val_delta_loss", delta_loss, on_epoch=True, batch_size=bs)

        for t in range(y_true.size(1)):
            m = mask[:, t].bool()
            valid_fraction = m.sum().float() / m.numel()
            self.log(f"Validation/T{t}/Pixel_Availability", valid_fraction, batch_size=bs)

        return loss


    def configure_optimizers(self):
        optimizer = get_opt_from_name(self.cfg["training"]["optimizer"],
                                      params=self.parameters(),
                                      lr=self.cfg["training"]["start_learn_rate"])

        # Scheduler: Verringert die Lernrate, wenn der val_loss nicht mehr sinkt
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.cfg["training"]["lr_factor"],
            patience=self.cfg["training"]["patience"],
            threshold=self.cfg["training"]["lr_threshold"]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Nur alle X Batches oder einmal pro Epoch, um das Log nicht zu fluten
        if batch_idx == 0: 
            x_ctx, x_fut, y_true, mask, meta, baseline_sample = batch
            bs = x_ctx.shape[0]
            # --- ASSERT: Device Check ---
            assert x_ctx.device == self.device, "Input ist nicht auf dem korrekten Device!"

            # Forward pass
            y_pred, y_delta_pred, baselines = self(x_ctx, y_true.size(1), x_fut, baseline_sample)
            
            y_delta_true = y_true - baselines
            
            # Check pro Timestep (T=5)
            for t in range(y_true.size(1)):
                m = mask[:, t].bool()

                if m.sum() > 1000:
                    true_mag = y_delta_true[:, t][m].abs().mean()
                    pred_mag = y_delta_pred[:, t][m].abs().mean()

                    # Der Bias-Plot: 1.0 = Perfekt, < 1.0 = Unterschätzung
                    # Wir addieren 1e-8 gegen Division durch Null
                    bias = pred_mag / (true_mag + 1e-8)

                else:
                    true_mag, pred_mag, bias  = float("nan"), float("nan"), float("nan")
                
                self.log(f"Validation/T{t}/True_Delta_Step", true_mag, batch_size=bs)
                self.log(f"Validation/T{t}/Pred_Delta_Step", pred_mag, batch_size=bs)
                self.log(f"Validation/T{t}/Bias_Ratio_Step", bias, batch_size=bs)

            if self.current_epoch % 5 == 0:  # Every 5 epochs

                os.makedirs("plots", exist_ok=True)

                fig = plot_prediction_deltas(
                    y_true, y_pred, y_delta_pred, baselines, mask,
                    batch_idx, self.current_epoch, 
                    save_path="plots"
                )
                fold_idx = getattr(self, "fold_idx", 0)
                self.logger.experiment.add_figure(
                    f"Visuals_Fold_{fold_idx}/Deltas",
                    fig, 
                    global_step=self.current_epoch
                )


    def compute_standard_loss(self, y_pred, y_true, y_delta_pred, baselines, mask):
        """Combined loss loss_img_l1 AND WEIGHTED delta_loss"""
        loss_img_l1 = self.l1_criterion(y_pred, y_true, mask)
        y_delta_true = y_true - baselines
        loss_delta = self.l1_criterion(y_delta_pred, y_delta_true, mask)
        
        alpha, beta = (0.0, 20.0) if self.current_epoch < 5 else (0.1, 20.0)
        return alpha * loss_img_l1 + beta * loss_delta, loss_delta

    def compute_timestep_filtered_loss(self, y_pred, y_true, y_delta_pred, baselines, mask):
        """Combined loss, but empty timesteps will be filtered based on mask"""
        loss_img_l1 = self.l1_criterion(y_pred, y_true, mask)
        y_delta_true = y_true - baselines
        
        # Punktweise Differenz
        diff_abs = (y_delta_pred - y_delta_true).abs() * mask  # L1 
        diff_sq = (y_delta_pred - y_delta_true)**2 * mask      # MSE
        
        # Pixel pro Zeitschritt zählen (B, T, C, H, W) -> Summe über alles außer T
        pixels_per_t = mask.sum(dim=(0, 2, 3, 4))
        
        # Logik: Nur Zeitschritte mit > 100 Pixeln tragen zum Gradienten bei
        # Wir 'nullen' die Differenzen für leere Zeitschritte
        valid_t_mask = (pixels_per_t > 100).float()
        # Wir müssen die Maske auf die Shape (1, T, 1, 1, 1) bringen für Broad-casting
        valid_t_mask_reshaped = valid_t_mask.view(1, -1, 1, 1, 1)
        
        filtered_diff = diff_sq * valid_t_mask_reshaped
        
        # Normalisierung: Summe der Fehler / Summe der validen Pixel
        # Wir addieren 1e-8 um Division durch Null zu vermeiden
        safe_loss_delta = filtered_diff.sum() / (mask.sum() + 1e-8)
        
        alpha, beta = (0.0, 50.0) if self.current_epoch < 5 else (0.1, 50.0)
        combined_loss = alpha * loss_img_l1 + beta * safe_loss_delta
        
        return combined_loss, safe_loss_delta

    def visualize_hidden_states(self, x_ctx, x_fut, baseline_sample):
        self.model.eval()
        with torch.no_grad():
            # Wir brauchen Zugriff auf die internen Layer deines ConvLSTM
            # Angenommen, dein ConvLSTM speichert die hidden states in einer Liste 
            # oder gibt sie am Ende des Encoders zurück.
            
            # Falls dein Modell die Hidden States nicht zurückgibt, nutze einen Hook:
            activations = {}
            def get_activation(name):
                def hook(model, input, output):
                    # output bei ConvLSTM ist oft (hidden_state, cell_state)
                    if isinstance(output, tuple):
                        activations[name] = output[0].detach()
                    else:
                        activations[name] = output.detach()
                return hook

            # Registriere den Hook am letzten Encoder-Layer (beispielhafter Name)
            handle = self.model.encoder.cell_list[-1].register_forward_hook(get_activation('last_hidden'))
            
            # Einmal durchlaufen lassen
            _ = self(x_ctx, 5, x_fut, baseline_sample)
            handle.remove()

            hidden = activations['last_hidden'] # Shape: (B, Hidden_Channels, H, W)
            
            # Visualisierung der ersten 8 Hidden Channels des ersten Batch-Samples
            fig, axes = plt.subplots(2, 4, figsize=(15, 8))
            for i in range(8):
                ax = axes[i//4, i%4]
                # Normalisiere für die Anzeige, um auch kleine Signale zu sehen
                h_img = hidden[0, i].cpu().numpy()
                im = ax.imshow(h_img, cmap='viridis')
                ax.set_title(f"Channel {i}\nMax: {h_img.max():.4f}")
                plt.colorbar(im, ax=ax)
            
            plt.tight_layout()

            fig.savefig(f"hidden_states_epoch_{self.current_epoch}.png")
            plt.close(fig)
            
            return fig
        
    def visualize_and_log_hidden(self, batch):
        x_ctx, x_fut, y_true, mask, meta, baseline_sample = batch
        
        # Deine Funktion von oben aufrufen
        fig = self.visualize_hidden_states(x_ctx, x_fut, baseline_sample)
        
        # In TensorBoard speichern
        fold_idx = getattr(self, "fold_idx", 0)
        self.logger.experiment.add_figure(
            f"Debug/Hidden_States_Fold_{fold_idx}", 
            fig, 
            global_step=self.current_epoch
        )
        plt.close(fig) # Wichtig, sonst läuft der RAM voll!