import torch
import torch.nn as nn


def get_loss_from_name(loss_name):
    if loss_name == "l2":
        return MaskedMSELoss()
    elif loss_name == "l1":
        return MaskedL1Loss()


# Mathematical hierarchy:
# 1. Global Summation: Calculate the squared error for every single valid pixel across all patches in the batch and all timesteps. Sum them all together into one massive number.
# 2. Global Normalization: Count the total number of valid pixels across the entire batch and all timesteps.
# 3. Final Loss: Divide the massive sum (1) by the total valid pixel count (2).


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, preds, targets, mask):
        """
        preds:  [B, T, C, H, W] - Prediction of model
        targets: [B, T, C, H, W] - Ground Truth
        mask:   [B, T, 1, H, W] - Binary Mask (1 = valid/Vegetation, 0 = clouded/No vegetation)
        """
        # 1. Ensure that mask has same type
        mask = mask.to(preds.dtype)

        # 2. Calculate squared errors
        squared_errors = (preds - targets) ** 2

        # 3. Apply Mask
        masked_squared_errors = squared_errors * mask

        # 4. Calculate mean over valid pixels (avoid division by zero)
        loss = masked_squared_errors.sum() / (mask.sum() + 1e-8)

        return loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, preds, targets, mask):
        """
        preds:  [B, T, 1, H, W] - Model prediction
        targets: [B, T, 1, H, W] - Ground Truth
        mask:   [B, T, 1, H, W] - Binary Mask (1 = valid/Vegetation, 0 = clouded/No vegetation)
        """
        assert (
            preds.shape == targets.shape == mask.shape
        ), "Shapes of preds, targets and mask must be the same"
        # 1. Ensure that mask has same type
        mask = mask.to(preds.dtype)

        # 2. Calculate absolute errors
        abs_errors = torch.abs(preds - targets)

        # 3. Apply Mask
        masked_abs_errors = abs_errors * mask

        # 4. Calculate mean over valid pixels (avoid division by zero)
        loss = masked_abs_errors.sum() / (mask.sum() + 1e-8)

        # # --- Deep Debugging ---
        # with torch.no_grad():
        #     valid_idx = mask > 0.5 # Boolean Maske
        #     if valid_idx.any():
        #         v_preds = preds[valid_idx]
        #         v_targets = targets[valid_idx]
        #         v_errors = abs_errors[valid_idx]

        #         # print(f"\n--- 🔎 LOSS DEBUG (Epoche {self.current_epoch}) ---")
        #         print(f"Valid Pixels: {valid_idx.sum().item()} / {mask.numel()}")
        #         print(f"Loss (Mean):  {loss.item():.6f}")

        #         print(f"PREDS  | Mean: {v_preds.mean():.4f} | Std: {v_preds.std():.4f} | Min: {v_preds.min():.4f} | Max: {v_preds.max():.4f}")
        #         print(f"TARGET | Mean: {v_targets.mean():.4f} | Std: {v_targets.std():.4f} | Min: {v_targets.min():.4f} | Max: {v_targets.max():.4f}")
        #         print(f"ERRORS | Mean: {v_errors.mean():.4f} | Max Error: {v_errors.max():.4f}")

        #         # Check: Ist der Bias-Gradient mathematisch plausibel?
        #         # Der theoretische L1-Gradient für den Bias ist: (Anzahl_zu_hoch - Anzahl_zu_niedrig) / Gesamt_Valid
        #         too_high = (v_preds > v_targets).sum().item()
        #         too_low = (v_preds < v_targets).sum().item()
        #         expected_bias_grad_direction = (too_high - too_low) / valid_idx.sum().item()
        #         print(f"Theoretischer Bias-Druck: {expected_bias_grad_direction:.4f} (Sollte zwischen -1 und 1 liegen)")

        return loss
