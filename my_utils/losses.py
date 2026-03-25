import torch
import torch.nn as nn


def get_loss_from_name(loss_name):
    if loss_name == "l2":
        return MaskedMSELoss()
    elif loss_name == "l1":
        return MaskedL1Loss()


# Losses are currently following the following hierachy:
# 1. Pixel level: For each of the patch_size x patch_size pixels, calculate the error between prediction and target, then average over all valid pixels (using mask).
# 2. Timestep level: For each of the target_length timesteps, calculate the pixel-level loss, then average over the target_length timesteps.
# 3. Patch level: For each patch, calculate the timestep-level loss
# 4. Batch level: Average the patch-level losses over the batch. e.g. batch_size = 4: BatchLoss = (PatchLoss1 + PatchLoss2 + PatchLoss3 + PatchLoss4) / 4
# 5. Epoch level:
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
        preds:  [B, T, C, H, W] - Model prediction
        targets: [B, T, C, H, W] - Ground Truth
        mask:   [B, T, 1, H, W] - Binary Mask (1 = valid/Vegetation, 0 = clouded/No vegetation)
        """
        # 1. Ensure that mask has same type
        mask = mask.to(preds.dtype)

        # 2. Calculate absolute errors
        abs_errors = torch.abs(preds - targets)

        # 3. Apply Mask
        masked_abs_errors = abs_errors * mask

        # 4. Calculate mean over valid pixels (avoid division by zero)
        loss = masked_abs_errors.sum() / (mask.sum() + 1e-8)

        return loss
