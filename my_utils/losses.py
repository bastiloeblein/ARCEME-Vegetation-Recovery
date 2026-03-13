import torch
import torch.nn as nn

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