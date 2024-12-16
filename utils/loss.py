import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class UncertaintyAwareCrossEntropyLoss(nn.Module):
    def __init__(self,ignore_index=255,reduction='none'):
        super(UncertaintyAwareCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets, uncertainty_masks):
        ce_loss = F.cross_entropy(inputs,targets,reduction='none',ignore_index=self.ignore_index)
        #uncertainty_masks = torch.transpose(uncertainty_masks,1,2)
        uncertainty_aware_cross_entropy_loss = torch.mul(ce_loss,(1+uncertainty_masks)**2)
        return uncertainty_aware_cross_entropy_loss.mean()

class Uce(nn.Module):
    def __init__(self,ignore_index=255,reduction='none',alpha=0):
        super(Uce, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, sigma, inputs, targets):
        loss = ((1+sigma)**self.alpha) * F.cross_entropy(inputs,targets,reduction='none',ignore_index = self.ignore_index)
        return torch.sum(loss)/torch.numel(loss)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass of Dice Loss for multi-class segmentation (n_classes=2).
        
        Args:
            inputs (Tensor): Model predictions (logits) of shape [B, 2, H, W].
            targets (Tensor): Ground truth masks of shape [B, H, W] (0 for background, 1 for foreground).

        Returns:
            Tensor: Dice loss value.
        """
        # Extract foreground channel (class 1) from inputs
        inputs = inputs[:, 1, :, :]  # Shape: [B, H, W]

        # Apply sigmoid to the logits to get probabilities
        inputs = torch.sigmoid(inputs)

        # Ensure targets have the same shape as inputs
        # (targets are assumed to be binary masks of shape [B, H, W])
        targets = targets.float()  # Ensure targets are float for multiplication

        # Flatten tensors to calculate Dice loss
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)

        # Compute Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)

        # Dice loss is 1 - Dice coefficient
        dice_loss = 1 - dice_coeff

        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, preds, targets):
        ce = self.ce_loss(preds, targets)
        dice = self.dice_loss(preds, targets)
        return self.alpha * ce + (1 - self.alpha) * dice