import torch
import torch.nn as nn


class GWDiceLoss(nn.Module):
    """
    Generalised Weighted Dice Loss
    """
    def __init__(
        self, 
        beta=1, 
        smooth=1e-5
    ):
        super(GWDiceLoss, self).__init__()
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, true):
        return generalised_weighted_dice_loss(pred, true, self.beta, self.smooth)


class BinaryDiceLoss(nn.Module):
    """
    Binary Weighted Dice Loss, where the predictions are thresholded before calculation
    """
    def __init__(
        self, 
        threshold=0.5, 
        beta=1, 
        smooth=1e-5
    ):
        super(BinaryDiceLoss, self).__init__()
        self.threshold = threshold
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, true):
        pred = torch.where(pred > self.threshold, 1, 0)
        return generalised_weighted_dice_loss(pred, true, self.beta, self.smooth)


def generalised_weighted_dice_loss(pred, true, beta = 1, smooth=1e-5):
    """
    Analogous to Weighted Beta F1 Score, where beta is the weighting factor for false negatives
    A beta of 1 is the same as regular unweighted generalised Dice Loss
    """
    pred = pred.view(-1)
    true = true.view(-1)
    
    tp = torch.sum(pred * true)
    fn = torch.sum((1 - pred) * true)
    fp = torch.sum(pred * (1 - true))

    weighting = (1 + beta ** 2)

    numerator = weighting * tp + smooth
    denominator = weighting * tp + (weighting - 1) * fn + fp + smooth
    dice = numerator / denominator

    return 1 - dice


def print_time_taken(start_time, end_time):
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"Training completed in {hours}h {minutes}m {seconds}s")