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
        self.beta_sq = beta ** 2
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        numerator = (self.beta_sq + 1) * torch.sum(pred * target) + self.smooth
        denominator = self.beta_sq * torch.sum(target) + torch.sum(pred) + self.smooth
        dice = numerator / denominator

        return 1 - dice


class BWDiceLoss(nn.Module):
    """
    Binary Weighted Dice Loss, where the predictions are thresholded before calculation
    """
    def __init__(
        self, 
        threshold=0.5, 
        beta=1, 
        smooth=1e-5
    ):
        super(BWDiceLoss, self).__init__()
        self.threshold = threshold
        self.beta_sq = beta ** 2
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        pred = torch.where(pred > self.threshold, 1, 0)
        
        numerator = (self.beta_sq + 1) * torch.sum(pred * target) + self.smooth
        denominator = self.beta_sq * torch.sum(target) + torch.sum(pred) + self.smooth
        dice = numerator / denominator

        return 1 - dice


def print_time_taken(start_time, end_time):
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"Training completed in {hours}h {minutes}m {seconds}s")