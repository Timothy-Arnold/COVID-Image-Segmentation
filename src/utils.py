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
        if len(pred.shape) == 4:
            pred = pred.squeeze(1)
            target = target.squeeze(1)
        
        batch_size = pred.size(0)
        dice_scores = []
        
        for i in range(batch_size):
            pred_sample = pred[i].view(-1)
            target_sample = target[i].view(-1)
            
            numerator = (self.beta_sq + 1) * torch.sum(pred_sample * target_sample) + self.smooth
            denominator = self.beta_sq * torch.sum(target_sample) + torch.sum(pred_sample) + self.smooth
            dice_score = numerator / denominator
            dice_scores.append(dice_score)
        
        mean_dice = torch.mean(torch.stack(dice_scores))
        return 1 - mean_dice


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
        pred = torch.where(pred > self.threshold, 1, 0)
        if len(pred.shape) == 4:
            pred = pred.squeeze(1)
            target = target.squeeze(1)

        batch_size = pred.size(0)
        dice_scores = []
        
        for i in range(batch_size):
            pred_sample = pred[i].view(-1)
            target_sample = target[i].view(-1)
            
            numerator = (self.beta_sq + 1) * torch.sum(pred_sample * target_sample) + self.smooth
            denominator = self.beta_sq * torch.sum(target_sample) + torch.sum(pred_sample) + self.smooth
            dice_score = numerator / denominator
            dice_scores.append(dice_score)
        
        mean_dice = torch.mean(torch.stack(dice_scores))
        return 1 - mean_dice


def print_time_taken(start_time, end_time):
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"Training completed in {hours}h {minutes}m {seconds}s")