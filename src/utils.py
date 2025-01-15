import torch


def dice_loss(pred, true, threshold=0.5, smooth=1e-5):
    pred = torch.where(pred > threshold, 1, 0)

    return generalized_dice_loss(pred, true, smooth=smooth)


def generalized_dice_loss(pred, true, smooth=1e-5):
    pred = pred.view(-1)
    true = true.view(-1)
    
    intersection = torch.sum(pred * true)
    union = torch.sum(pred) + torch.sum(true)
    
    dice = (2 * intersection + smooth) / (union + smooth)
    
    return 1 - dice


def generalized_weighted_dice_loss(pred, true, beta=1, smooth=1e-5):
    pred = pred.view(-1)
    true = true.view(-1)
    
    tp = torch.sum(pred * true)
    fn = torch.sum((1 - pred) * true)
    fp = torch.sum(pred * (1 - true))

    dice = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)

    return 1 - dice


def print_time_taken(start_time, end_time):
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"Training completed in {hours}h {minutes}m {seconds}s")