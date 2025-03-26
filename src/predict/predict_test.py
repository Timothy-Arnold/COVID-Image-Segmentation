import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' if TkAgg doesn't work
import matplotlib.pyplot as plt
from PIL import Image
import time

import torch
import torchvision.transforms as transforms

import src.config_unet as config_unet
from src.data.dataset import split_data
from src.utils.utils import GWDiceLoss, BWDiceLoss
from src.model.model_unet import UNet


beta_weighting = 3
threshold = 0.5

model_path = config_unet.MODEL_SAVE_PATH
df = pd.read_csv(config_unet.DF_PATH)

model = torch.load(model_path)
model.eval()
model = model.to(config_unet.DEVICE)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((config_unet.IMAGE_WIDTH, config_unet.IMAGE_HEIGHT))
])

binary_dice_loss = BWDiceLoss(threshold=threshold, beta=1)
generalised_dice_loss = GWDiceLoss(beta=1)
generalised_weighted_dice_loss = GWDiceLoss(beta=beta_weighting)


if __name__ == "__main__":
    start_time = time.time()
    bdl_total = 0
    gdl_total = 0
    gwdl_total = 0

    _, _, test_loader = split_data(df, config_unet.BATCH_SIZE, config_unet.MAX_BATCH_SIZE, config_unet.NUM_WORKERS)

    all_images = []
    all_masks = []
    for batch in test_loader:
        images, masks = batch
        all_images.append(images)
        all_masks.append(masks)

    all_images = torch.cat(all_images, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Create batches of size 1, for the sake of fully accurate test score
    # No longer affected by batches of unequal size having equal weighting for final score
    num_samples = len(all_images)
    for i in range(num_samples):
        image = all_images[i:i+1].to(config_unet.DEVICE)
        mask = all_masks[i:i+1].to(config_unet.DEVICE)

        pred = model(image)

        bdl = binary_dice_loss(pred, mask)
        gdl = generalised_dice_loss(pred, mask)
        gwdl = generalised_weighted_dice_loss(pred, mask)
        bdl_total += bdl.item()
        gdl_total += gdl.item()
        gwdl_total += gwdl.item()

    mean_bdl = np.round(bdl_total / num_samples, 5)
    mean_gdl = np.round(gdl_total / num_samples, 5)
    mean_gwdl = np.round(gwdl_total / num_samples, 5)

    jackard = (1 - mean_bdl) / (1 + mean_bdl) # Calculated from dice loss

    print(f"Binary Dice Loss: {mean_bdl}")
    print(f"Generalised Dice Loss: {mean_gdl}")
    print(f"Generalised Weighted Dice Loss: {mean_gwdl}")
    print(f"Jaccard Index / IoU: {jackard}")

    end_time = time.time()
    print(f"Time taken: {np.round(end_time - start_time, 2)} seconds")
