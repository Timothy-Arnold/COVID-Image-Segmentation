import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' if TkAgg doesn't work
import matplotlib.pyplot as plt

from PIL import Image
import os
import sys

import config
from model import UNet

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from monai.losses import DiceLoss

# def visualize_predictions(model_path, df_sample):
# Set up the figure
n_samples = 3

model_path = config.MODEL_SAVE_PATH
df = pd.read_csv(config.DF_PATH)
df_sample = df.sample(n=n_samples, random_state=2).reset_index(drop=True)

fig, axes = plt.subplots(len(df_sample), 4, figsize=(20, 5*len(df_sample)))
fig.suptitle('Image, Original Mask and Predicted Mask Comparison', fontsize=16)

model = torch.load(model_path)
model.eval()
model = model.to(config.DEVICE)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
])

loss = DiceLoss()

with torch.no_grad():
    for idx, row in enumerate(df_sample.itertuples()):
        # Load and transform image
        image_path = os.path.join(config.ROOT_DIR, row.scan)
        image = Image.open(image_path).convert("L")
        image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
        
        # Load and transform mask 
        mask_path = os.path.join(config.ROOT_DIR, row.mask)
        mask = Image.open(mask_path).convert("L")
        mask_tensor = transform(mask).squeeze(0)
        
        # Get prediction
        pred = model(image_tensor)
        pred = pred.cpu().squeeze(0).squeeze(0)
        
        # Calculate DICE loss
        dice_loss = loss(pred, mask_tensor)
        
        # Plot original image
        axes[idx, 0].imshow(image, cmap='gray')
        axes[idx, 0].set_title('Original Image')
        axes[idx, 0].axis('off')
        
        # Plot original mask
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title('Original Mask')
        axes[idx, 1].axis('off')
        
        # Plot predicted mask with Dice loss in title
        axes[idx, 2].imshow(pred, cmap='gray')
        axes[idx, 2].set_title(f'Predicted Mask (Dice loss: {dice_loss:.4f})')
        axes[idx, 2].axis('off')
        
        # Create combined visualization
        # Convert image to RGB for colored overlay
        image_array = image_tensor.squeeze(0).squeeze(0).cpu().numpy()
        
        image_rgb = np.stack([image_array] * 3, axis=-1)
        
        # Create red mask overlay
        mask_overlay = np.zeros_like(image_rgb)
        mask_overlay[:,:,0] = np.array(mask_tensor) # Red channel
        
        # Create blue prediction overlay
        pred_overlay = np.zeros_like(image_rgb)
        pred_overlay[:,:,2] = pred.numpy() # Blue channel
        
        # Combine image with overlays
        combined = image_rgb.astype(float)
        combined += mask_overlay.astype(float) * 0.3 # Red mask with 0.3 opacity
        combined += pred_overlay.astype(float) * 0.3 # Blue prediction with 0.3 opacity
        combined = np.clip(combined, 0, 1)
        
        axes[idx, 3].imshow(combined)
        axes[idx, 3].set_title('Combined View\n(Red: Mask, Blue: Prediction)')
        axes[idx, 3].axis('off')

plt.tight_layout()
plt.show(block=True)
