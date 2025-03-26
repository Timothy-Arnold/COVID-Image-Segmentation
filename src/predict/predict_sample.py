import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' if TkAgg doesn't work
import matplotlib.pyplot as plt
from PIL import Image
import os

import torch
import torchvision.transforms as transforms
from monai.losses import DiceLoss

import src.config_unet as config_unet
from src.model.model_unet import UNet
from src.utils.utils import GWDiceLoss, BWDiceLoss


n_samples = 4
beta = 3
threshold = 0.5
random_state = 0

model_path = config_unet.MODEL_SAVE_PATH
df = pd.read_csv(config_unet.DF_TEST_PATH)
df_sample = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)

fig, axes = plt.subplots(n_samples, 4, figsize=(20, 4*n_samples))
# fig.suptitle('Image, Original Mask and Predicted Mask Comparison', fontsize=16)

model = torch.load(model_path)
model.eval()
model = model.to(config_unet.DEVICE)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((config_unet.IMAGE_WIDTH, config_unet.IMAGE_HEIGHT))
])

binary_dice_loss = BWDiceLoss(threshold=threshold)
generalised_dice_loss = DiceLoss()
generalised_weighted_dice_loss = GWDiceLoss(beta=beta)


if __name__ == "__main__":
    with torch.no_grad():
        for idx, row in enumerate(df_sample.itertuples()):
            # Load and transform image
            image_path = os.path.join(config_unet.ROOT_DIR, row.scan)
            image = Image.open(image_path).convert("L")
            
            # Load and transform mask 
            mask_path = os.path.join(config_unet.ROOT_DIR, row.mask)
            mask = Image.open(mask_path).convert("L")

            if os.path.basename(image_path).startswith("Jun_radiopaedia"):
                image = np.array(image)
                image = np.flipud(image)
                image = Image.fromarray(image)
                mask = np.array(mask)
                mask = np.flipud(mask)
                mask = Image.fromarray(mask)

            image_tensor = transform(image).unsqueeze(0).to(config_unet.DEVICE)
            mask_tensor = transform(mask).squeeze(0)
            
            # Get prediction
            pred = model(image_tensor)
            pred = pred.cpu().squeeze(0).squeeze(0)
            
            # Calculate G DICE loss
            bdl = binary_dice_loss(pred, mask_tensor)
            gdl = generalised_dice_loss(pred, mask_tensor)
            wgdl = generalised_weighted_dice_loss(pred, mask_tensor)

            # Plot original image
            axes[idx, 0].imshow(image, cmap='gray')
            axes[idx, 0].set_title('CT Scan')
            axes[idx, 0].axis('off')
            
            # Plot original mask
            axes[idx, 1].imshow(mask, cmap='gray')
            axes[idx, 1].set_title('True Mask')
            axes[idx, 1].axis('off')
            
            # Plot predicted mask with Dice loss in title
            axes[idx, 2].imshow(pred, cmap='gray')
            axes[idx, 2].set_title(f'Predicted Mask\n \
Bdl: {bdl:.4f} - Gdl: {gdl:.4f} - WGdl: {wgdl:.4f}')
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
            axes[idx, 3].set_title('Combined - Red: Truth, Blue: Prediction, Magenta: Overlap')
            axes[idx, 3].axis('off')

    plt.tight_layout()
    plt.show(block=True)
