import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from verstack.stratified_continuous_split import scsplit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms
import torchvision.transforms as transforms

# Directory containing the scans
scans_dir = "C:/Users/timcy/Documents/Code/Personal/U-Net/data/scans/"
masks_dir = "C:/Users/timcy/Documents/Code/Personal/U-Net/data/masks/"

# List to store the paths of all png files
scans = []
masks = []
shapes = []

# Iterate over all files in the directory
for filename in tqdm(os.listdir(scans_dir)):
    scan = np.array(Image.open(os.path.join(scans_dir, filename)).convert('L'))
    mask = np.array(Image.open(os.path.join(masks_dir, filename)).convert('L'))
    scans.append(scan)
    masks.append(mask)
    shape = scan.shape
    shapes.append(shape)

image = Image.open(os.path.join(scans_dir, filename)).convert('L')
mask = Image.open(os.path.join(masks_dir, filename)).convert('L')

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((128, 128)),
])
image_tensor = transform(image)
mask_tensor = transform(mask)
print(image_tensor.shape)
print(mask_tensor.shape)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dec1 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec3 = self.conv_block(128, 64)
        self.dec4 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        x = self.upconv1(x4)
        x = torch.cat([x, x3], dim=0)
        x = self.dec1(x)

        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=0)
        x = self.dec2(x)

        x = self.upconv3(x)
        x = torch.cat([x, x1], dim=0)
        x = self.dec3(x)

        x = self.dec4(x)
        return x
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1)
    model.eval()

    with torch.no_grad():
        prediction = model(image_tensor).squeeze(0)

    prediction = prediction.cpu().numpy()
    prediction_image = Image.fromarray(prediction, mode='L')
    prediction_image.show()