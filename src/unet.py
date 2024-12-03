import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image
from verstack.stratified_continuous_split import scsplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torchvision.transforms as transforms
from data_preprocess import ROOT_DIR, LungDataset # Import custom dataset class


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rs = 42
in_channels = 1
out_channels = 1
learning_rate = 1e-3
batch_size = 32
max_epochs = 100
early_stopping_steps = 10


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
    

# def train(model, train_loader, max_epochs, early_stopping_steps):
#     criterion = nn.NLLLoss()
#     optimizer = optim.Adam(model.parameters())

#     validation_decrease_counter = 0
#     highest_validation_accuracy = 0
#     stopped_early = False

#     for epoch in range(max_epochs):
#         for batch in train_loader:
#             images, labels = batch
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             predictions = model(images)
#             loss = criterion(predictions, labels)
#             loss.backward()
#             optimizer.step()

#         train_accuracy = test_accuracy(model, train_loader)
#         validation_accuracy = test_accuracy(model, validation_loader)
#         print(f"Epoch {epoch + 1} - Train accuracy: {format(train_accuracy, '.2f')}% - Validation accuracy: {format(validation_accuracy, '.2f')}%")

#         if validation_accuracy > highest_validation_accuracy:
#             highest_validation_accuracy = validation_accuracy
#             validation_decrease_counter = 0
#         else:
#             validation_decrease_counter += 1

#         if validation_decrease_counter == early_stopping_steps:
#             print("Early stopping criteria met")
#             stopped_early = True
#             break

#     if not stopped_early:
#         print("Maximum number of epochs reached")


if __name__ == "__main__":
    # Prepare datasets
    df = pd.read_csv("data/df_full.csv")

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = LungDataset(df, ROOT_DIR, transform=transform)

    # Split into train/val/test sets based on 6:1:1 ratio, stratified by mask coverage percentage.
    df_indexed = df.reset_index()

    train_indices, test_indices = scsplit(
        df_indexed,
        stratify=df_indexed["mask_coverage"],
        test_size=0.25,
        train_size=0.75,
        random_state=rs,
    )
    val_indices, test_indices = scsplit(
        test_indices,
        stratify=test_indices["mask_coverage"],
        test_size=0.5,
        train_size=0.5,
        random_state=rs,
    )
    train_dataset = Subset(dataset, train_indices["index"].values)
    val_dataset = Subset(dataset, val_indices["index"].values)
    test_dataset = Subset(dataset, test_indices["index"].values)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Train model

    model = UNet(in_channels=in_channels, out_channels=out_channels)