# TODO upscale the final mask later down the line...
# TODO Add early stopping
# TODO Upgrade to more advanced architecture

import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' if TkAgg doesn't work
import matplotlib.pyplot as plt

from PIL import Image
from verstack.stratified_continuous_split import scsplit
import config

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision.transforms as transforms
from dataset import ROOT_DIR, LungDataset # Import custom dataset class
from monai.losses import DiceLoss


class UNet(nn.Module):
    """
    U-Net model for image segmentation.
    """
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
        x = torch.cat([x, x3], dim=1)
        x = self.dec1(x)

        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.upconv3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec3(x)

        x = self.dec4(x)
        x = nn.Sigmoid()(x)
        return x
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


def split_data(df, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    ])
    dataset = LungDataset(df, ROOT_DIR, transform=transform)

    df_indexed = df.reset_index()

    val_ratio = config.VAL_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    test_ratio = config.TEST_SIZE / (config.VAL_SIZE + config.TEST_SIZE)

    train_indices, test_indices = scsplit(
        df_indexed,
        stratify=df_indexed["mask_coverage"],
        test_size=1-config.TRAIN_SIZE,
        train_size=config.TRAIN_SIZE,
        random_state=config.RS,
    )
    val_indices, test_indices = scsplit(
        test_indices,
        stratify=test_indices["mask_coverage"],
        test_size=val_ratio,
        train_size=test_ratio,
        random_state=config.RS,
    )
    train_dataset = Subset(dataset, train_indices["index"].values)
    val_dataset = Subset(dataset, val_indices["index"].values)
    test_dataset = Subset(dataset, test_indices["index"].values)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader


def train(
        model, 
        train_loader, 
        val_loader, 
        test_loader,
        max_epochs, 
        early_stopping_steps
    ):

    training_history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
    }
    start_time = time()

    print(f"Training! Max Epochs: {config.MAX_EPOCHS}, Early Stopping Steps: {config.EARLY_STOPPING_STEPS}")
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train_loss = 0
        total_test_loss = 0

        for batch in train_loader:
            images, masks = batch
            images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)

            predictions = model(images)
            loss = loss_fn(predictions, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss

        with torch.no_grad():
            model.eval()
            for batch in test_loader:
                images, masks = batch
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)

                predictions = model(images)
                loss = loss_fn(predictions, masks)
                total_test_loss += loss

        average_train_loss = (total_train_loss / len(train_loader))
        average_test_loss = (total_test_loss / len(test_loader))

        print(f"Epoch {epoch} - Average train Dice loss: {average_train_loss:.4f} - Average test Dice loss: {average_test_loss:.4f}")

        training_history["train_loss"].append(average_train_loss.item())
        training_history["test_loss"].append(average_test_loss.item())

    end_time = time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    return model, training_history


def plot_training_history(training_history):
    plt.grid(True)
    plt.plot(training_history["train_loss"], label="Train Dice loss")
    plt.plot(training_history["test_loss"], label="Test Dice loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Dice loss")
    plt.legend(loc="upper right")
    plt.savefig(config.LOSS_PLOT_SAVE_PATH)


if __name__ == "__main__":
    # Prepare datasets
    df = pd.read_csv(config.DF_PATH)
    num_workers = 1 #os.cpu_count()
    # Split into tr1ain/val/test sets based on 6:1:1 ratio, stratified by mask coverage percentage.
    train_loader, val_loader, test_loader = split_data(df, config.BATCH_SIZE, num_workers)

    # Train model
    model = UNet(in_channels=config.IN_CHANNELS, out_channels=config.OUT_CHANNELS).to(config.DEVICE)

    loss_fn = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    model, training_history = train(
        model, 
        train_loader, 
        val_loader, 
        test_loader,
        config.MAX_EPOCHS, 
        config.EARLY_STOPPING_STEPS
    )

    # Save outputs
    plot_training_history(training_history)

    torch.save(model, config.MODEL_SAVE_PATH)
