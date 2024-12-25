# TODO upscale the final mask later down the line...
# TODO Add early stopping
# TODO Upgrade to more advanced architecture

import numpy as np
import pandas as pd
from time import time
import json
import os
import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' if TkAgg doesn't work
import matplotlib.pyplot as plt

from verstack.stratified_continuous_split import scsplit
import config

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision.transforms as transforms
from dataset import LungDataset # Import custom dataset class
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
        x = F.sigmoid(x)
        return x
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def split_data(df, batch_size, num_workers):
    train_transform = transforms.Compose([
        # transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2),  # Randomly adjusts brightness by ±20%
        transforms.ToTensor(), 
        transforms.Resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    ])

    val_ratio = config.VAL_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    test_ratio = config.TEST_SIZE / (config.VAL_SIZE + config.TEST_SIZE)

    df_train, df_test = scsplit(
        df,
        stratify=df["mask_coverage"],
        test_size=1-config.TRAIN_SIZE,
        train_size=config.TRAIN_SIZE,
        random_state=config.RS,
    )
    df_val, df_test = scsplit(
        df_test,
        stratify=df_test["mask_coverage"],
        test_size=val_ratio,
        train_size=test_ratio,
        random_state=config.RS,
    )

    train_dataset = LungDataset(df_train, config.ROOT_DIR, transform=train_transform)
    val_dataset = LungDataset(df_val, config.ROOT_DIR, transform=test_transform)
    test_dataset = LungDataset(df_test, config.ROOT_DIR, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader


def print_time_taken(start_time, end_time):
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"Training completed in {hours}h {minutes}m {seconds}s")


def train(
        model, 
        optimizer,
        loss_fn,
        train_loader, 
        val_loader, 
        test_loader
    ):

    # Initialize training history with 1 for each loss metric, for the zeroth epoch
    training_history = {
        "train_loss": [1],
        "val_loss": [1],
        "test_loss": [1],
    }
    early_stopper = EarlyStopper(patience=config.EARLY_STOPPING_STEPS, min_delta=0)

    stopped_early = False

    start_time = time()
    print(f"Training {config.MODEL_NAME}! Max Epochs: {config.MAX_EPOCHS}, Early Stopping Steps: {config.EARLY_STOPPING_STEPS}")
    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        total_train_loss = 0
        total_val_loss = 0
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
            for batch in val_loader:
                images, masks = batch
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)

                predictions = model(images)
                loss = loss_fn(predictions, masks)
                total_val_loss += loss

            for batch in test_loader:
                images, masks = batch
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)

                predictions = model(images)
                loss = loss_fn(predictions, masks)
                total_test_loss += loss

        average_train_loss = total_train_loss / len(train_loader)
        average_val_loss = total_val_loss / len(val_loader)
        average_test_loss = total_test_loss / len(test_loader)

        print(
f"Epoch {epoch} - Average train Dice loss: {average_train_loss:.4f} \
- Average val Dice loss: {average_val_loss:.4f} \
- Average test Dice loss: {average_test_loss:.4f}")

        training_history["train_loss"].append(average_train_loss.item())
        training_history["val_loss"].append(average_val_loss.item())
        training_history["test_loss"].append(average_test_loss.item())

        if early_stopper.early_stop(average_val_loss):
            stopped_early = True
            print(f"Early stopping triggered after {epoch} epochs - No improvement for {config.EARLY_STOPPING_STEPS} epochs")
            break

    if not stopped_early:
        print(f"Training completed after max epochs: {epoch}")

    training_history["epochs"] = epoch

    end_time = time()
    print_time_taken(start_time, end_time)

    return model, training_history


def save_outputs(model, training_history):
    # Create output directory if it doesn't exist
    if not os.path.exists(f'output/{config.MODEL_NAME}'):
        os.makedirs(f'output/{config.MODEL_NAME}')

    plt.grid(True)
    plt.plot(training_history["train_loss"], label="Train Dice loss")
    plt.plot(training_history["val_loss"], label="Val Dice loss")
    plt.plot(training_history["test_loss"], label="Test Dice loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Dice loss")
    plt.legend(loc="upper right")
    plt.savefig(config.LOSS_PLOT_SAVE_PATH)

    hyperparameters = {
        'device': str(config.DEVICE),
        'random_seed': config.RS,
        'input_channels': config.IN_CHANNELS, 
        'output_channels': config.OUT_CHANNELS,
        'learning_rate': config.LR,
        'batch_size': config.BATCH_SIZE,
        'max_epochs': config.MAX_EPOCHS,
        'early_stopping_steps': config.EARLY_STOPPING_STEPS,
        'early_stopping_min_delta': config.EARLY_STOPPING_MIN_DELTA,
        'epochs': training_history["epochs"],
        'image_width': config.IMAGE_WIDTH,
        'image_height': config.IMAGE_HEIGHT,
        'train_size': config.TRAIN_SIZE,
        'val_size': config.VAL_SIZE, 
        'test_size': config.TEST_SIZE
    }

    results = {
        "hyperparameters": hyperparameters,
        "train_loss": training_history["train_loss"][-1],
        "val_loss": training_history["val_loss"][-1],
        "test_loss": training_history["test_loss"][-1],
    }

    with open(config.HYPER_PARAM_SAVE_PATH, 'w') as f:
        json.dump(results, f, indent=4)

    torch.save(model, config.MODEL_SAVE_PATH)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(config.RS)
    torch.cuda.manual_seed_all(config.RS)
    np.random.seed(config.RS)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Prepare datasets
    df = pd.read_csv(config.DF_PATH)
    # Split into tr1ain/val/test sets based on 6:1:1 ratio, stratified by mask coverage percentage.
    num_workers = 1 #os.cpu_count()
    train_loader, val_loader, test_loader = split_data(df, config.BATCH_SIZE, num_workers)

    # Train model
    model = UNet(in_channels=config.IN_CHANNELS, out_channels=config.OUT_CHANNELS).to(config.DEVICE)

    loss_fn = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    model, training_history = train(
        model, 
        optimizer,
        loss_fn,
        train_loader, 
        val_loader, 
        test_loader
    )

    # Save model and results
    save_outputs(model, training_history)
