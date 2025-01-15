import numpy as np
import pandas as pd
from time import time
import json
import os
import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' if TkAgg doesn't work
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss

import config
from dataset import split_data
from utils import print_time_taken, generalized_weighted_dice_loss


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
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_model_state = None

    def early_stop(self, validation_loss, model):
        early_stop = False
        best_model = False

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            best_model = True
            self.best_model_state = model.state_dict()
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True

        return early_stop, best_model


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
    print(f"Training '{config.MODEL_NAME}'! Max Epochs: {config.MAX_EPOCHS}, Early Stopping Steps: {config.EARLY_STOPPING_STEPS}")
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

        early_stop, best_model = early_stopper.early_stop(average_val_loss, model)

        colour_prefix = "\033[35m" if best_model else ""
        colour_suffix = "\033[0m" if best_model else ""

        print(
f"{colour_prefix}Epoch {epoch} - Average train Dice loss: {average_train_loss:.4f} \
- Average val Dice loss: {average_val_loss:.4f} \
- Average test Dice loss: {average_test_loss:.4f}{colour_suffix}")

        training_history["train_loss"].append(average_train_loss.item())
        training_history["val_loss"].append(average_val_loss.item())
        training_history["test_loss"].append(average_test_loss.item())

        if early_stop:
            stopped_early = True
            print(f"Early stopping triggered after {epoch} epochs - No improvement for {config.EARLY_STOPPING_STEPS} epochs")
            break

    if not stopped_early:
        print(f"Training completed after max epochs: {epoch}")

    training_history["epochs"] = epoch

    end_time = time()
    print_time_taken(start_time, end_time)

    # Before returning, load the best model state
    if early_stopper.best_model_state is not None:
        model.load_state_dict(early_stopper.best_model_state)
        print(f"Saving best model with validation loss: {early_stopper.min_validation_loss:.4f}")

    return model, training_history


def save_outputs(model, training_history):
    # Create output directory if it doesn't exist
    if not os.path.exists(f'output/{config.MODEL_NAME}'):
        os.makedirs(f'output/{config.MODEL_NAME}')

    # Plot loss history
    plt.figure(figsize=(12, 7))
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
        'image_width': config.IMAGE_WIDTH,
        'image_height': config.IMAGE_HEIGHT,
        'train_size': config.TRAIN_SIZE,
        'val_size': config.VAL_SIZE, 
        'test_size': config.TEST_SIZE
    }

    # Find results of model which performed best on validation set
    best_val_loss = min(training_history["val_loss"])
    best_val_loss_index = training_history["val_loss"].index(best_val_loss)

    results = {
        'epochs': training_history["epochs"],
        "train_loss": training_history["train_loss"][best_val_loss_index],
        "val_loss": best_val_loss,
        "test_loss": training_history["test_loss"][best_val_loss_index],
    }

    overall_result = {
        "hyperparameters": hyperparameters,
        "results": results,
    }

    with open(config.HYPER_PARAM_SAVE_PATH, 'w') as f:
        json.dump(overall_result, f, indent=4)

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
    # Split into train/val/test sets, stratified by mask coverage percentage 
    # in order to ensure that the test set is representative of the population.
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
