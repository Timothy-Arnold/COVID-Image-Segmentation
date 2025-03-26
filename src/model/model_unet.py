import numpy as np
import pandas as pd
from time import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.config_unet as config_unet
from src.data.dataset import split_data
from src.utils.utils import GWDiceLoss, print_time_taken
from src.model.model_utils import EarlyStopper, save_outputs


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
    early_stopper = EarlyStopper(patience=config_unet.EARLY_STOPPING_STEPS, min_delta=0)

    stopped_early = False

    start_time = time()
    print(f"Training '{config_unet.MODEL_NAME}'! Max Epochs: {config_unet.MAX_EPOCHS}, Early Stopping Steps: {config_unet.EARLY_STOPPING_STEPS}")
    for epoch in range(1, config_unet.MAX_EPOCHS + 1):
        model.train()
        total_train_loss = 0
        total_val_loss = 0
        total_test_loss = 0

        for batch in train_loader:
            images, masks = batch
            images, masks = images.to(config_unet.DEVICE), masks.to(config_unet.DEVICE)

            predictions = model(images)
            loss = loss_fn(predictions, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        with torch.no_grad():
            model.eval()
            for batch in val_loader:
                images, masks = batch
                images, masks = images.to(config_unet.DEVICE), masks.to(config_unet.DEVICE)

                predictions = model(images)
                loss = loss_fn(predictions, masks)
                total_val_loss += loss.item()

            for batch in test_loader:
                images, masks = batch
                images, masks = images.to(config_unet.DEVICE), masks.to(config_unet.DEVICE)

                predictions = model(images)
                loss = loss_fn(predictions, masks)
                total_test_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)
        average_val_loss = total_val_loss / len(val_loader)
        average_test_loss = total_test_loss / len(test_loader)

        if (epoch == 3) & (average_val_loss > 0.9):
            print("Optimisation not working, stopping early")
            break

        current_lr = optimizer.param_groups[0]['lr']
        if epoch <= np.log(0.5) / np.log(config_unet.LR_GAMMA): 
            # Stop decaying after LR is halved
            lr_scheduler_exp.step()

            lr_scheduler_plateau.step(average_val_loss)

        early_stop, best_model = early_stopper.early_stop(average_val_loss, model)

        colour_prefix = "\033[35m" if best_model else ""
        colour_suffix = "\033[0m" if best_model else ""

        logging.info(
f"{colour_prefix}Epoch {epoch} - Train DL: {average_train_loss:.4f} \
- Val DL: {average_val_loss:.4f} \
- Test DL: {average_test_loss:.4f} \
- Current lr: {current_lr:.2e}{colour_suffix}"
        )

        training_history["train_loss"].append(average_train_loss)
        training_history["val_loss"].append(average_val_loss)
        training_history["test_loss"].append(average_test_loss)

        if early_stop:
            stopped_early = True
            print(f"Early stopping triggered after {epoch} epochs - No improvement for {config_unet.EARLY_STOPPING_STEPS} epochs")
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


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(config_unet.MODEL_RS)
    torch.cuda.manual_seed_all(config_unet.MODEL_RS)
    np.random.seed(config_unet.MODEL_RS)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set up logging to track training progress over time
    logging.basicConfig(
        format='%(asctime)s  %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    # Prepare datasets
    df = pd.read_csv(config_unet.DF_PATH)

    train_loader, val_loader, test_loader = split_data(
        df, 
        config_unet.BATCH_SIZE, 
        config_unet.MAX_BATCH_SIZE, 
        config_unet.NUM_WORKERS
    )

    # Train model
    model = UNet(in_channels=config_unet.IN_CHANNELS, out_channels=config_unet.OUT_CHANNELS).to(config_unet.DEVICE)

    loss_fn = GWDiceLoss(beta=config_unet.BETA_WEIGHTING)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_unet.LR)

    lr_scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_unet.LR_GAMMA)
    lr_scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config_unet.LR_FACTOR, patience=config_unet.LR_PATIENCE)

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
