import numpy as np
import pandas as pd
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.config_unet as config
from src.data.dataset import split_data
from src.utils.model_utils import train, save_outputs
from src.utils.general_utils import GWDiceLoss


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


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(config.MODEL_RS)
    torch.cuda.manual_seed_all(config.MODEL_RS)
    np.random.seed(config.MODEL_RS)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set up logging to track training progress over time
    logging.basicConfig(
        format='%(asctime)s  %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    # Prepare datasets
    df = pd.read_csv(config.DF_PATH)

    train_loader, val_loader, test_loader = split_data(
        df, 
        config.BATCH_SIZE, 
        config.MAX_BATCH_SIZE, 
        config.NUM_WORKERS
    )

    # Train model
    model = UNet(in_channels=config.IN_CHANNELS, out_channels=config.OUT_CHANNELS).to(config.DEVICE)

    loss_fn = GWDiceLoss(beta=config.BETA_WEIGHTING)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    lr_scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.LR_GAMMA)
    lr_scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.LR_FACTOR, patience=config.LR_PATIENCE)

    model, training_history = train(
        model, 
        optimizer,
        loss_fn,
        train_loader, 
        val_loader, 
        test_loader,
        lr_scheduler_exp,
        lr_scheduler_plateau,
        config
    )

    # Save model and results
    save_outputs(model, training_history, config)
