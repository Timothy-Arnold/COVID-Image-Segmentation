import numpy as np
import pandas as pd
from time import time
import logging
import json
import os
import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' if TkAgg doesn't work
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.config_vit as config_vit
from src.data.dataset import split_data
from src.utils.utils import GWDiceLoss, print_time_taken


class ImageToPatches(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        assert len(x.size()) == 4
        y = self.unfold(x)
        y = y.permute(0, 2, 1)
        return y


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_size):
        super().__init__()
        self.in_channels = in_channels
        self.embed_size = embed_size
        self.embed_layer = nn.Linear(in_features=in_channels, out_features=embed_size)
    
    def forward(self, x):
        assert len(x.size()) == 3
        # B, T, C = x.size()
        x = self.embed_layer(x)
        return x


class ViTInput(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_size):
        super().__init__()
        self.i2p = ImageToPatches(image_size, patch_size)
        self.pe = PatchEmbedding(patch_size * patch_size * in_channels, embed_size)
        num_patches = (image_size // patch_size) ** 2
        self.position_embed = nn.Parameter(torch.randn(num_patches, embed_size))

    def forward(self, x):
        x = self.i2p(x)
        x = self.pe(x)
        x = x + self.position_embed
        return x
    

class MLP(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(p=dropout),
        )
    
    def forward(self, x):
        return self.layers(x)
    

class SelfAttentionEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.ln1 = nn.LayerNorm(embed_size)
        self.mha = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, dropout)
    
    def forward(self, x):
        y = self.ln1(x)
        x = x + self.mha(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x
    

class OutputProjection(nn.Module):
    def __init__(self, image_size, patch_size, embed_size, output_dims):
        super().__init__()
        self.patch_size = patch_size
        self.output_dims = output_dims
        self.projection = nn.Linear(embed_size, patch_size * patch_size * output_dims)
        self.fold = nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, stride=patch_size)
    # end def
    
    def forward(self, x):
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        x = self.fold(x)
        return x


class ViT(nn.Module):
    def __init__(
            self,
            image_size,
            patch_size,
            in_channels,
            out_channels,
            embed_size,
            num_blocks,
            num_heads,
            dropout
        ):
        super(ViT, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout

        heads = [SelfAttentionEncoderBlock(embed_size, num_heads, dropout) for i in range(num_blocks)]
        self.layers = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            ViTInput(image_size, patch_size, in_channels, embed_size),
            nn.Sequential(*heads),
            OutputProjection(image_size, patch_size, embed_size, out_channels),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


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
    early_stopper = EarlyStopper(patience=config_vit.EARLY_STOPPING_STEPS, min_delta=0)

    stopped_early = False

    start_time = time()
    print(f"Training '{config_vit.MODEL_NAME}'! Max Epochs: {config_vit.MAX_EPOCHS}, Early Stopping Steps: {config_vit.EARLY_STOPPING_STEPS}")
    for epoch in range(1, config_vit.MAX_EPOCHS + 1):
        model.train()
        total_train_loss = 0
        total_val_loss = 0
        total_test_loss = 0

        for batch in train_loader:
            images, masks = batch
            images, masks = images.to(config_vit.DEVICE), masks.to(config_vit.DEVICE)

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
                images, masks = images.to(config_vit.DEVICE), masks.to(config_vit.DEVICE)

                predictions = model(images)
                loss = loss_fn(predictions, masks)
                total_val_loss += loss.item()

            for batch in test_loader:
                images, masks = batch
                images, masks = images.to(config_vit.DEVICE), masks.to(config_vit.DEVICE)

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

        # Stop decaying after LR is halved:
        if epoch <= np.log(0.5) / np.log(config_vit.LR_GAMMA): 
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
            print(f"Early stopping triggered after {epoch} epochs - No improvement for {config_vit.EARLY_STOPPING_STEPS} epochs")
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
    if not os.path.exists(config_vit.FOLDER_PATH):
        os.makedirs(config_vit.FOLDER_PATH)

    # Plot loss history
    plt.figure(figsize=(12, 7))
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth=0.25)
    plt.plot(training_history["train_loss"], label="Train Dice loss")
    plt.plot(training_history["val_loss"], label="Val Dice loss")
    plt.plot(training_history["test_loss"], label="Test Dice loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Dice loss")
    plt.legend(loc="upper right")
    plt.title(f"Model: {config_vit.MODEL_NAME}")
    plt.savefig(config_vit.LOSS_PLOT_SAVE_PATH)

    hyperparameters = {
        'device': str(config_vit.DEVICE),
        'input_channels': config_vit.IN_CHANNELS, 
        'output_channels': config_vit.OUT_CHANNELS,
        'image_width': config_vit.IMAGE_WIDTH,
        'image_height': config_vit.IMAGE_HEIGHT,
        'data_split_random_seed': config_vit.DATA_SPLIT_RS,
        'model_random_seed': config_vit.MODEL_RS,
        'learning_rate': config_vit.LR,
        'learning_rate_gamma': config_vit.LR_GAMMA,
        'learning_rate_patience': config_vit.LR_PATIENCE,
        'learning_rate_factor': config_vit.LR_FACTOR,
        'max_epochs': config_vit.MAX_EPOCHS,
        'early_stopping_steps': config_vit.EARLY_STOPPING_STEPS,
        'early_stopping_min_delta': config_vit.EARLY_STOPPING_MIN_DELTA,
        'batch_size': config_vit.BATCH_SIZE,
        'max_batch_size': config_vit.MAX_BATCH_SIZE,
        'num_workers': config_vit.NUM_WORKERS,
        'train_size': config_vit.TRAIN_SIZE,
        'val_size': config_vit.VAL_SIZE, 
        'test_size': config_vit.TEST_SIZE,
        'beta_weighting': config_vit.BETA_WEIGHTING,
        'patch_size': config_vit.PATCH_SIZE,
        'embed_size': config_vit.EMBED_SIZE,
        'num_blocks': config_vit.NUM_BLOCKS,
        'num_heads': config_vit.NUM_HEADS,
        'dropout': config_vit.DROPOUT,
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

    with open(config_vit.HYPER_PARAM_SAVE_PATH, 'w') as f:
        json.dump(overall_result, f, indent=4)

    torch.save(model, config_vit.MODEL_SAVE_PATH)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(config_vit.MODEL_RS)
    torch.cuda.manual_seed_all(config_vit.MODEL_RS)
    np.random.seed(config_vit.MODEL_RS)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set up logging to track training progress over time
    logging.basicConfig(
        format='%(asctime)s  %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    # Prepare datasets
    df = pd.read_csv(config_vit.DF_PATH)

    train_loader, val_loader, test_loader = split_data(
        df, 
        config_vit.BATCH_SIZE, 
        config_vit.MAX_BATCH_SIZE, 
        config_vit.NUM_WORKERS
    )

    # Train model
    model = ViT(
        image_size=config_vit.IMAGE_HEIGHT,
        patch_size=config_vit.PATCH_SIZE,
        in_channels=config_vit.IN_CHANNELS, 
        out_channels=config_vit.OUT_CHANNELS,
        embed_size=config_vit.EMBED_SIZE,
        num_blocks=config_vit.NUM_BLOCKS,
        num_heads=config_vit.NUM_HEADS,
        dropout=config_vit.DROPOUT
    ).to(config_vit.DEVICE)

    loss_fn = GWDiceLoss(beta=config_vit.BETA_WEIGHTING)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_vit.LR)

    lr_scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_vit.LR_GAMMA)
    lr_scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config_vit.LR_FACTOR, patience=config_vit.LR_PATIENCE)

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
