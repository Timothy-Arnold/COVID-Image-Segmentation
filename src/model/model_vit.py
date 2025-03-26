import numpy as np
import pandas as pd
from time import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.config_vit as config_vit
from src.data.dataset import split_data
from src.utils.utils import GWDiceLoss, print_time_taken
from src.model.model_utils import EarlyStopper, save_outputs


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

        heads = [SelfAttentionEncoderBlock(embed_size, num_heads, dropout) for head in range(num_blocks)]
        self.layers = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            ViTInput(image_size, patch_size, in_channels, embed_size),
            nn.Sequential(*heads),
            OutputProjection(image_size, patch_size, embed_size, out_channels),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


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

        if (epoch == 5) & (average_val_loss > 0.95):
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
    save_outputs(model, training_history, config_vit)
