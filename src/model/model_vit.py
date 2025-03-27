import numpy as np
import pandas as pd
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.config_vit as config
from src.data.dataset import split_data
from src.utils.model_utils import train, save_outputs
from src.utils.general_utils import GWDiceLoss


def get_sinusoid_encoding(num_tokens, token_len):
    """ Make Sinusoid Encoding Table

        Args:
            num_tokens (int): number of tokens
            token_len (int): length of a token
            
        Returns:
            (torch.FloatTensor) sinusoidal position encoding table
    """

    def get_position_angle_vec(i):
        return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

    sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


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
        
        # Create positional embeddings using sine and cosine functions
        position = torch.arange(num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-torch.log(torch.tensor(10000.0)) / embed_size))
        pos_embed = torch.zeros(num_patches, embed_size)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.position_embed = nn.Parameter(pos_embed, requires_grad=False)

    def forward(self, x):
        x = self.i2p(x)
        x = self.pe(x)
        x += self.position_embed
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
    model = ViT(
        image_size=config.IMAGE_HEIGHT,
        patch_size=config.PATCH_SIZE,
        in_channels=config.IN_CHANNELS, 
        out_channels=config.OUT_CHANNELS,
        embed_size=config.EMBED_SIZE,
        num_blocks=config.NUM_BLOCKS,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

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
