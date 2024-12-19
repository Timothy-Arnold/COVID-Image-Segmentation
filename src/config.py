import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RS = 42
IN_CHANNELS = 1
OUT_CHANNELS = 1
LR = 1e-3
BATCH_SIZE = 16
MAX_EPOCHS = 4
EARLY_STOPPING_STEPS = 10
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256