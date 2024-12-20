import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RS = 42
IN_CHANNELS = 1
OUT_CHANNELS = 1
LR = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 40
EARLY_STOPPING_STEPS = 10
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

TRAIN_SIZE = 0.75
VAL_SIZE = 0.125
TEST_SIZE = 0.125

DF_PATH = 'data/df_full.csv'
LOSS_PLOT_SAVE_PATH = 'output/loss_plot.png'
MODEL_SAVE_PATH = 'output/unet_trained.pth'