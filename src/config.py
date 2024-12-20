import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RS = 1
IN_CHANNELS = 1
OUT_CHANNELS = 1
LR = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOPPING_STEPS = 10
EARLY_STOPPING_MIN_DELTA = 0
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

TRAIN_SIZE = 0.75
VAL_SIZE = 0.125
TEST_SIZE = 0.125

DF_PATH = 'data/df_full.csv'
LOSS_PLOT_SAVE_PATH = 'output/loss_plot.png'
MODEL_SAVE_PATH = 'output/unet_trained.pth'