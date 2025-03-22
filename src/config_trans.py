import torch
import os

MODEL_NAME = "unet_standard_0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IN_CHANNELS = 1
OUT_CHANNELS = 1
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

DATA_SPLIT_RS = 1
MODEL_RS = 1

LR = 1e-4
LR_GAMMA = 0.995
LR_PATIENCE = 10
LR_FACTOR = 0.5
MAX_EPOCHS = 200
EARLY_STOPPING_STEPS = 20
EARLY_STOPPING_MIN_DELTA = 0.001

BATCH_SIZE = 16
MAX_BATCH_SIZE = 32
NUM_WORKERS = 1 #os.cpu_count()

TRAIN_SIZE = 0.75
VAL_SIZE = 0.125
TEST_SIZE = 0.125

BETA_WEIGHTING = 3 # Weighting for false negatives, as opposed to false positives

ROOT_DIR = "C:/Users/timcy/Documents/Code/Personal/U-Net/"
DF_PATH = 'data/df_full.csv'
DF_TEST_PATH = 'data/df_test.csv'
LOSS_PLOT_SAVE_PATH = f'output/{MODEL_NAME}/loss_plot.png'
HYPER_PARAM_SAVE_PATH = f'output/{MODEL_NAME}/hyper_params.json'
MODEL_SAVE_PATH = f'output/{MODEL_NAME}/trained.pth'
