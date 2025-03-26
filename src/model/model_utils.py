import os
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' if TkAgg doesn't work
import matplotlib.pyplot as plt
import torch


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


def save_outputs(model, training_history, config):
    # Create output directory if it doesn't exist
    if not os.path.exists(config.FOLDER_PATH):
        os.makedirs(config.FOLDER_PATH)

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
    plt.title(f"Model: {config.MODEL_NAME}")
    plt.savefig(config.LOSS_PLOT_SAVE_PATH)

    hyperparameters = {
        'device': str(config.DEVICE),
        'input_channels': config.IN_CHANNELS, 
        'output_channels': config.OUT_CHANNELS,
        'image_width': config.IMAGE_WIDTH,
        'image_height': config.IMAGE_HEIGHT,
        'data_split_random_seed': config.DATA_SPLIT_RS,
        'model_random_seed': config.MODEL_RS,
        'learning_rate': config.LR,
        'learning_rate_gamma': config.LR_GAMMA,
        'learning_rate_patience': config.LR_PATIENCE,
        'learning_rate_factor': config.LR_FACTOR,
        'max_epochs': config.MAX_EPOCHS,
        'early_stopping_steps': config.EARLY_STOPPING_STEPS,
        'early_stopping_min_delta': config.EARLY_STOPPING_MIN_DELTA,
        'batch_size': config.BATCH_SIZE,
        'max_batch_size': config.MAX_BATCH_SIZE,
        'num_workers': config.NUM_WORKERS,
        'train_size': config.TRAIN_SIZE,
        'val_size': config.VAL_SIZE, 
        'test_size': config.TEST_SIZE,
        'beta_weighting': config.BETA_WEIGHTING,
        'patch_size': config.PATCH_SIZE,
        'embed_size': config.EMBED_SIZE,
        'num_blocks': config.NUM_BLOCKS,
        'num_heads': config.NUM_HEADS,
        'dropout': config.DROPOUT,
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