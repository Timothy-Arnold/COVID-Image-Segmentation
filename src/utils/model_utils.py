import os
import json
from time import time
import logging
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' if TkAgg doesn't work
import matplotlib.pyplot as plt
import torch

from src.utils.general_utils import print_time_taken


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
        test_loader,
        lr_scheduler_exp,
        lr_scheduler_plateau,
        config
    ):
    # Initialize training history with 1 for each loss metric, for the zeroth epoch
    training_history = {
        "train_loss": [1],
        "val_loss": [1],
        "test_loss": [1],
    }
    early_stopper = EarlyStopper(patience=config.EARLY_STOPPING_STEPS, min_delta=0)

    stopped_early = False

    start_time = time()
    print(f"Training '{config.MODEL_NAME}'! Max Epochs: {config.MAX_EPOCHS}, Early Stopping Steps: {config.EARLY_STOPPING_STEPS}")
    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        total_train_loss = 0
        total_val_loss = 0
        total_test_loss = 0

        for batch in train_loader:
            images, masks = batch
            images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)

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
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)

                predictions = model(images)
                loss = loss_fn(predictions, masks)
                total_val_loss += loss.item()

            for batch in test_loader:
                images, masks = batch
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)

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
        if epoch <= np.log(0.5) / np.log(config.LR_GAMMA): 
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
            print(f"Early stopping triggered after {epoch} epochs - No improvement for {config.EARLY_STOPPING_STEPS} epochs")
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


def save_outputs(model, training_history, config) -> None:
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