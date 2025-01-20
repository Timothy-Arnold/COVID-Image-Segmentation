import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from verstack.stratified_continuous_split import scsplit

import config

# Iterate over all files in the directories
scans = [f"data/scans/{scan}" for scan in os.listdir(config.ROOT_DIR + "data/scans/")]
masks = [f"data/masks/{mask}" for mask in os.listdir(config.ROOT_DIR + "data/masks/")]

# Create circle mask
dimension_size = 512
circumference = dimension_size - 1
center = circumference / 2
radius = center + 0.5 + 5  # Adjust radius as needed 

y, x = np.ogrid[:dimension_size, :dimension_size]

dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)

circle_mask = dist_from_center < radius


class LungDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.circle_mask = circle_mask

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = Image.open(img_path).convert("L")
        mask_path = os.path.join(self.root_dir, self.df.iloc[idx, 1])
        mask = Image.open(mask_path).convert("L")

        # Apply circle mask to masks, ONLY in order to correct a faulty mask that exists in the dataset
        # image = np.array(image)
        # image = circle_mask * image
        # image = Image.fromarray(image)
        mask = np.array(mask)
        mask = circle_mask * mask
        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        if os.path.basename(img_path).startswith("Jun_radiopaedia"):
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])

        return image, mask
    

def split_data(df, batch_size, max_batch_size, num_workers):
    train_transform = transforms.Compose([
        # transforms.RandomRotation(15),
        # transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(), 
        transforms.Resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    ])

    # Validate that dataset split ratios sum to 1
    total_split = config.TRAIN_SIZE + config.VAL_SIZE + config.TEST_SIZE
    if not np.isclose(total_split, 1.0, rtol=1e-5):
        raise ValueError(f"Dataset split ratios must sum to 1.0, but got {total_split} "
                        f"(train={config.TRAIN_SIZE}, val={config.VAL_SIZE}, test={config.TEST_SIZE})")

    val_ratio = config.VAL_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    test_ratio = config.TEST_SIZE / (config.VAL_SIZE + config.TEST_SIZE)

    # df_train, df_test = scsplit(
    #     df,
    #     stratify=df["mask_coverage"],
    #     test_size=1-config.TRAIN_SIZE,
    #     train_size=config.TRAIN_SIZE,
    #     random_state=config.DATA_SPLIT_RS,
    # )
    # df_val, df_test = scsplit(
    #     df_test,
    #     stratify=df_test["mask_coverage"],
    #     test_size=val_ratio,
    #     train_size=test_ratio,
    #     random_state=config.DATA_SPLIT_RS,
    # )

    df_train, df_test = train_test_split(
        df, 
        test_size=1-config.TRAIN_SIZE, 
        shuffle=True,
        random_state=config.DATA_SPLIT_RS
    )

    df_val, df_test = train_test_split(
        df_test, 
        test_size=test_ratio, 
        shuffle=True,
        random_state=config.DATA_SPLIT_RS
    )

    # Save test df for predictions later
    df_test.to_csv("data/df_test.csv", index=False)

    train_dataset = LungDataset(df_train, config.ROOT_DIR, transform=train_transform)
    val_dataset = LungDataset(df_val, config.ROOT_DIR, transform=test_transform)
    test_dataset = LungDataset(df_test, config.ROOT_DIR, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=max_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=max_batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    mask_coverages = []

    for mask in tqdm(masks):
        mask_img = np.array(Image.open(mask).convert("L"))
        # Find overall percentage of mask coverage
        mask_coverage = np.mean(mask_img) / 2.55
        mask_coverages.append(mask_coverage)

    df = pd.DataFrame({"scan": scans, "mask": masks, "mask_coverage": mask_coverages})

    df.to_csv("data/df_full.csv", index=False)
