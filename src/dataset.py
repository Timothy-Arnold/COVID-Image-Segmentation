import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from PIL import Image
from torch.utils.data import Dataset
import config


# Iterate over all files in the directories
scans = [f"data/scans/{scan}" for scan in os.listdir(config.ROOT_DIR + "data/scans/")]
masks = [f"data/masks/{mask}" for mask in os.listdir(config.ROOT_DIR + "data/masks/")]

class LungDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.annotations = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("L")
        mask_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 1])
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


if __name__ == "__main__":
    mask_coverages = []

    for mask in tqdm(masks):
        mask_img = np.array(Image.open(mask).convert("L"))
        # Find overall percentage of mask coverage
        mask_coverage = np.mean(mask_img) / 2.55
        mask_coverages.append(mask_coverage)

    df = pd.DataFrame({"scan": scans, "mask": masks, "mask_coverage": mask_coverages})

    df.to_csv("data/df_full.csv", index=False)
