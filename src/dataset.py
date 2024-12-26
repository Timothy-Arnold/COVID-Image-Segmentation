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

        # Apply circle mask to mask
        image = np.array(image)
        mask = np.array(mask)
        image = circle_mask * image
        mask = circle_mask * mask
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # if os.path.basename(img_path).startswith("Jun_radiopaedia"):
        #     image = torch.flip(image, dims=[0])
        #     mask = torch.flip(mask, dims=[0])

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
