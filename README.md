# U-Net
U Net model for segmentation of lung lesions in CT scans of COVID patients.

Data from Kaggle: https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset/data

The MIT license is for the the code in this repository, not this data.

Papers read and used:

- https://arxiv.org/abs/1707.03237
- https://arxiv.org/abs/1505.04597

Ideas tried:

- Using Monai generalised dice loss
- lowering LR
- Resizing to 256x256, with larger batch size
- Adding early stopping
- Fiddling with batch size, resizing, and LR
- Remove "out of centre circle" masks, and scans too? Add as part of transform?
- Add a bit of data augmentation - random dimming
- Cutting off corners using a circular mask
- Flipping scans to all have the same orientation

Ideas to try:

- Use a different version of dice loss, to penalise false negatives more?
- New architecture?

TODOs 
- upscale the final mask later down the line
- Try more advanced architecture
