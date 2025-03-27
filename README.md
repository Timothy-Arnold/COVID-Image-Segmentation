# COVID-Image-Segmentation
Using various ML models for the segmentation of lung lesions in CT scans of COVID patients.

Data from Kaggle: https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset/data

The MIT license is for the code in this repository, not this data.

Papers read / used:

- https://arxiv.org/abs/1707.03237
- https://arxiv.org/abs/1505.04597
- https://arxiv.org/pdf/2010.11929
- https://arxiv.org/pdf/2101.11986

Ideas tried:

- Using Monai generalised dice loss
- lowering LR
- Resizing to 256x256, with larger batch size
- Adding early stopping
- Fiddling with batch size, resizing, and LR
- Add a bit of data augmentation - random dimming
- Cutting off corners using a circular mask
- Flipping scans to all have the same orientation
- Making custom weighted dice loss function which penalises false negatives more
- Implement learning rate decay
- Try different architectures e.g. Vision Transformers, Mask2Former

Ideas to try:

- Improve interpretability of model predictions, perhaps with SHAP or LIME, or attention patterns for ViT
- Upscale the final mask later down the line, for slightly more accurate dice loss results
