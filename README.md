# Brain Tumor MRI Classification

Classifying brain MRI scans into glioma, meningioma, pituitary tumor or no tumor with a fine-tuned ResNet18,
and checking **what the model actually learned**.

> A learning project, not a diagnostic tool.

## Summary

- **Leakage:** 370 of the 1,600 test images (23%) are duplicates of training images, mostly in the no-tumor class (257 of 400).
- **Source bias:** the no-tumor images come from a different source than most tumor images. They differ in size, color format and brightness, and some have website banners.
- **Result:** after removing the leakage, my model (v1) scores **93.0%** on the test set.
- **What it learned:** it partly recognizes the image source instead of the tumor. Gliomas from the main source: 315 of 318 correct (99%).
  Gliomas from another source: 13 of 63 correct (21%), and 38 of them (60%) were called "no tumor".
  Grad-CAM points the same way: for the missed gliomas I checked, the model looked at the edge of the image, not at the tumor.
- **Fix attempt (v2):** a new split with both sources in training. It helped a little on the small other-source group,
  but not enough to call it a real improvement. More data from other sources would be the real fix.

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar, version 2
(7,200 images, 4 classes, license CC BY 4.0). It combines three public datasets: figshare, SARTAJ and Br35H.
According to the dataset page, all no-tumor images come from Br35H, which explains the source difference.
There is no source label per image, so I use the image size as a proxy: images that are exactly 512x512 (almost all tumor images)
are the "main source", all others the "other source".

The dataset page says version 2 removed duplicates and the overlap between train and test.
I still found 370 test images with an identical copy in the training set (see `01_eda`, section 6).

## Approach

1. [`01_eda`](notebooks/01_eda.ipynb): class balance, image sizes and color modes, brightness and black border per class.
   Duplicates were found with perceptual hashing (identical hashes only).
2. [`02_preprocessing`](notebooks/02_preprocessing.ipynb): remove duplicates, leakage and banner images (7,200 → 6,200),
   crop the black border, pad to square, equal brightness, resize to 224x224. Section 8 adds the v2 split, stratified on class and source.
3. [`03_training`](notebooks/03_training.ipynb): pretrained ResNet18 with AdamW, class weights, learning rate scheduler and early stopping.
   Learning rate picked on the validation set. Experiment with vs without the black border: no real difference. Section 15 trains v2.
4. [`04_evaluation`](notebooks/04_evaluation.ipynb): test scores, confusion matrix, error analysis, Grad-CAM (written by hand) and a fair v1 vs v2 comparison.

## Results

**v1** on the original test set (1,467 images after cleaning):

| class | precision | recall |
|---|---|---|
| glioma | 0.98 | 0.86 |
| meningioma | 0.95 | 0.88 |
| no tumor | 0.80 | 1.00 |
| pituitary | 0.99 | 0.99 |
| **accuracy** | | **93.0%** |

**v1 vs v2** on the 234 test images that neither model has seen (they were in the original test folder and in the v2 test set):

| group | images | v1 correct | v2 correct |
|---|---|---|---|
| glioma, other source | 9 | 2 | 5 |
| meningioma, other source | 19 | 12 | 14 |
| glioma + meningioma, main source | 103 | 103 | 99 |
| no tumor + pituitary | 103 | 103 | 103 |
| **accuracy** | 234 | **94.0%** | **94.4%** |

v2 got 5 more other-source tumors right, but lost 4 correct predictions in the main-source group, so overall accuracy barely changed.
With groups this small I can't conclude that v2 is better.

## Limitations

- No patient IDs, so scans of the same patient can end up in both train and test.
- Only identical hashes count as duplicates, so near-duplicates can still cause some leakage.
- The source is estimated from the image size (512x512 = main source), not from real metadata.
- The other-source tumor groups are small (63 gliomas and 135 meningiomas in total), so results on them are uncertain.

Built with Python, PyTorch, scikit-learn, pandas, NumPy, matplotlib, Pillow, SciPy and imagehash.
