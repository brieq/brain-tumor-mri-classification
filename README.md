# Brain Tumor MRI Classification

Classifying brain MRI scans into glioma, meningioma, pituitary tumor or no tumor with a fine-tuned ResNet18,
and finding out **what the model actually learned**.

> A learning project, not a diagnostic tool.

## The short version

At first sight this dataset looks easy, but two things make it harder than it seems:

- **23% of the test images also appear in the training set** (64% for no tumor), because of duplicates.
- **The no-tumor images come from a different source** than the tumor images (size, color format, brightness, some even have website banners).

After removing the leakage, my model scores **93%**. The error analysis shows why: it partly recognizes the **source**, not the tumor.
Gliomas from the main source are 99% correct, gliomas from another source only **21%** (60% are called "no tumor").
Grad-CAM points the same way: for the missed gliomas I checked, the model looked at the edge of the image instead of the tumor.

A new split with both sources in training (v2) helped: other-source gliomas went from 2/9 to 5/9 correct on unseen test images.
But that group is too small to call it proof. **The real fix is more data from other sources.**

## Approach

Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (7,200 images, 4 classes).

1. [`01_eda`](notebooks/01_eda.ipynb): class balance, image sizes and color modes, duplicates with perceptual hashing, brightness and black border per class.
2. [`02_preprocessing`](notebooks/02_preprocessing.ipynb): remove duplicates, leakage and banner images (7,200 → 6,200), crop the black border,
   pad to square, equal brightness, resize to 224x224. Section 8 adds the v2 split, stratified on class and source.
3. [`03_training`](notebooks/03_training.ipynb): pretrained ResNet18 with AdamW, class weights, learning rate scheduler and early stopping.
   Learning rate picked on validation. Experiment with vs without the black border: no real difference. Section 15 trains v2.
4. [`04_evaluation`](notebooks/04_evaluation.ipynb): test scores, confusion matrix, error analysis, Grad-CAM (by hand) and a fair v1 vs v2 comparison.

## Results

| v1, original test set (1,467 images) | precision | recall |
|---|---|---|
| glioma | 0.98 | 0.86 |
| meningioma | 0.95 | 0.88 |
| no tumor | 0.80 | 1.00 |
| pituitary | 0.99 | 0.99 |
| **accuracy** | | **93.0%** |

| v1 vs v2, 234 test images neither model has seen | images | v1 | v2 |
|---|---|---|---|
| glioma, other source | 9 | 2 | 5 |
| meningioma, other source | 19 | 12 | 14 |
| glioma + meningioma, main source | 103 | 103 | 99 |
| no tumor + pituitary | 103 | 103 | 103 |
| **accuracy** | 234 | **94.0%** | **94.4%** |

## Limitations

- No patient IDs, so scans of the same patient can end up in both train and test.
- Only exact duplicates are found, not near-duplicates.
- The source is estimated from the image size (512x512 = main source), not from real metadata.
- The other-source tumor groups are small, so the v2 results on that group are uncertain.

Built with Python, PyTorch, scikit-learn, pandas, NumPy, matplotlib, Pillow, SciPy and imagehash.
