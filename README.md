# Breast Ultrasound Classification with Grad-CAM Explainability

A CNN pipeline for classifying breast ultrasound images (normal / benign / malignant) using
transfer learning, with Grad-CAM explainability validated against radiologist-annotated
segmentation masks.

## Dataset

[BUSI (Breast Ultrasound Images)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) -
780 images from 600 patients, each with ground-truth lesion segmentation masks, across three
classes: normal (133), benign (437), malignant (210).

## Method

- **Model**: ResNet18 pretrained on ImageNet, fine-tuned (backbone layer4 + classification head)
  for 3-class classification.
- **Split**: stratified 70/15/15 train/val/test (seed=42), preserving class balance across splits.
- **Augmentation**: random horizontal flip, rotation, colour jitter (train only).
- **Explainability**: Grad-CAM heatmaps on the final convolutional block, overlaid against the
  dataset's ground-truth segmentation masks to check whether the model attends to clinically
  relevant regions.

## Setup

```bash
pip install -r requirements.txt
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset -p data/raw --unzip
```

## Usage

```bash
cd src
python train.py     # trains model, saves checkpoint + metrics to outputs/
python gradcam.py    # generates Grad-CAM vs. ground-truth mask figures to outputs/figures/
```

## Results

See [REPORT.md](REPORT.md) for full write-up, results, and discussion.

## Repo structure

```
src/
  dataset.py    # data loading, stratified split, transforms
  train.py      # model definition, training loop, evaluation
  gradcam.py    # Grad-CAM generation + mask overlay
outputs/
  checkpoints/  # trained model weights (gitignored)
  figures/      # Grad-CAM visualisations
  metrics.json  # training history, test metrics, confusion matrix
```

## Limitations

This project uses breast **ultrasound** imagery (BUSI) rather than mammography, chosen for
dataset size and same-day feasibility. The classification + Grad-CAM pipeline here transfers
directly to mammography datasets (e.g. CBIS-DDSM) given more time and compute.
