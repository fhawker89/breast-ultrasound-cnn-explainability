# Breast Ultrasound Classification with Grad-CAM Explainability

## Motivation

AI methods for breast cancer detection are only clinically useful if they are both accurate
and interpretable — a model that flags a scan as malignant without indicating why offers little
to a radiologist deciding whether to trust it. This project builds a CNN classifier for breast
ultrasound images and validates its explanations against radiologist-annotated lesion masks,
as a small-scale demonstration of the fair/explainable AI approach central to breast imaging
research.

## Method

**Dataset.** [BUSI (Breast Ultrasound Images)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset):
780 images from 600 patients, labelled normal (133), benign (437), or malignant (210), each
with a ground-truth lesion segmentation mask.

**Split.** Stratified 70/15/15 train/validation/test split (545/118/117 images), preserving
class balance across all three sets.

**Model.** ResNet18, pretrained on ImageNet, fine-tuned on this task: the final convolutional
block (`layer4`) and classification head are trained, earlier layers are frozen. Training used
Adam (lr=1e-4) for 15 epochs, with the best validation-accuracy checkpoint kept for evaluation.

**Augmentation.** Random horizontal flip, small rotation (±10°), and colour jitter on the
training set only.

**Explainability.** Grad-CAM heatmaps were computed on `layer4` for held-out test images and
overlaid against the dataset's ground-truth segmentation masks, to check whether the regions
driving each prediction correspond to the actual lesion location rather than spurious
background texture.

## Results

Test accuracy: **88.9%** (104/117 correct).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.88 | 0.75 | 0.81 | 20 |
| Benign | 0.86 | 0.95 | 0.91 | 66 |
| Malignant | 0.96 | 0.84 | 0.90 | 31 |

Confusion matrix (rows = true, columns = predicted; order normal/benign/malignant):

```
[15,  5,  0]
[ 2, 63,  1]
[ 0,  5, 26]
```

Malignant precision is highest of the three classes (0.96), and critically, **no malignant
case was misclassified as normal** — the model's errors on malignant cases were confusions
with benign, the clinically safer failure mode. Most misclassifications involve normal images
being read as benign, plausibly because "normal" tissue can still contain benign-looking
structures not marked as lesions.

Grad-CAM visualisations (see `outputs/figures/`) show the model's attention concentrated
directly over the annotated lesion in the large majority of correctly classified benign and
malignant cases, rather than on surrounding tissue or imaging artefacts (probe markers, text
overlays) — evidence that the model is learning the lesion itself as the discriminative
feature, not a confound.

## Discussion

At under 800 training images, a lightweight transfer-learning approach (freezing most of a
pretrained backbone) was appropriate — training a CNN from scratch on this dataset size would
likely overfit. The Grad-CAM/mask agreement suggests the model's predictions are grounded in
the correct anatomical region, which is the minimum bar for trusting a classifier's output in
a clinical-adjacent setting, though it does not by itself establish clinical validity.

## Limitations

- **Ultrasound, not mammography.** This project used BUSI (breast ultrasound) rather than a
  mammography dataset, chosen for its small size and fast turnaround. The same pipeline
  (transfer learning + Grad-CAM-vs-mask validation) transfers directly to mammography datasets
  such as CBIS-DDSM, given more compute and time.
- **No formal fairness/subgroup evaluation.** BUSI does not include demographic metadata
  beyond patient age ranges, so no subgroup fairness analysis (e.g. by density, age, ethnicity)
  was performed here — a natural next step for the "fair" half of fair/explainable AI.
- **Small test set (117 images).** Class-wise metrics, especially for the minority "normal"
  class, carry meaningful uncertainty at this sample size.
- **Grad-CAM overlap was assessed qualitatively**, not with a quantitative localisation metric
  (e.g. IoU between the CAM's thresholded region and the ground-truth mask), which would be a
  natural addition for a more rigorous evaluation.
