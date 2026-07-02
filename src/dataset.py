"""BUSI dataset loading, stratified splitting, and PyTorch Dataset class.

This module only handles getting data ready to feed into a model -- finding
the image/mask files on disk, splitting them into train/val/test, and
defining the image transforms. No model code lives here.
"""
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

# Fixed class ordering used everywhere in the project (labels are indices
# into this list, e.g. "benign" is always label 1). Keeping one canonical
# list avoids the different scripts accidentally using inconsistent label
# numbering.
CLASSES = ["normal", "benign", "malignant"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# The model is a ResNet18 pretrained on ImageNet, so inputs must be
# normalised with the same per-channel mean/std ImageNet was trained with --
# otherwise the pretrained weights would be looking at pixel statistics
# they've never seen and transfer learning would work much worse.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224  # the input resolution ResNet18 was trained on


def collect_samples(raw_dir):
    """Walk the BUSI folder structure and build one record per ultrasound image.

    BUSI stores each class in its own subfolder (normal/benign/malignant).
    Each real image (e.g. "benign (12).png") has one or more matching mask
    files (e.g. "benign (12)_mask.png", and "_mask_1.png", "_mask_2.png"...
    for images with more than one annotated lesion). This function pairs
    every image with all of its mask files, so downstream code always knows
    where the ground-truth lesion outline is for a given image.

    Returns a list of dicts: {image_path, mask_paths, label, class_name}.
    """
    samples = []
    for class_name in CLASSES:
        class_dir = os.path.join(raw_dir, class_name)
        images = sorted(glob.glob(os.path.join(class_dir, "*.png")))
        # Mask files also end in .png and live in the same folder, so they'd
        # otherwise show up as if they were extra images -- filter them out.
        images = [p for p in images if "_mask" not in os.path.basename(p)]
        for img_path in images:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            # Glob for "<stem>_mask*.png" to catch both "_mask.png" and the
            # numbered "_mask_1.png", "_mask_2.png" variants for multi-lesion images.
            mask_paths = sorted(
                glob.glob(os.path.join(class_dir, f"{stem}_mask*.png"))
            )
            samples.append(
                {
                    "image_path": img_path,
                    "mask_paths": mask_paths,
                    "label": CLASS_TO_IDX[class_name],
                    "class_name": class_name,
                }
            )
    return samples


def stratified_split(samples, val_size=0.15, test_size=0.15, seed=42):
    """Split samples into train/val/test (70/15/15 by default), preserving
    the normal/benign/malignant class balance in every split.

    BUSI is imbalanced (437 benign vs. 210 malignant vs. 133 normal), so a
    plain random split risks e.g. the test set ending up with too few
    malignant examples to evaluate reliably. `stratify=` makes sklearn keep
    each split's class proportions matching the full dataset's.

    This is a two-step split (train_test_split can only carve off one chunk
    at a time): first pull out the test set, then split what's left into
    train/val. `relative_val` re-expresses val_size as a fraction of the
    *remaining* data after the test set was removed, so the final train/val/test
    proportions come out as intended (e.g. 0.15 of the original 100%, not
    0.15 of the leftover 85%).
    """
    labels = [s["label"] for s in samples]
    train_val, test = train_test_split(
        samples, test_size=test_size, stratify=labels, random_state=seed
    )
    train_val_labels = [s["label"] for s in train_val]
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val_labels, random_state=seed
    )
    return train, val, test


def get_transforms(train: bool):
    """Image preprocessing pipeline. Training and eval use different
    pipelines: training adds random augmentation, eval doesn't."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                # Augmentation: with only ~550 training images, the model can
                # easily memorise them instead of learning general features.
                # These transforms create slightly different versions of each
                # image every epoch, which discourages that. Ultrasound
                # images have no fixed "up" orientation and lesion shape
                # shouldn't depend on left/right, so flips and small
                # rotations/brightness changes are safe -- they don't change
                # what class the image actually belongs to.
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    # No augmentation for validation/test: we want a stable, repeatable
    # measurement of how the model performs, not extra randomness.
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class BUSIDataset(Dataset):
    """PyTorch Dataset wrapping a list of sample dicts from collect_samples().

    Loads and transforms one image at a time -- PyTorch's DataLoader calls
    __getitem__ behind the scenes to build batches, so nothing here runs
    until training/evaluation actually asks for data.
    """

    def __init__(self, samples, train: bool, label_fn=None):
        """label_fn: optional callable(sample) -> int, overrides sample['label'].
        Used elsewhere in the project to derive binary sub-task labels (e.g.
        normal-vs-abnormal for a cascade classifier) without duplicating the
        underlying 3-class sample list. Not used by the original 3-class
        model -- it defaults to just returning sample['label'] unchanged."""
        self.samples = samples
        self.transform = get_transforms(train)
        self.label_fn = label_fn or (lambda s: s["label"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Images are loaded from disk lazily (not all at once into memory) --
        # fine at this dataset size, and keeps memory usage low.
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)
        return image, self.label_fn(sample)


def set_seed(seed=42):
    """Fix every source of randomness this project touches (Python's random
    module, NumPy, and PyTorch) so that runs are reproducible -- the same
    seed gives the same train/val/test split and the same model
    initialisation/training trajectory."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
