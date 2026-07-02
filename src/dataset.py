"""BUSI dataset loading, stratified splitting, and PyTorch Dataset class."""
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ["normal", "benign", "malignant"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def collect_samples(raw_dir):
    """Return list of dicts: {image_path, mask_paths, label, class_name}."""
    samples = []
    for class_name in CLASSES:
        class_dir = os.path.join(raw_dir, class_name)
        images = sorted(glob.glob(os.path.join(class_dir, "*.png")))
        images = [p for p in images if "_mask" not in os.path.basename(p)]
        for img_path in images:
            stem = os.path.splitext(os.path.basename(img_path))[0]
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
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class BUSIDataset(Dataset):
    def __init__(self, samples, train: bool):
        self.samples = samples
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)
        return image, sample["label"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
