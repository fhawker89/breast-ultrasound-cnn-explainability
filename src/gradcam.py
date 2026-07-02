"""Generate Grad-CAM heatmaps and overlay them against ground-truth segmentation masks."""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import (
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    collect_samples,
    get_transforms,
    set_seed,
    stratified_split,
)
from train import RAW_DIR, CHECKPOINT_PATH, DEVICE, ROOT_DIR, build_model

FIGURES_DIR = os.path.join(ROOT_DIR, "outputs", "figures")
SAMPLES_PER_CLASS = 3


def load_mask(mask_paths, size=IMG_SIZE):
    """Combine (possibly multiple) mask files into one binary mask, resized to model input size."""
    if not mask_paths:
        return np.zeros((size, size), dtype=np.uint8)
    combined = None
    for path in mask_paths:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        combined = m if combined is None else np.maximum(combined, m)
    combined = cv2.resize(combined, (size, size))
    return (combined > 127).astype(np.uint8)


def make_input_tensor(image_path):
    transform = get_transforms(train=False)
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    rgb_image = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    return tensor, rgb_image


def main():
    set_seed(42)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    samples = collect_samples(RAW_DIR)
    _, _, test_samples = stratified_split(samples)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    for class_idx, class_name in enumerate(CLASSES):
        class_samples = [s for s in test_samples if s["label"] == class_idx][:SAMPLES_PER_CLASS]

        fig, axes = plt.subplots(len(class_samples), 3, figsize=(9, 3 * len(class_samples)))
        if len(class_samples) == 1:
            axes = axes.reshape(1, -1)

        for row, sample in enumerate(class_samples):
            input_tensor, rgb_image = make_input_tensor(sample["image_path"])
            input_tensor = input_tensor.to(DEVICE)

            grayscale_cam = cam(input_tensor=input_tensor)[0]
            cam_overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

            gt_mask = load_mask(sample["mask_paths"])

            with torch.no_grad():
                pred_idx = model(input_tensor).argmax(dim=1).item()
            pred_name = CLASSES[pred_idx]

            axes[row, 0].imshow(rgb_image)
            axes[row, 0].set_title("Ultrasound image")
            axes[row, 1].imshow(rgb_image)
            axes[row, 1].imshow(gt_mask, cmap="Reds", alpha=0.4)
            axes[row, 1].set_title("Ground-truth mask")
            axes[row, 2].imshow(cam_overlay)
            axes[row, 2].set_title(f"Grad-CAM (pred: {pred_name})")

            for col in range(3):
                axes[row, col].axis("off")

        fig.suptitle(f"Class: {class_name}")
        fig.tight_layout()
        out_path = os.path.join(FIGURES_DIR, f"gradcam_{class_name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
