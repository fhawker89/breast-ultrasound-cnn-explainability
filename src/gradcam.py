"""Generate Grad-CAM heatmaps and overlay them against ground-truth segmentation masks.

Grad-CAM ("Gradient-weighted Class Activation Mapping") produces a heatmap
showing which regions of an input image most influenced a CNN's prediction.
On its own that's just a picture -- the useful step this script adds is
placing that heatmap side-by-side with BUSI's radiologist-drawn lesion mask
for the same image, so you can actually check whether the model is looking
at the real lesion, or at something else entirely (an imaging artefact, a
probe marker, unrelated tissue) that happened to correlate with the label.
"""
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
SAMPLES_PER_CLASS = 3  # how many example images per class to visualise


def load_mask(mask_paths, size=IMG_SIZE):
    """Combine (possibly multiple) mask files into one binary mask, resized to model input size.

    Some BUSI images have more than one annotated lesion, stored as separate
    mask files (_mask.png, _mask_1.png, ...). np.maximum merges them into a
    single mask covering all annotated regions for that image. Images with
    no lesion (the "normal" class) have no mask files at all, so we return
    an all-zero mask instead of erroring.
    """
    if not mask_paths:
        return np.zeros((size, size), dtype=np.uint8)
    combined = None
    for path in mask_paths:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        combined = m if combined is None else np.maximum(combined, m)
    combined = cv2.resize(combined, (size, size))
    # Masks are saved as grayscale images where lesion pixels are near-white
    # (255) and background is near-black (0) -- threshold at the midpoint to
    # get a clean binary (0 or 1) mask.
    return (combined > 127).astype(np.uint8)


def make_input_tensor(image_path):
    """Load one image and return both:
    - a normalised tensor ready to feed into the model, and
    - a plain 0-1 float RGB array (no normalisation) for display/overlay purposes.

    Grad-CAM's own show_cam_on_image() helper expects a "normal-looking"
    0-1 RGB image to blend the heatmap onto -- feeding it the
    ImageNet-normalised tensor instead would produce a heatmap overlaid on
    a washed-out, wrong-looking image, so the two representations are kept
    separate here.
    """
    transform = get_transforms(train=False)
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # add a batch dimension of size 1
    rgb_image = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    return tensor, rgb_image


def main():
    set_seed(42)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    samples = collect_samples(RAW_DIR)
    # Same seed + same split logic as training, so this reproduces the exact
    # same test set the model was evaluated on -- these are genuinely
    # held-out images the model never trained on, not cherry-picked examples.
    _, _, test_samples = stratified_split(samples)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # Grad-CAM needs to know which layer's activations to explain. The last
    # convolutional block (layer4) is the standard choice: it's the deepest
    # set of features that still has spatial structure (a grid of positions
    # corresponding to regions of the input image) before the network
    # flattens everything down to a single classification vector -- exactly
    # what's needed to draw a heatmap over the image.
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    for class_idx, class_name in enumerate(CLASSES):
        class_samples = [s for s in test_samples if s["label"] == class_idx][:SAMPLES_PER_CLASS]

        # One figure per class, with 3 example images stacked as rows and
        # (original image / ground-truth mask / Grad-CAM overlay) as columns.
        fig, axes = plt.subplots(len(class_samples), 3, figsize=(9, 3 * len(class_samples)))
        if len(class_samples) == 1:
            # subplots() returns a 1D array instead of 2D when there's only
            # one row, which breaks the axes[row, col] indexing below --
            # reshape to keep the indexing consistent either way.
            axes = axes.reshape(1, -1)

        for row, sample in enumerate(class_samples):
            input_tensor, rgb_image = make_input_tensor(sample["image_path"])
            input_tensor = input_tensor.to(DEVICE)

            # Runs a forward + backward pass internally to compute which
            # pixels the prediction was most sensitive to; returns a
            # grayscale heatmap the same size as the input image.
            grayscale_cam = cam(input_tensor=input_tensor)[0]
            cam_overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

            gt_mask = load_mask(sample["mask_paths"])

            with torch.no_grad():
                pred_idx = model(input_tensor).argmax(dim=1).item()
            pred_name = CLASSES[pred_idx]

            # Column 1: the raw ultrasound image, for reference.
            axes[row, 0].imshow(rgb_image)
            axes[row, 0].set_title("Ultrasound image")
            # Column 2: the same image with the radiologist's lesion outline
            # overlaid in semi-transparent red -- this is the "ground truth"
            # to compare the model's attention against.
            axes[row, 1].imshow(rgb_image)
            axes[row, 1].imshow(gt_mask, cmap="Reds", alpha=0.4)
            axes[row, 1].set_title("Ground-truth mask")
            # Column 3: the Grad-CAM heatmap -- where the model actually
            # looked. Comparing this to column 2 by eye is the whole point
            # of the exercise.
            axes[row, 2].imshow(cam_overlay)
            axes[row, 2].set_title(f"Grad-CAM (pred: {pred_name})")

            for col in range(3):
                axes[row, col].axis("off")  # image plots don't need axis ticks/labels

        fig.suptitle(f"Class: {class_name}")
        fig.tight_layout()
        out_path = os.path.join(FIGURES_DIR, f"gradcam_{class_name}.png")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)  # free the figure's memory before starting the next one
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
