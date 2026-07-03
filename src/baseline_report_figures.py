"""Generate extra report figures from the already-trained 88.9%-accuracy
baseline checkpoint. This does a single inference pass over the test set --
no training or retraining involved, so it's cheap to (re)run at any time.

Produces:
- confusion matrix (raw counts + %, already used) and a pure normalised version
- per-class precision/recall/F1 bar chart
- one-vs-rest ROC curves + AUC per class
- loss/accuracy vs. epoch curves (from the training history already saved)
- a gallery of misclassified test images, for a "failure analysis" discussion
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BUSIDataset, CLASSES, collect_samples, set_seed, stratified_split
from gradcam import make_input_tensor
from plots import (
    plot_class_metrics_bar,
    plot_confusion_matrix,
    plot_loss_accuracy,
    plot_normalized_confusion_matrix,
    plot_roc_curves,
)
from train import CHECKPOINT_PATH, DEVICE, METRICS_PATH, RAW_DIR, ROOT_DIR, build_model

FIGURES_DIR = os.path.join(ROOT_DIR, "outputs", "figures")
MAX_MISCLASSIFIED = 9


def run_inference(model, test_samples):
    """One forward pass over the test set, collecting true labels, predicted
    labels, and full softmax probability vectors (needed for ROC curves,
    which the original training run didn't save)."""
    loader = DataLoader(BUSIDataset(test_samples, train=False), batch_size=16)
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            probs = torch.softmax(model(images), dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(probs.argmax(axis=1).tolist())
            all_labels.extend(labels.tolist())
    return all_labels, all_preds, all_probs


def plot_misclassified_gallery(test_samples, labels, preds, probs, out_path, max_examples=MAX_MISCLASSIFIED):
    wrong = [
        (s, t, p, pr) for s, t, p, pr in zip(test_samples, labels, preds, probs) if t != p
    ][:max_examples]
    if not wrong:
        print("No misclassified examples to plot -- skipping gallery.")
        return

    n_cols = 3
    n_rows = (len(wrong) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten() if len(wrong) > 1 else [axes]

    for ax, (sample, true_idx, pred_idx, prob) in zip(axes, wrong):
        _, rgb_image = make_input_tensor(sample["image_path"])
        ax.imshow(rgb_image)
        ax.set_title(
            f"true: {CLASSES[true_idx]}\npred: {CLASSES[pred_idx]} ({prob[pred_idx]:.2f})",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[len(wrong):]:
        ax.axis("off")

    fig.suptitle("Misclassified test examples")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    set_seed(42)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    samples = collect_samples(RAW_DIR)
    _, _, test_samples = stratified_split(samples)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    labels, preds, probs = run_inference(model, test_samples)

    plot_confusion_matrix(
        metrics["confusion_matrix"], CLASSES,
        "Baseline confusion matrix (counts + row %)",
        os.path.join(FIGURES_DIR, "baseline_confusion_matrix.png"),
    )
    plot_normalized_confusion_matrix(
        metrics["confusion_matrix"], CLASSES,
        "Baseline confusion matrix (row-normalised)",
        os.path.join(FIGURES_DIR, "baseline_confusion_matrix_normalized.png"),
    )
    plot_class_metrics_bar(
        metrics["classification_report"], CLASSES,
        "Baseline per-class precision / recall / F1",
        os.path.join(FIGURES_DIR, "baseline_class_metrics.png"),
    )
    plot_roc_curves(
        labels, np.array(probs), CLASSES,
        "Baseline ROC curves (one-vs-rest)",
        os.path.join(FIGURES_DIR, "baseline_roc_curves.png"),
    )
    plot_loss_accuracy(
        metrics["history"],
        "Baseline training curves",
        os.path.join(FIGURES_DIR, "baseline_loss_accuracy.png"),
    )
    plot_misclassified_gallery(
        test_samples, labels, preds, probs,
        os.path.join(FIGURES_DIR, "baseline_misclassified.png"),
    )


if __name__ == "__main__":
    main()
