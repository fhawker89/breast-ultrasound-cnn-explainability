"""MC-Dropout-style uncertainty estimation for the already-trained 88.9%
baseline 3-class model -- with no retraining required.

The baseline checkpoint (resnet18_busi.pt) was trained with a plain
`nn.Linear` classification head (no dropout). Since `nn.Dropout` has no
learnable parameters of its own, we can rebuild the model with a dropout
layer wrapped around that *same* trained Linear layer, load the existing
weights straight in, and get stochastic, dropout-based uncertainty estimates
at inference time -- without training anything from scratch.

This is a faster, more approximate cousin of "proper" MC-Dropout (where the
network is trained from the start with dropout active, so its weights adapt
to be robust to it). Worth being upfront about that distinction if asked --
see the printed caveat in main().
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BUSIDataset, CLASSES, collect_samples, set_seed, stratified_split
from train import CHECKPOINT_PATH, DEVICE, RAW_DIR, ROOT_DIR, build_model

FIGURES_DIR = os.path.join(ROOT_DIR, "outputs", "figures")
METRICS_PATH = os.path.join(ROOT_DIR, "outputs", "uncertainty_baseline_metrics.json")
DROPOUT_P = 0.3
MC_SAMPLES = 30
BATCH_SIZE = 16


def load_baseline_with_test_time_dropout():
    """Build the model WITH a dropout layer in the head, then load the
    baseline checkpoint's weights into it. The checkpoint's `fc.weight` /
    `fc.bias` (a plain nn.Linear, trained without dropout) get remapped onto
    `fc.1.weight` / `fc.1.bias` (index 1 of the new nn.Sequential(Dropout,
    Linear) head) -- same trained weights, just wrapped with a dropout layer
    in front of them for inference-time stochasticity."""
    model = build_model(num_classes=len(CLASSES), freeze_backbone=True, dropout_p=DROPOUT_P).to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    remapped = {
        (k.replace("fc.", "fc.1.") if k.startswith("fc.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(remapped)
    return model


def enable_mc_dropout(model):
    """Keep the model in eval mode (BatchNorm frozen, using its learned
    running stats) but force Dropout layers to stay stochastic -- the
    standard MC-Dropout inference trick."""
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def predictive_entropy(mean_probs):
    """Standard multi-class uncertainty measure: entropy of the average
    prediction across the T stochastic passes. 0 = completely confident in
    one class every time; higher = probability mass spread across classes."""
    eps = 1e-12
    return -np.sum(mean_probs * np.log(mean_probs + eps), axis=-1)


def main():
    set_seed(42)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    samples = collect_samples(RAW_DIR)
    _, _, test_samples = stratified_split(samples)
    loader = DataLoader(BUSIDataset(test_samples, train=False), batch_size=BATCH_SIZE)

    model = load_baseline_with_test_time_dropout()
    enable_mc_dropout(model)

    all_mean_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            # T stochastic forward passes per batch -- each one has a
            # different random dropout mask, so gives a slightly different
            # prediction. Averaging them gives the predictive mean; the
            # spread across them is the uncertainty signal.
            batch_probs = torch.stack(
                [torch.softmax(model(images), dim=1) for _ in range(MC_SAMPLES)]
            )  # shape: [MC_SAMPLES, batch, n_classes]
            mean_probs = batch_probs.mean(dim=0).cpu().numpy()
            all_mean_probs.extend(mean_probs.tolist())
            all_labels.extend(labels.tolist())

    all_mean_probs = np.array(all_mean_probs)
    all_labels = np.array(all_labels)
    preds = all_mean_probs.argmax(axis=1)
    correct = preds == all_labels
    entropy = predictive_entropy(all_mean_probs)

    mc_accuracy = float(correct.mean())
    entropy_correct = entropy[correct]
    entropy_incorrect = entropy[~correct]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(
        [entropy_correct, entropy_incorrect],
        labels=[f"Correct (n={correct.sum()})", f"Incorrect (n={(~correct).sum()})"],
    )
    ax.set_ylabel("Predictive entropy (MC-Dropout, 30 passes)")
    ax.set_title("Does the model's uncertainty track its correctness?")
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "baseline_mc_dropout_uncertainty.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    summary = {
        "method": "test-time dropout (dropout inserted post-hoc into a model trained without it)",
        "dropout_p": DROPOUT_P,
        "mc_samples": MC_SAMPLES,
        "mc_dropout_mean_accuracy": mc_accuracy,
        "mean_entropy_correct": float(entropy_correct.mean()),
        "mean_entropy_incorrect": float(entropy_incorrect.mean()) if len(entropy_incorrect) else None,
    }
    print(json.dumps(summary, indent=2))

    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
