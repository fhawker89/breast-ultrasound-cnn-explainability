"""Train a ResNet18 transfer-learning classifier on BUSI (normal/benign/malignant).

This is the original single-model pipeline that produced the 88.9% test
accuracy baseline result. It trains one 3-way classifier directly (as
opposed to the later cascade approach, which splits the decision into two
simpler binary stages -- see cascade.py for that).
"""
import json
import os

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from dataset import BUSIDataset, CLASSES, collect_samples, set_seed, stratified_split

# All paths are built from this file's own location rather than the current
# working directory, so the script behaves the same whether you run it as
# `python train.py` from inside src/, or `python src/train.py` from the
# project root -- both were tried during development and the first one
# silently produced an empty dataset because "data/" resolved relative to
# src/ instead of the project root.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw", "Dataset_BUSI_with_GT")
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "outputs", "checkpoints", "resnet18_busi.pt")
METRICS_PATH = os.path.join(ROOT_DIR, "outputs", "metrics.json")
EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-4
# Uses a GPU automatically if one's available; falls back to CPU otherwise.
# This project was trained entirely on CPU (no local GPU), which is why the
# dataset/epoch count were kept small enough to be practical without one.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes=len(CLASSES), freeze_backbone=True, dropout_p=0.0):
    """Build a ResNet18 classifier, starting from ImageNet-pretrained weights.

    Why transfer learning at all: with only ~550 training images, there's
    nowhere near enough data to learn general visual features (edges,
    textures, shapes) from scratch. ImageNet pretraining gives the network
    those low-level features for free; fine-tuning only teaches it what's
    specific to ultrasound images on top of that foundation.

    freeze_backbone=True (used by this file's baseline model) only trains
    the last residual block (`layer4`) plus the new classification head,
    leaving the earlier, more generic layers untouched. This is a smaller,
    faster, lower-overfitting-risk starting point than fine-tuning the whole
    network -- appropriate for a first baseline on a small dataset.
    (freeze_backbone=False and dropout_p are used by the later cascade
    experiments, which fully fine-tune the network instead -- unused here
    but kept in this shared function so both approaches use identical model
    construction code.)
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        # Freeze everything first...
        for param in model.parameters():
            param.requires_grad = False
        # ...then re-enable gradients just for the final block, so only it
        # (plus the replaced classification head below) gets updated during
        # training.
        for param in model.layer4.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    if dropout_p > 0:
        # Dropout before the final linear layer randomly zeroes some of its
        # inputs during training, which discourages the head from relying
        # too heavily on any single feature -- a regularisation technique.
        # Not used in the baseline (dropout_p=0.0 by default); added for the
        # later MC-Dropout uncertainty-estimation experiment.
        model.fc = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(model.fc.in_features, num_classes))
    else:
        # Replace ImageNet's original 1000-class output layer with a fresh,
        # randomly-initialised layer sized for our 3 classes. This new layer
        # is what actually learns to tell normal/benign/malignant apart.
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer=None):
    """Run one full pass over `loader`'s data. Used for both training and
    evaluation -- passing optimizer=None switches it into eval mode (no
    gradient updates), so this one function avoids duplicating the loop
    logic in two places."""
    is_train = optimizer is not None
    # model.train() enables dropout/lets BatchNorm update its running stats;
    # model.eval() turns both off, which matters when measuring true
    # validation/test performance rather than training-time behaviour.
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    # torch.set_grad_enabled(False) during evaluation stops PyTorch building
    # the computation graph needed for backprop, which isn't needed for
    # inference and would otherwise waste memory/time.
    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)  # raw class scores (logits), shape [batch, 3]
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()  # clear gradients from the previous batch
                loss.backward()  # compute gradients for this batch
                optimizer.step()  # update the trainable weights

            # loss.item() is the *average* loss for this batch, so multiply
            # by batch size before accumulating -- otherwise batches of
            # different sizes (the last batch in an epoch is often smaller)
            # would be weighted incorrectly when averaging over the epoch.
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)  # highest-scoring class = prediction
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def main():
    set_seed(42)  # same seed as everywhere else in the project, for a reproducible split
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    samples = collect_samples(RAW_DIR)
    train_samples, val_samples, test_samples = stratified_split(samples)
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}  Test: {len(test_samples)}")

    train_loader = DataLoader(BUSIDataset(train_samples, train=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BUSIDataset(val_samples, train=False), batch_size=BATCH_SIZE)
    test_loader = DataLoader(BUSIDataset(test_samples, train=False), batch_size=BATCH_SIZE)

    model = build_model().to(DEVICE)  # uses the defaults: frozen backbone, no dropout, 3 classes
    criterion = nn.CrossEntropyLoss()
    # filter(...) passes the optimizer only the parameters that actually have
    # requires_grad=True -- i.e. just layer4 + the new head, since the rest
    # of the network was frozen in build_model(). Optimizing frozen
    # parameters would do nothing but waste compute.
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        # Keep only the checkpoint from whichever epoch had the best
        # validation accuracy so far -- not necessarily the last epoch.
        # This protects against overfitting: if the model starts memorising
        # the training set in later epochs, val_acc will stop improving (or
        # get worse) and we simply won't save those weights.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    # Reload the best-on-validation checkpoint (not whatever the model looks
    # like after the final epoch) before measuring final test performance --
    # this is the number that gets reported as "the" result.
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    test_loss, test_acc, test_preds, test_labels = run_epoch(model, test_loader, criterion)
    report = classification_report(test_labels, test_preds, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(test_labels, test_preds).tolist()

    print(f"\nTest accuracy: {test_acc:.4f}")
    print(classification_report(test_labels, test_preds, target_names=CLASSES))

    # Save everything needed to inspect/plot results later without having to
    # retrain: per-epoch history (for loss/accuracy curves), the final test
    # metrics, and the confusion matrix.
    with open(METRICS_PATH, "w") as f:
        json.dump({"history": history, "test_acc": test_acc, "test_loss": test_loss, "classification_report": report, "confusion_matrix": cm, "classes": CLASSES}, f, indent=2)


if __name__ == "__main__":
    main()
