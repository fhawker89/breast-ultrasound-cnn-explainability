"""Train a ResNet18 transfer-learning classifier on BUSI (normal/benign/malignant)."""
import json
import os

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from dataset import BUSIDataset, CLASSES, collect_samples, set_seed, stratified_split

RAW_DIR = os.path.join("data", "raw", "Dataset_BUSI_with_GT")
CHECKPOINT_PATH = os.path.join("outputs", "checkpoints", "resnet18_busi.pt")
METRICS_PATH = os.path.join("outputs", "metrics.json")
EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes=len(CLASSES)):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def main():
    set_seed(42)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    samples = collect_samples(RAW_DIR)
    train_samples, val_samples, test_samples = stratified_split(samples)
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}  Test: {len(test_samples)}")

    train_loader = DataLoader(BUSIDataset(train_samples, train=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BUSIDataset(val_samples, train=False), batch_size=BATCH_SIZE)
    test_loader = DataLoader(BUSIDataset(test_samples, train=False), batch_size=BATCH_SIZE)

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    test_loss, test_acc, test_preds, test_labels = run_epoch(model, test_loader, criterion)
    report = classification_report(test_labels, test_preds, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(test_labels, test_preds).tolist()

    print(f"\nTest accuracy: {test_acc:.4f}")
    print(classification_report(test_labels, test_preds, target_names=CLASSES))

    with open(METRICS_PATH, "w") as f:
        json.dump({"history": history, "test_acc": test_acc, "test_loss": test_loss, "classification_report": report, "confusion_matrix": cm, "classes": CLASSES}, f, indent=2)


if __name__ == "__main__":
    main()
