"""Shared plotting helpers: confusion matrix heatmaps and loss/accuracy-vs-epoch curves."""
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, title, out_path):
    cm = np.array(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm_pct[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]*100:.0f}%)", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="row-normalised fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_normalized_confusion_matrix(cm, classes, title, out_path):
    """Pure row-normalised confusion matrix (values 0-1, no raw counts) --
    the standard format for comparing per-class performance when classes
    have very different support sizes (here: 133 normal vs 437 benign vs
    210 malignant), since raw counts alone make the minority class look
    artificially small in a heatmap."""
    cm = np.array(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color=color, fontsize=11)

    fig.colorbar(im, ax=ax, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_class_metrics_bar(report, classes, title, out_path):
    """Grouped bar chart of precision/recall/F1 per class, from an
    sklearn classification_report(output_dict=True) dict."""
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, metric in enumerate(metrics):
        values = [report[c][metric] for c in classes]
        ax.bar(x + (i - 1) * width, values, width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_roc_curves(y_true, y_probs, classes, title, out_path):
    """One-vs-rest ROC curve + AUC for each class. y_probs is an
    [n_samples, n_classes] array of predicted probabilities."""
    from sklearn.metrics import roc_curve, auc

    y_true = np.array(y_true)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for class_idx, class_name in enumerate(classes):
        binary_true = (y_true == class_idx).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_probs[:, class_idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_loss_accuracy(history, title, out_path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
