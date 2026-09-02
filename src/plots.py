from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .metrics import (
    average_history_curve,
    average_history_until_best,
    smooth_curve,
)


def _save_and_close(path: Path) -> None:
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def save_cv_plots(
    histories: list[dict],
    best_epochs: list[int],
    fold_results: pd.DataFrame,
    y_true: np.ndarray,
    oof_probabilities: np.ndarray,
    confusion: np.ndarray,
    auc: float,
    output_dir: str | Path,
) -> dict[str, str]:
    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    train_loss = average_history_curve(histories, "train_loss")
    validation_loss = average_history_curve(histories, "validation_loss")
    train_accuracy = average_history_curve(histories, "train_accuracy")
    validation_accuracy = average_history_curve(histories, "validation_accuracy")

    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(train_loss, label="Train Loss", linewidth=2)
    plt.plot(validation_loss, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Mean Cross-Validation Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = figure_dir / "full_loss_curve.png"
    _save_and_close(path)
    paths["full_loss_curve"] = str(path)

    train_loss_best = smooth_curve(
        average_history_until_best(histories, best_epochs, "train_loss")
    )
    validation_loss_best = smooth_curve(
        average_history_until_best(histories, best_epochs, "validation_loss")
    )
    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(train_loss_best, label="Train Loss", linewidth=2)
    plt.plot(validation_loss_best, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Mean Loss up to Each Fold's Best Epoch")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = figure_dir / "best_epoch_loss_curve.png"
    _save_and_close(path)
    paths["best_epoch_loss_curve"] = str(path)

    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(train_accuracy, label="Train Accuracy", linewidth=2)
    plt.plot(validation_accuracy, label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Mean Cross-Validation Accuracy")
    plt.ylim(0.4, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = figure_dir / "accuracy_curve.png"
    _save_and_close(path)
    paths["accuracy_curve"] = str(path)

    classes = ["TD", "ASD"]
    plt.figure(figsize=(5.8, 5.0), dpi=120)
    plt.imshow(confusion, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
    color_threshold = confusion.max() / 2.0
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            plt.text(
                column,
                row,
                format(confusion[row, column], "d"),
                ha="center",
                va="center",
                color="white"
                if confusion[row, column] > color_threshold
                else "black",
                fontsize=14,
                fontweight="bold",
            )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = figure_dir / "confusion_matrix.png"
    _save_and_close(path)
    paths["confusion_matrix"] = str(path)

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true, oof_probabilities
    )
    plt.figure(figsize=(6, 6), dpi=120)
    plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Out-of-Fold ROC Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = figure_dir / "roc_curve.png"
    _save_and_close(path)
    paths["roc_curve"] = str(path)

    x = np.arange(1, len(fold_results) + 1)
    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(x, fold_results["accuracy_thr"], marker="o", label="Accuracy")
    plt.plot(
        x,
        fold_results["balanced_accuracy_thr"],
        marker="o",
        label="Balanced Accuracy",
    )
    plt.plot(x, fold_results["f1_macro_thr"], marker="o", label="Macro F1")
    plt.plot(x, fold_results["auc"], marker="o", label="AUC")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.title("Fold Metrics (Exploratory Optimized Threshold)")
    plt.ylim(0.4, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = figure_dir / "fold_metrics.png"
    _save_and_close(path)
    paths["fold_metrics"] = str(path)
    return paths


def save_pearson_maps(
    fc_matrices: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path,
    atlas_name: str = "rois_ho",
) -> tuple[dict[str, float], dict[str, str]]:
    """Save ASD/TD mean FC matrices, their difference and edge distributions."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    asd_mean = fc_matrices[labels == 1].mean(axis=0)
    td_mean = fc_matrices[labels == 0].mean(axis=0)
    difference = asd_mean - td_mean
    upper = np.triu_indices(asd_mean.shape[0], k=1)
    asd_values = asd_mean[upper]
    td_values = td_mean[upper]
    difference_values = difference[upper]
    asd_global_mean = float(asd_values.mean())
    td_global_mean = float(td_values.mean())
    difference_limit = max(
        float(np.percentile(np.abs(difference_values), 99)), 1e-6
    )

    stats = {
        "asd_mean_connectivity": asd_global_mean,
        "td_mean_connectivity": td_global_mean,
        "asd_minus_td_mean": asd_global_mean - td_global_mean,
        "difference_min": float(difference.min()),
        "difference_max": float(difference.max()),
    }
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=120)
    first = axes[0, 0].imshow(
        asd_mean, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest"
    )
    axes[0, 0].set_title("ASD - Mean FC Matrix")
    axes[0, 0].set(xlabel="ROI", ylabel="ROI")
    fig.colorbar(first, ax=axes[0, 0], fraction=0.046, pad=0.04).set_label(
        "Pearson r"
    )

    second = axes[0, 1].imshow(
        td_mean, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest"
    )
    axes[0, 1].set_title("TD - Mean FC Matrix")
    axes[0, 1].set(xlabel="ROI", ylabel="ROI")
    fig.colorbar(second, ax=axes[0, 1], fraction=0.046, pad=0.04).set_label(
        "Pearson r"
    )

    third = axes[1, 0].imshow(
        difference,
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
        interpolation="nearest",
    )
    axes[1, 0].set_title("ASD - TD Difference\nRed: ASD higher | Blue: TD higher")
    axes[1, 0].set(xlabel="ROI", ylabel="ROI")
    fig.colorbar(third, ax=axes[1, 0], fraction=0.046, pad=0.04).set_label(
        "Delta Pearson r"
    )

    common_min = min(float(asd_values.min()), float(td_values.min()))
    common_max = max(float(asd_values.max()), float(td_values.max()))
    bins = np.linspace(common_min, common_max, 81)
    axes[1, 1].hist(
        asd_values,
        bins=bins,
        alpha=0.45,
        density=True,
        label=f"ASD | mean={asd_global_mean:.3f}",
    )
    axes[1, 1].hist(
        td_values,
        bins=bins,
        alpha=0.45,
        density=True,
        label=f"TD | mean={td_global_mean:.3f}",
    )
    axes[1, 1].axvline(asd_global_mean, linestyle="--", linewidth=1.5)
    axes[1, 1].axvline(td_global_mean, linestyle=":", linewidth=1.8)
    axes[1, 1].set_title("ASD vs TD - Edge Correlation Distribution")
    axes[1, 1].set(xlabel="Pearson correlation", ylabel="Density")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=9)

    fig.suptitle(
        f"Pearson Functional Connectivity | {atlas_name}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output / f"{atlas_name}_pearson_fc_overview.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths["overview"] = str(path)

    individual = {
        "asd_mean": (asd_mean, -1.0, 1.0, "ASD - Mean FC Matrix"),
        "td_mean": (td_mean, -1.0, 1.0, "TD - Mean FC Matrix"),
        "asd_minus_td": (
            difference,
            -difference_limit,
            difference_limit,
            "ASD - TD Difference Matrix",
        ),
    }
    for name, (matrix, lower, upper_limit, title) in individual.items():
        plt.figure(figsize=(6, 5), dpi=120)
        plt.imshow(
            matrix,
            cmap="coolwarm",
            vmin=lower,
            vmax=upper_limit,
            interpolation="nearest",
        )
        plt.title(f"{title} | {atlas_name}")
        plt.xlabel("ROI")
        plt.ylabel("ROI")
        plt.colorbar(label="Delta Pearson r" if name == "asd_minus_td" else "Pearson r")
        plt.tight_layout()
        path = output / f"{atlas_name}_{name}_fc.png"
        _save_and_close(path)
        paths[name] = str(path)

    return stats, paths

