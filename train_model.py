"""Run 10-fold ASD-vs-TD classification using rois_ho Pearson-FC features."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import TrainConfig
from src.data import load_atlas
from src.metrics import compute_metrics, find_best_threshold
from src.plots import save_cv_plots
from src.training import predict_probabilities, seed_everything, train_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="NPZ file containing X, y and subjects.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/training"),
        help="Directory for metrics and figures.",
    )
    parser.add_argument("--atlas-name", default="rois_ho")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device. 'auto' uses CUDA when available.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    oof_probabilities = np.zeros(len(y), dtype=np.float32)
    exploratory_predictions = np.zeros(len(y), dtype=int)
    fold_rows: list[dict] = []
    histories: list[dict] = []
    best_epochs: list[int] = []
    best_validation_losses: list[float] = []
    exploratory_thresholds: list[float] = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(X, y), start=1
    ):
        print(f"\n{'-' * 80}\nFOLD {fold}/{config.n_splits}\n{'-' * 80}")
        X_train_raw = X[train_indices]
        X_validation_raw = X[validation_indices]
        y_train = y[train_indices]
        y_validation = y[validation_indices]

        # Fit feature selection and scaling only on the training partition.
        selector = SelectKBest(f_classif, k=min(config.k_features, X.shape[1]))
        X_train_selected = selector.fit_transform(X_train_raw, y_train)
        X_validation_selected = selector.transform(X_validation_raw)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_selected).astype(np.float32)
        X_validation = scaler.transform(X_validation_selected).astype(np.float32)

        model, history, best_epoch, best_validation_loss = train_fold(
            X_train,
            y_train,
            X_validation,
            y_validation,
            fold_number=fold,
            config=config,
            device=device,
        )
        probability = predict_probabilities(model, X_validation, device)
        oof_probabilities[validation_indices] = probability
        metrics_05 = compute_metrics(y_validation, probability, threshold=0.5)

        # Exploratory only: this threshold is selected and scored on the same
        # validation fold. Do not treat it as unbiased held-out performance.
        threshold, _ = find_best_threshold(y_validation, probability)
        metrics_threshold = compute_metrics(
            y_validation, probability, threshold=threshold
        )
        exploratory_predictions[validation_indices] = metrics_threshold[
            "predictions"
        ]

        histories.append(history)
        best_epochs.append(best_epoch)
        best_validation_losses.append(best_validation_loss)
        exploratory_thresholds.append(threshold)
        fold_rows.append(
            {
                "fold": fold,
                "accuracy_05": metrics_05["accuracy"],
                "balanced_accuracy_05": metrics_05["balanced_accuracy"],
                "f1_macro_05": metrics_05["f1_macro"],
                "best_threshold": threshold,
                "accuracy_thr": metrics_threshold["accuracy"],
                "balanced_accuracy_thr": metrics_threshold[
                    "balanced_accuracy"
                ],
                "precision_macro_thr": metrics_threshold["precision_macro"],
                "recall_macro_thr": metrics_threshold["recall_macro"],
                "f1_macro_thr": metrics_threshold["f1_macro"],
                "auc": metrics_threshold["auc"],
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
            }
        )
        print(
            f"Fold {fold} | 0.5 Acc={metrics_05['accuracy']:.4f} | "
            f"AUC={metrics_05['auc']:.4f} | Best epoch={best_epoch}"
        )

    primary_metrics = compute_metrics(y, oof_probabilities, threshold=0.5)
    exploratory = {
        "accuracy": accuracy_score(y, exploratory_predictions),
        "balanced_accuracy": balanced_accuracy_score(y, exploratory_predictions),
        "precision_macro": precision_score(
            y, exploratory_predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y, exploratory_predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y, exploratory_predictions, average="macro", zero_division=0
        ),
        "auc": roc_auc_score(y, oof_probabilities),
        "confusion_matrix": confusion_matrix(y, exploratory_predictions),
    }
    return {
        "primary_metrics": primary_metrics,
        "exploratory_metrics": exploratory,
        "oof_probabilities": oof_probabilities,
        "exploratory_predictions": exploratory_predictions,
        "fold_results": pd.DataFrame(fold_rows),
        "histories": histories,
        "best_epochs": best_epochs,
        "best_validation_losses": best_validation_losses,
        "exploratory_thresholds": exploratory_thresholds,
    }


def serializable_metrics(metrics: dict) -> dict:
    return {
        key: value.tolist()
        if isinstance(value, np.ndarray)
        else float(value)
        if isinstance(value, (np.floating, float))
        else value
        for key, value in metrics.items()
        if key != "predictions"
    }


def save_results(
    results: dict,
    y: np.ndarray,
    output_dir: Path,
    atlas_name: str,
    config: TrainConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = results["primary_metrics"]
    exploratory = results["exploratory_metrics"]

    results["fold_results"].to_csv(output_dir / "fold_results.csv", index=False)
    np.save(output_dir / "oof_probabilities.npy", results["oof_probabilities"])
    np.save(
        output_dir / "exploratory_threshold_predictions.npy",
        results["exploratory_predictions"],
    )
    report = classification_report(
        y,
        primary["predictions"],
        target_names=["TD", "ASD"],
        digits=4,
    )
    (output_dir / "classification_report_threshold_05.txt").write_text(
        report, encoding="utf-8"
    )

    summary = {
        "model": "rois_ho Pearson FC + Balanced-Loss MLP",
        "atlas": atlas_name,
        "label_mapping": {"0": "TD", "1": "ASD"},
        "primary_threshold_05": serializable_metrics(primary),
        "exploratory_fold_optimized_threshold": {
            **serializable_metrics(exploratory),
            "mean_threshold": float(np.mean(results["exploratory_thresholds"])),
            "thresholds": results["exploratory_thresholds"],
            "warning": (
                "Thresholds were selected and evaluated on the same validation "
                "folds; these scores are exploratory, not unbiased test estimates."
            ),
        },
        "training": {
            "n_splits": config.n_splits,
            "k_features": config.k_features,
            "batch_size": config.batch_size,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "label_smoothing": config.label_smoothing,
            "class_weights": list(config.class_weights),
            "input_noise_std": config.input_noise_std,
            "mean_best_epoch": float(np.mean(results["best_epochs"])),
            "mean_best_validation_loss": float(
                np.mean(results["best_validation_losses"])
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure_paths = save_cv_plots(
        histories=results["histories"],
        best_epochs=results["best_epochs"],
        fold_results=results["fold_results"],
        y_true=y,
        oof_probabilities=results["oof_probabilities"],
        confusion=primary["confusion_matrix"],
        auc=primary["auc"],
        output_dir=output_dir,
    )

    print(f"\nPrimary OOF results at threshold 0.50\n{report}")
    print(f"AUC: {primary['auc']:.4f}")
    print(f"Results saved to: {output_dir.resolve()}")
    for name, path in figure_paths.items():
        print(f"  {name}: {path}")


def main() -> None:
    args = parse_args()
    config = TrainConfig()
    device = resolve_device(args.device)
    seed_everything(config.random_state)

    X, y, subjects = load_atlas(args.data_path)
    print(
        f"Atlas={args.atlas_name} | X={X.shape} | Subjects={len(subjects)} | "
        f"ASD={int(y.sum())} | TD={int((y == 0).sum())} | Device={device}"
    )
    results = run_cross_validation(X, y, config, device)
    save_results(results, y, args.output_dir, args.atlas_name, config)


if __name__ == "__main__":
    main()
