import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig
from .models import BalancedLossMLP


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    fold_number: int,
    config: TrainConfig,
    device: torch.device,
) -> tuple[BalancedLossMLP, dict[str, list[float]], int, float]:
    """Train one cross-validation fold and restore its best checkpoint."""
    train_features = torch.as_tensor(X_train, dtype=torch.float32)
    train_labels = torch.as_tensor(y_train, dtype=torch.long)
    validation_features = torch.as_tensor(
        X_validation, dtype=torch.float32, device=device
    )
    validation_labels = torch.as_tensor(
        y_validation, dtype=torch.long, device=device
    )

    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = BalancedLossMLP(
        input_dim=X_train.shape[1],
        dropout=config.dropout,
        noise_std=config.input_noise_std,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(config.class_weights, dtype=torch.float32, device=device),
        label_smoothing=config.label_smoothing,
    )

    best_score = -np.inf
    best_state = None
    best_epoch = 0
    best_validation_loss = np.inf
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
        "validation_balanced_accuracy": [],
        "validation_auc": [],
    }

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        epoch_true: list[int] = []
        epoch_predicted: list[int] = []

        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            batch_losses.append(loss.item())
            epoch_predicted.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
            epoch_true.extend(labels.detach().cpu().tolist())

        scheduler.step()
        train_loss = float(np.mean(batch_losses))
        train_accuracy = accuracy_score(epoch_true, epoch_predicted)

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_features)
            validation_loss = criterion(
                validation_logits, validation_labels
            ).item()
            validation_probability = (
                torch.softmax(validation_logits, dim=1)[:, 1].cpu().numpy()
            )

        validation_prediction = (validation_probability >= 0.5).astype(int)
        validation_accuracy = accuracy_score(y_validation, validation_prediction)
        validation_balanced_accuracy = balanced_accuracy_score(
            y_validation, validation_prediction
        )
        validation_auc = roc_auc_score(y_validation, validation_probability)

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)
        history["validation_balanced_accuracy"].append(validation_balanced_accuracy)
        history["validation_auc"].append(validation_auc)

        selection_score = (
            0.42 * validation_accuracy
            + 0.24 * validation_balanced_accuracy
            + 0.30 * validation_auc
            - 0.04 * validation_loss
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_validation_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Fold {fold_number} | Epoch {epoch:03d}/{config.max_epochs} | "
                f"Train Loss={train_loss:.4f} | Val Loss={validation_loss:.4f} | "
                f"Train Acc={train_accuracy:.4f} | Val Acc={validation_accuracy:.4f} | "
                f"Val BalAcc={validation_balanced_accuracy:.4f} | "
                f"Val AUC={validation_auc:.4f} | Score={selection_score:.4f}"
            )

        if epochs_without_improvement >= config.patience:
            print(
                f"Fold {fold_number} | Early stopping at epoch {epoch} | "
                f"Best epoch={best_epoch} | "
                f"Best Val Loss={best_validation_loss:.4f}"
            )
            break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_validation_loss


def predict_probabilities(
    model: BalancedLossMLP,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        features = torch.as_tensor(X, dtype=torch.float32, device=device)
        logits = model(features)
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

