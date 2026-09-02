from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters used by the Pearson-FC MLP experiment."""

    random_state: int = 42
    n_splits: int = 10
    k_features: int = 5000
    batch_size: int = 32
    max_epochs: int = 120
    patience: int = 14
    learning_rate: float = 2.2e-4
    weight_decay: float = 1e-3
    label_smoothing: float = 0.1
    dropout: float = 0.72
    class_weights: tuple[float, float] = (1.0, 1.35)
    input_noise_std: float = 0.025

