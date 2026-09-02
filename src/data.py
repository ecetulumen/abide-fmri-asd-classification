from pathlib import Path

import numpy as np


def load_atlas(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Pearson-FC vectors, binary labels and subject identifiers."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        required = {"X", "y", "subjects"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Missing NPZ keys: {sorted(missing)}")

        X = data["X"].astype(np.float32)
        y = data["y"].astype(int)
        subjects = data["subjects"]

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; received {X.shape}")
    if len(X) != len(y) or len(y) != len(subjects):
        raise ValueError("X, y and subjects must contain the same number of rows.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Labels must use 0=TD and 1=ASD.")
    if len(np.unique(subjects)) != len(subjects):
        raise ValueError("Duplicate subject IDs detected; subject-level leakage is possible.")

    return X, y, subjects


def infer_n_rois(n_features: int) -> int:
    """Infer ROI count from n_features = n_rois * (n_rois - 1) / 2."""
    n_rois = int((1 + np.sqrt(1 + 8 * n_features)) / 2)
    if n_rois * (n_rois - 1) // 2 != n_features:
        raise ValueError(
            f"Feature count is not a valid upper triangle: {n_features}"
        )
    return n_rois


def vectors_to_fc_matrices(X: np.ndarray) -> tuple[np.ndarray, int]:
    """Convert upper-triangle FC vectors into symmetric correlation matrices."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; received {X.shape}")

    n_subjects, n_features = X.shape
    n_rois = infer_n_rois(n_features)
    matrices = np.zeros((n_subjects, n_rois, n_rois), dtype=np.float32)
    upper = np.triu_indices(n_rois, k=1)

    matrices[:, upper[0], upper[1]] = X
    matrices += matrices.transpose(0, 2, 1)
    diagonal = np.arange(n_rois)
    matrices[:, diagonal, diagonal] = 1.0
    return matrices, n_rois

