"""Build Pearson functional-connectivity features from ABIDE ROI time series."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_ROIS = {
    "rois_aal": 116,
    "rois_cc200": 200,
    "rois_dosenbach160": 160,
    "rois_ho": 111,
    "rois_tt": 97,
}


def _dataset_value(dataset: Any, key: str) -> Any:
    """Read a value from either a mapping or a scikit-learn Bunch."""
    if isinstance(dataset, Mapping):
        if key not in dataset:
            raise KeyError(f"Downloaded ABIDE data does not contain '{key}'.")
        return dataset[key]
    if not hasattr(dataset, key):
        raise KeyError(f"Downloaded ABIDE data does not contain '{key}'.")
    return getattr(dataset, key)


def prepare_phenotypic_table(phenotypic: Any) -> pd.DataFrame:
    """Validate subject IDs and convert ABIDE DX_GROUP to 0=TD, 1=ASD."""
    frame = pd.DataFrame(phenotypic).copy()
    required = {"SUB_ID", "DX_GROUP"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing phenotypic columns: {sorted(missing)}")

    frame = frame[["SUB_ID", "DX_GROUP"]].copy()
    frame["SUB_ID"] = pd.to_numeric(frame["SUB_ID"], errors="coerce")
    frame["DX_GROUP"] = pd.to_numeric(frame["DX_GROUP"], errors="coerce")
    frame = frame.dropna(subset=["SUB_ID", "DX_GROUP"])
    frame["SUB_ID"] = frame["SUB_ID"].astype(int)
    frame["DX_GROUP"] = frame["DX_GROUP"].astype(int)
    frame["label"] = frame["DX_GROUP"].map({1: 1, 2: 0})
    frame = frame.dropna(subset=["label"])
    frame["label"] = frame["label"].astype(int)

    if frame["SUB_ID"].duplicated().any():
        duplicates = frame.loc[frame["SUB_ID"].duplicated(), "SUB_ID"].tolist()
        raise ValueError(f"Duplicate subject IDs in phenotypic data: {duplicates[:10]}")
    return frame.reset_index(drop=True)


def load_roi_timeseries(
    file_path: str | Path,
    expected_n_rois: int | None = None,
) -> np.ndarray:
    """Load a PCP ROI time-series file and orient it as time points x ROIs."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"ROI time-series file not found: {path}")

    timeseries = np.loadtxt(path, comments="#", dtype=np.float32)
    if timeseries.ndim == 1:
        timeseries = timeseries.reshape(-1, 1)
    if timeseries.ndim != 2:
        raise ValueError(f"ROI time series must be 2-D; received {timeseries.shape}")

    if expected_n_rois is not None:
        if timeseries.shape[1] == expected_n_rois:
            pass
        elif timeseries.shape[0] == expected_n_rois:
            timeseries = timeseries.T
        else:
            raise ValueError(
                f"Expected {expected_n_rois} ROIs but received shape "
                f"{timeseries.shape} from {path}"
            )
    elif timeseries.shape[0] < timeseries.shape[1]:
        timeseries = timeseries.T

    if timeseries.shape[0] < 10 or timeseries.shape[1] < 2:
        raise ValueError(f"ROI time series is too small: {timeseries.shape}")
    return timeseries


def pearson_upper_triangle(timeseries: np.ndarray) -> np.ndarray:
    """Return unique undirected Pearson edges from a time points x ROIs array."""
    values = np.asarray(timeseries, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 10 or values.shape[1] < 2:
        raise ValueError(
            "timeseries must be a 2-D array with at least 10 time points and 2 ROIs"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(
        correlation, nan=0.0, posinf=0.0, neginf=0.0
    )
    correlation = np.clip(correlation, -1.0, 1.0)
    upper = np.triu_indices_from(correlation, k=1)
    return correlation[upper].astype(np.float32)


def extract_atlas_features(
    paths: Sequence[str | Path],
    phenotypic: pd.DataFrame,
    atlas_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract one Pearson-FC vector per valid subject for one atlas."""
    if len(paths) != len(phenotypic):
        raise ValueError(
            f"{atlas_name}: {len(paths)} derivative files do not match "
            f"{len(phenotypic)} phenotypic rows."
        )

    expected_n_rois = EXPECTED_ROIS.get(atlas_name)
    by_subject: dict[int, tuple[np.ndarray, int]] = {}
    skipped = 0

    for path_value, row in zip(paths, phenotypic.itertuples(index=False)):
        subject_id = int(row.SUB_ID)
        label = int(row.label)
        if path_value is None or str(path_value).strip() in {"", "no_filename"}:
            skipped += 1
            continue

        try:
            timeseries = load_roi_timeseries(path_value, expected_n_rois)
            features = pearson_upper_triangle(timeseries)
        except (OSError, TypeError, ValueError):
            skipped += 1
            continue

        if subject_id in by_subject:
            raise ValueError(f"{atlas_name}: duplicate derivative for {subject_id}")
        by_subject[subject_id] = (features, label)

    if not by_subject:
        raise RuntimeError(f"{atlas_name}: no valid ROI time-series files were found.")

    subjects = np.asarray(sorted(by_subject), dtype=int)
    X = np.vstack([by_subject[sid][0] for sid in subjects]).astype(np.float32)
    y = np.asarray([by_subject[sid][1] for sid in subjects], dtype=int)
    return X, y, subjects, skipped


def build_common_feature_caches(
    dataset: Any,
    output_dir: str | Path,
    atlases: Sequence[str],
    pipeline: str = "cpac",
) -> pd.DataFrame:
    """Create aligned NPZ caches with X, y and subjects for all atlases."""
    if not atlases:
        raise ValueError("At least one atlas must be requested.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    phenotypic = prepare_phenotypic_table(_dataset_value(dataset, "phenotypic"))

    representations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for atlas_name in atlases:
        paths = _dataset_value(dataset, atlas_name)
        representations[atlas_name] = extract_atlas_features(
            paths, phenotypic, atlas_name
        )

    subject_sets = [set(item[2].tolist()) for item in representations.values()]
    common_subjects = np.asarray(sorted(set.intersection(*subject_sets)), dtype=int)
    if common_subjects.size == 0:
        raise RuntimeError("No subjects have valid files across every requested atlas.")

    summary_rows = []
    reference_labels: np.ndarray | None = None
    for atlas_name, (X, y, subjects, skipped) in representations.items():
        index_by_subject = {subject_id: index for index, subject_id in enumerate(subjects)}
        indices = np.asarray(
            [index_by_subject[subject_id] for subject_id in common_subjects], dtype=int
        )
        X_common = X[indices]
        y_common = y[indices]

        if reference_labels is None:
            reference_labels = y_common
        elif not np.array_equal(reference_labels, y_common):
            raise ValueError(f"Label alignment differs for atlas {atlas_name}.")

        cache_path = output / (
            f"{pipeline}_{atlas_name}_FILTERED_corr_features_FIXED.npz"
        )
        np.savez_compressed(
            cache_path,
            X=X_common,
            y=y_common,
            subjects=common_subjects,
        )
        summary_rows.append(
            {
                "atlas": atlas_name,
                "n_subjects_before_common": int(len(subjects)),
                "n_subjects_common": int(len(common_subjects)),
                "n_td_common": int((y_common == 0).sum()),
                "n_asd_common": int((y_common == 1).sum()),
                "feature_dim": int(X_common.shape[1]),
                "skipped_files": int(skipped),
                "cache_path": str(cache_path),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "feature_summary.csv", index=False)
    metadata = {
        "pipeline": pipeline,
        "atlases": list(atlases),
        "n_common_subjects": int(len(common_subjects)),
        "label_mapping": {"0": "TD", "1": "ASD"},
        "feature_order": "numpy.triu_indices(n_rois, k=1)",
    }
    (output / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return summary
