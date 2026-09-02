# ABIDE ASD Classification with `rois_ho` Functional Connectivity

This repository contains an ASD-versus-TD classification pipeline based on
resting-state fMRI functional-connectivity features from the ABIDE dataset.
The final experiment uses Pearson connectivity from the Harvard-Oxford
(`rois_ho`) atlas and a class-weighted multilayer perceptron (MLP).

## Dataset acquisition

The ROI time series were obtained from **ABIDE I** through the
[ABIDE Preprocessed Connectomes Project (PCP)](http://preprocessed-connectomes-project.org/abide/)
using Nilearn's
[`fetch_abide_pcp`](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_abide_pcp.html)
function.

The original multi-atlas download used the C-PAC preprocessing pipeline,
0.01-0.1 Hz band-pass filtering, no global signal regression and the available
quality-control filter:

```python
from nilearn.datasets import fetch_abide_pcp

abide = fetch_abide_pcp(
    data_dir="data/abide_pcp",
    pipeline="cpac",
    band_pass_filtering=True,
    global_signal_regression=False,
    derivatives=["rois_aal", "rois_cc200", "rois_ho", "rois_tt"],
    quality_checked=True,
)
```

Only subjects with valid ROI time-series files across all four atlases were
retained in the common cohort. This produced **846 subjects: 391 ASD and 455
TD**. The phenotype field `DX_GROUP` was converted from the original ABIDE
coding (`1=ASD`, `2=control`) to the model labels used here (`1=ASD`,
`0=TD`).

For each subject, Pearson correlations were calculated between every pair of
ROI time series. For `rois_ho`, 111 ROIs produce 6,105 unique undirected
edges:

```python
import numpy as np

correlation_matrix = np.corrcoef(roi_time_series, rowvar=False)
upper_triangle = np.triu_indices(111, k=1)
features = correlation_matrix[upper_triangle]
```

The prepared feature cache is stored as an NPZ archive containing:

- `X`: shape `(846, 6105)`, Pearson-FC upper-triangle vectors
- `y`: shape `(846,)`, where `0=TD` and `1=ASD`
- `subjects`: shape `(846,)`, unique subject identifiers

The ABIDE data and generated NPZ cache are not included in this repository.

## Analysis pipeline

1. Load the prepared `rois_ho` Pearson-FC feature vectors.
2. Apply ANOVA F-score feature selection inside each training fold.
3. Standardize features using statistics learned from the training fold only.
4. Train a class-weighted MLP with Gaussian input noise, label smoothing,
   dropout, AdamW and early stopping.
5. Evaluate with stratified 10-fold cross-validation.
6. Save accuracy, balanced accuracy, macro F1, ROC-AUC, confusion matrix,
   training curves and fold-level results.
7. Reconstruct group-level FC matrices and visualize ASD, TD and ASD-TD
   connectivity patterns.

## Repository structure

```text
abide-fmri-asd-classification/
├── src/
│   ├── __init__.py
│   ├── config.py          # Hyperparameters
│   ├── data.py            # NPZ loading and FC matrix reconstruction
│   ├── metrics.py         # Evaluation and history utilities
│   ├── models.py          # PyTorch MLP
│   ├── plots.py           # Training plots and Pearson FC maps
│   └── training.py        # One-fold training and inference
├── results/
│   ├── accuracy_curve.png
│   ├── conf_matrix.png
│   ├── conf_matrix_balanced.png
│   ├── fold_performance.png
│   └── roc_curve.png
├── train_model.py         # 10-fold training entry point
├── visualize_fc.py        # Pearson FC map entry point
├── requirements.txt
└── .gitignore
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/ecetulumen/abide-fmri-asd-classification.git
cd abide-fmri-asd-classification

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Model training

Run the following command from the repository root after preparing the NPZ
feature cache:

```bash
python train_model.py \
  --data-path data/processed/cpac_rois_ho_FILTERED_corr_features_FIXED.npz \
  --output-dir outputs/training \
  --device auto
```

Use `--device cuda` to require a CUDA GPU, `--device cpu` for CPU, or leave
the default `--device auto`.

## Pearson FC map visualization

The map-generation code is included in three parts:

- `visualize_fc.py` loads the NPZ file and starts the visualization workflow.
- `src/plots.py` contains `save_pearson_maps()`, which creates the ASD mean,
  TD mean, ASD-TD difference and edge-distribution plots.
- `src/data.py` contains `vectors_to_fc_matrices()`, which reconstructs each
  symmetric 111 x 111 FC matrix from its 6,105 upper-triangle features.

Run it with:

```bash
python visualize_fc.py \
  --data-path data/processed/cpac_rois_ho_FILTERED_corr_features_FIXED.npz \
  --output-dir outputs/pearson_maps
```

The script saves a four-panel overview together with separate ASD, TD and
ASD-TD matrix figures and a JSON file containing group-level summary values.

## Results

Selected outputs from the final experiment are stored in the `results/`
directory and displayed below.

| ROC curve | Confusion matrix |
| --- | --- |
| ![ROC curve](results/roc_curve.png) | ![Confusion matrix](results/conf_matrix.png) |

| Training accuracy | Fold-level performance |
| --- | --- |
| ![Training and validation accuracy](results/accuracy_curve.png) | ![Fold-level performance](results/fold_performance.png) |

### Balanced-loss confusion matrix

![Balanced-loss confusion matrix](results/conf_matrix_balanced.png)

### Pearson functional-connectivity maps

![ASD and TD Pearson functional-connectivity maps](results/pearson_fc_overview.png)

The group-mean edge distributions largely overlap (ASD mean Pearson
`r=0.294`; TD mean Pearson `r=0.299`). The ASD-TD matrix is presented as a
descriptive group comparison rather than a statistical significance map.

## Evaluation note

The primary summary uses out-of-fold probabilities with the fixed threshold
`0.50`. The code retains the original fold-wise threshold search as an
explicitly exploratory result because each threshold is selected and scored on
the same validation fold. A publication-level estimate based on an optimized
threshold should use nested cross-validation and keep the outer test fold
untouched.

Feature selection and standardization are fitted only on the training portion
of each fold. Duplicate subject identifiers cause an error to reduce the risk
of subject-level leakage.
