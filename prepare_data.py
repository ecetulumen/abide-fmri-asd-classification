"""Download ABIDE I PCP ROI time series and create Pearson-FC NPZ caches."""

import argparse
from pathlib import Path

from src.feature_extraction import EXPECTED_ROIS, build_common_feature_caches


DEFAULT_ATLASES = ("rois_aal", "rois_cc200", "rois_ho", "rois_tt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/abide_pcp"),
        help="Directory used by Nilearn for downloaded ABIDE PCP files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for aligned Pearson-FC NPZ caches.",
    )
    parser.add_argument(
        "--atlases",
        nargs="+",
        choices=tuple(EXPECTED_ROIS),
        default=list(DEFAULT_ATLASES),
        help="ROI derivatives that must be present in the common cohort.",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        help="Optional subject limit for a small trial download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from nilearn.datasets import fetch_abide_pcp
    except ImportError as error:
        raise RuntimeError(
            "Nilearn is required. Install the project dependencies with "
            "'pip install -r requirements.txt'."
        ) from error

    print("Downloading/loading ABIDE I PCP ROI time series...")
    dataset = fetch_abide_pcp(
        data_dir=str(args.download_dir),
        n_subjects=args.n_subjects,
        pipeline="cpac",
        band_pass_filtering=True,
        global_signal_regression=False,
        derivatives=args.atlases,
        quality_checked=True,
    )
    summary = build_common_feature_caches(
        dataset,
        output_dir=args.output_dir,
        atlases=args.atlases,
        pipeline="cpac",
    )

    print("\nPearson feature preparation completed:")
    print(summary.to_string(index=False))
    print(f"\nCaches saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
