"""Create ASD/TD Pearson functional-connectivity maps for rois_ho."""

import argparse
import json
from pathlib import Path

from src.data import load_atlas, vectors_to_fc_matrices
from src.plots import save_pearson_maps


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
        default=Path("outputs/pearson_maps"),
    )
    parser.add_argument("--atlas-name", default="rois_ho")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y, subjects = load_atlas(args.data_path)
    matrices, n_rois = vectors_to_fc_matrices(X)
    stats, paths = save_pearson_maps(
        matrices,
        y,
        output_dir=args.output_dir,
        atlas_name=args.atlas_name,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pearson_map_statistics.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print(
        f"Atlas={args.atlas_name} | Subjects={len(subjects)} | "
        f"ROIs={n_rois} | Edges={X.shape[1]}"
    )
    print(json.dumps(stats, indent=2))
    print("Saved figures:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

