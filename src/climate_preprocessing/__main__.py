"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DATASET_ROOT, LOG_DIR, METADATA_DIR, PROCESSED_DIR
from climate_preprocessing.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IMD climate preprocessing.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=METADATA_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--include-mask-channel", action="store_true")
    parser.add_argument("--drop-invalid-rainfall-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        dataset_root=args.dataset_root,
        processed_dir=args.processed_dir,
        metadata_dir=args.metadata_dir,
        log_dir=args.log_dir,
        include_mask_channel=args.include_mask_channel,
        drop_invalid_rainfall_csv=args.drop_invalid_rainfall_csv,
    )


if __name__ == "__main__":
    main()
