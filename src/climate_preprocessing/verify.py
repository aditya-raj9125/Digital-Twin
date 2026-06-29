"""Pipeline verification and diagnostic plots."""

from __future__ import annotations

import calendar
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CHANNEL_ORDER,
    EXPECTED_RAINFALL_INVALID_CELLS,
    EXPECTED_RAINFALL_VALID_CELLS,
    METADATA_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    REFERENCE_VARIABLE,
    VALIDATION_YEAR,
    VARIABLES,
    VariableConfig,
)
from climate_preprocessing.grd import YearFile, detect_record_count
from climate_preprocessing.mask import (
    assert_valid_cells_match_mask,
    save_mask_plot,
    verify_mask_identical_across_days,
)


def _count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    return max(line_count - 1, 0)


def verify_files_and_csvs(
    files_by_variable: dict[str, dict[int, YearFile]],
    years: list[int],
    processed_dir: Path,
    logger: logging.Logger,
) -> None:
    """Verify file counts, leap years, and generated CSV row counts."""

    expected_file_count = len(years)
    for key, variable in VARIABLES.items():
        if len(files_by_variable[key]) != expected_file_count:
            raise ValueError(f"File count mismatch for {key}")
        for year in years:
            year_file = files_by_variable[key].get(year)
            if year_file is None:
                raise FileNotFoundError(f"Missing {key} file for {year}")
            records = detect_record_count(year_file.path, variable)
            expected_records = 366 if calendar.isleap(year) else 365
            if records != expected_records:
                raise ValueError(f"Leap-year record mismatch for {year_file.path}")

            csv_path = processed_dir / variable.output_folder / f"{year}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing CSV output: {csv_path}")
            expected_rows = expected_records * variable.shape[0] * variable.shape[1]
            actual_rows = _count_csv_rows(csv_path)
            if actual_rows != expected_rows:
                raise ValueError(
                    f"CSV row mismatch for {csv_path}: expected {expected_rows}, got {actual_rows}"
                )
    logger.info("File count, leap year, and CSV row verification successful")


def verify_rainfall_mask(
    files_by_variable: dict[str, dict[int, YearFile]],
    rainfall_mask: np.ndarray,
    processed_dir: Path,
    logger: logging.Logger,
) -> None:
    """Verify permanent rainfall mask and write report artifacts."""

    verify_mask_identical_across_days(files_by_variable[REFERENCE_VARIABLE], rainfall_mask)
    valid_cells = int(rainfall_mask.sum())
    invalid_cells = int(rainfall_mask.size - valid_cells)
    if valid_cells != EXPECTED_RAINFALL_VALID_CELLS:
        raise ValueError(f"Rainfall valid cell mismatch: {valid_cells}")
    if invalid_cells != EXPECTED_RAINFALL_INVALID_CELLS:
        raise ValueError(f"Rainfall invalid cell mismatch: {invalid_cells}")

    rainfall = VARIABLES[REFERENCE_VARIABLE]
    rows = np.where(rainfall_mask.any(axis=1))[0]
    cols = np.where(rainfall_mask.any(axis=0))[0]
    report = [
        "Rainfall Spatial Mask Verification",
        f"Valid cells: {valid_cells}",
        f"Invalid cells: {invalid_cells}",
        f"Percentage valid: {valid_cells / rainfall_mask.size * 100:.4f}",
        f"Percentage invalid: {invalid_cells / rainfall_mask.size * 100:.4f}",
        f"Latitude extent: {float(rainfall.latitudes[rows[0]])} to {float(rainfall.latitudes[rows[-1]])}",
        f"Longitude extent: {float(rainfall.longitudes[cols[0]])} to {float(rainfall.longitudes[cols[-1]])}",
        "Mask identical across years: yes",
        "Mask identical across days: yes",
    ]
    plots_dir = processed_dir / "verification_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "verification_report.txt").write_text("\n".join(report) + "\n")
    save_mask_plot(rainfall_mask, plots_dir / "rainfall_valid_mask.png")
    logger.info("Rainfall mask verification successful")


def verify_tensors(
    processed_dir: Path,
    logger: logging.Logger,
    rainfall_mask: np.ndarray,
) -> None:
    """Verify tensor dimensions, date separation, and NaN counts."""

    tensor_dir = processed_dir / "stacked_tensor"
    train_path = tensor_dir / "train_tensor_2012_2024.npz"
    validation_path = tensor_dir / "validation_2025.npz"
    reference = VARIABLES[REFERENCE_VARIABLE]

    for path in (train_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing tensor: {path}")
        with np.load(path) as data:
            tensor = data["tensor"]
            dates = pd.to_datetime(data["dates"])
            channels = data["channels"].tolist()
            if tensor.ndim != 4:
                raise ValueError(f"Tensor must be 4D: {path} shape={tensor.shape}")
            if tensor.shape[2:] != reference.shape:
                raise ValueError(f"Tensor dimensions incorrect: {path} shape={tensor.shape}")
            if tensor.shape[1] not in (len(CHANNEL_ORDER), len(CHANNEL_ORDER) + 1):
                raise ValueError(f"Unexpected channel count: {path} shape={tensor.shape}")
            if len(dates) != tensor.shape[0]:
                raise ValueError(f"Tensor date count mismatch: {path}")
            if "train" in path.name and (dates.year == VALIDATION_YEAR).any():
                raise ValueError("Training tensor includes 2025")
            for channel_index, channel_key in enumerate(CHANNEL_ORDER):
                assert_valid_cells_match_mask(
                    tensor[:, channel_index],
                    rainfall_mask,
                    f"{path.name}:{channel_key}",
                )
            if tensor.shape[1] == len(CHANNEL_ORDER) + 1:
                mask_channel = tensor[:, 3]
                expected_mask = np.broadcast_to(rainfall_mask, mask_channel.shape)
                if not np.array_equal(mask_channel, expected_mask.astype(np.float32)):
                    raise ValueError(f"Mask channel mismatch in {path}")
            nan_counts = np.isnan(tensor).sum(axis=(0, 2, 3))
            logger.info("Channels for %s: %s", path.name, channels)
            logger.info("NaN counts for %s: %s", path.name, nan_counts.tolist())
    logger.info("Tensor verification successful")


def create_random_visualizations(
    processed_dir: Path,
    logger: logging.Logger,
) -> None:
    """Create random rainfall, Tmax, and Tmin verification maps."""

    output_dir = processed_dir / "verification_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = processed_dir / "stacked_tensor" / "validation_2025.npz"
    rng = np.random.default_rng(RANDOM_SEED)

    with np.load(tensor_path) as data:
        tensor = data["tensor"]
        dates = data["dates"]
        day_index = int(rng.integers(0, tensor.shape[0]))
        for channel_key in CHANNEL_ORDER:
            variable = VARIABLES[channel_key]
            channel = variable.channel_index
            plt.figure(figsize=(9, 6))
            plt.imshow(tensor[day_index, channel], origin="lower", aspect="auto")
            plt.colorbar(label=variable.tensor_label)
            plt.title(f"{variable.tensor_label} {dates[day_index]}")
            plt.xlabel("Longitude index")
            plt.ylabel("Latitude index")
            output_path = output_dir / f"random_{channel_key}_map.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            logger.info("Verification image saved: %s", output_path)


def verify_pipeline(
    files_by_variable: dict[str, dict[int, YearFile]],
    years: list[int],
    processed_dir: Path = PROCESSED_DIR,
    metadata_dir: Path = METADATA_DIR,
    logger: logging.Logger | None = None,
    rainfall_mask: np.ndarray | None = None,
) -> None:
    """Run all verification checks."""

    del metadata_dir
    if logger is None:
        logger = logging.getLogger("climate_preprocessing")
    if rainfall_mask is None:
        mask_path = processed_dir / "metadata" / "rainfall_mask.npy"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing rainfall mask: {mask_path}")
        rainfall_mask = np.load(mask_path).astype(bool)
    verify_rainfall_mask(files_by_variable, rainfall_mask, processed_dir, logger)
    verify_files_and_csvs(files_by_variable, years, processed_dir, logger)
    verify_tensors(processed_dir, logger, rainfall_mask)
    create_random_visualizations(processed_dir, logger)
    logger.info("Verification successful")
