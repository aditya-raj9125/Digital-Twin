"""Stacked tensor generation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    CHANNEL_ORDER,
    INCLUDE_MASK_CHANNEL,
    MASK_CHANNEL_LABEL,
    PROCESSED_DIR,
    REFERENCE_VARIABLE,
    TRAIN_END_YEAR,
    VALIDATION_YEAR,
    VARIABLES,
)
from climate_preprocessing.dates import dates_for_year
from climate_preprocessing.grd import YearFile, read_grd_file
from climate_preprocessing.mask import (
    apply_rainfall_mask,
    assert_valid_cells_match_mask,
    load_or_create_rainfall_mask,
)
from climate_preprocessing.regrid import interpolate_to_reference_grid


def _read_year_arrays(
    year: int,
    files_by_variable: dict[str, dict[int, YearFile]],
    processed_dir: Path,
    rainfall_mask: np.ndarray,
    logger: logging.Logger,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    reference = VARIABLES[REFERENCE_VARIABLE]
    for key in CHANNEL_ORDER:
        if year not in files_by_variable[key]:
            raise FileNotFoundError(f"Missing {key} GRD file for {year}")
        array = read_grd_file(files_by_variable[key][year].path, VARIABLES[key], logger)
        if key != REFERENCE_VARIABLE:
            array = interpolate_to_reference_grid(array, VARIABLES[key], reference, logger)
            array = apply_rainfall_mask(array, rainfall_mask, fill_valid_nans=True)
            assert_valid_cells_match_mask(array, rainfall_mask, key)
            interpolated_dir = processed_dir / "interpolated" / VARIABLES[key].output_folder
            interpolated_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                interpolated_dir / f"{year}.npz",
                data=array,
                latitudes=reference.latitudes,
                longitudes=reference.longitudes,
            )
        arrays[key] = array
    return arrays


def _stack_arrays(
    arrays: dict[str, np.ndarray],
    rainfall_mask: np.ndarray,
    include_mask_channel: bool,
) -> np.ndarray:
    reference_shape = arrays[REFERENCE_VARIABLE].shape
    for key, array in arrays.items():
        if array.shape != reference_shape:
            raise ValueError(
                f"Aligned shape mismatch for {key}: expected {reference_shape}, got {array.shape}"
            )
    channels = [arrays[key] for key in CHANNEL_ORDER]
    if include_mask_channel:
        mask_channel = np.broadcast_to(
            rainfall_mask.astype(np.float32),
            (reference_shape[0], *rainfall_mask.shape),
        )
        channels.append(mask_channel)
    return np.stack(channels, axis=1).astype(np.float32)


def _save_period_tensor(
    years: list[int],
    output_path: Path,
    files_by_variable: dict[str, dict[int, YearFile]],
    rainfall_mask: np.ndarray,
    logger: logging.Logger,
    include_mask_channel: bool,
) -> None:
    tensors: list[np.ndarray] = []
    date_labels: list[str] = []

    for year in tqdm(years, desc=f"Tensor {output_path.stem}", unit="year"):
        arrays = _read_year_arrays(
            year,
            files_by_variable,
            output_path.parent.parent,
            rainfall_mask,
            logger,
        )
        stacked = _stack_arrays(arrays, rainfall_mask, include_mask_channel)
        tensors.append(stacked)
        date_labels.extend(dates_for_year(year).strftime("%Y-%m-%d").tolist())

    tensor = np.concatenate(tensors, axis=0)
    channel_labels = [VARIABLES[key].tensor_label for key in CHANNEL_ORDER]
    if include_mask_channel:
        channel_labels.append(MASK_CHANNEL_LABEL)
    np.savez_compressed(
        output_path,
        tensor=tensor,
        dates=np.array(date_labels),
        channels=np.array(channel_labels),
        latitudes=VARIABLES[REFERENCE_VARIABLE].latitudes,
        longitudes=VARIABLES[REFERENCE_VARIABLE].longitudes,
        rainfall_mask=rainfall_mask.astype(np.uint8),
    )
    logger.info("Tensor saved: %s shape=%s", output_path, tensor.shape)


def create_stacked_tensors(
    files_by_variable: dict[str, dict[int, YearFile]],
    years: list[int],
    processed_dir: Path = PROCESSED_DIR,
    logger: logging.Logger | None = None,
    include_mask_channel: bool = INCLUDE_MASK_CHANNEL,
    rainfall_mask: np.ndarray | None = None,
) -> None:
    """Create train and validation stacked climate tensors."""

    if logger is None:
        logger = logging.getLogger("climate_preprocessing")

    output_dir = processed_dir / "stacked_tensor"
    output_dir.mkdir(parents=True, exist_ok=True)
    if rainfall_mask is None:
        rainfall_mask = load_or_create_rainfall_mask(
            files_by_variable[REFERENCE_VARIABLE],
            processed_dir,
            logger,
        )

    train_years = [year for year in sorted(years) if year <= TRAIN_END_YEAR]
    validation_years = [year for year in sorted(years) if year == VALIDATION_YEAR]
    if VALIDATION_YEAR in train_years:
        raise ValueError("Validation year 2025 must never be included in training")
    if not validation_years:
        raise ValueError(f"Missing validation year {VALIDATION_YEAR}")

    _save_period_tensor(
        train_years,
        output_dir / "train_tensor_2012_2024.npz",
        files_by_variable,
        rainfall_mask,
        logger,
        include_mask_channel,
    )
    _save_period_tensor(
        validation_years,
        output_dir / "validation_2025.npz",
        files_by_variable,
        rainfall_mask,
        logger,
        include_mask_channel,
    )


def load_tensor_dates(path: Path) -> pd.DatetimeIndex:
    """Load date labels from an NPZ tensor file."""

    with np.load(path) as data:
        return pd.to_datetime(data["dates"])
