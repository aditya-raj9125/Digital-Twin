"""Rainfall spatial mask generation, caching, and application."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from config import (
    EXPECTED_RAINFALL_INVALID_CELLS,
    EXPECTED_RAINFALL_VALID_CELLS,
    PROCESSED_DIR,
    REFERENCE_VARIABLE,
    VARIABLES,
)
from climate_preprocessing.grd import YearFile, read_grd_file


def metadata_dir_for(processed_dir: Path = PROCESSED_DIR) -> Path:
    """Return the processed metadata directory."""

    return Path(processed_dir) / "metadata"


def mask_path(processed_dir: Path = PROCESSED_DIR) -> Path:
    """Return the canonical rainfall mask cache path."""

    return metadata_dir_for(processed_dir) / "rainfall_mask.npy"


def _representative_valid_mask(year_file: YearFile) -> np.ndarray:
    rainfall = VARIABLES[REFERENCE_VARIABLE]
    first_day = read_grd_file(year_file.path, rainfall)[0]
    return np.isfinite(first_day)


def verify_and_cache_rainfall_mask(
    rainfall_files: dict[int, YearFile],
    processed_dir: Path = PROCESSED_DIR,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Verify the permanent rainfall mask across years, then cache it."""

    if not rainfall_files:
        raise ValueError("No rainfall files supplied for mask verification")

    metadata_dir = metadata_dir_for(processed_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    sorted_years = sorted(rainfall_files)
    reference_mask = _representative_valid_mask(rainfall_files[sorted_years[0]])

    for year in sorted_years[1:]:
        candidate = _representative_valid_mask(rainfall_files[year])
        if not np.array_equal(reference_mask, candidate):
            raise ValueError(
                f"Rainfall mask differs between {sorted_years[0]} and {year}; "
                "stopping preprocessing"
            )

    valid_cells = int(reference_mask.sum())
    invalid_cells = int(reference_mask.size - valid_cells)
    if valid_cells != EXPECTED_RAINFALL_VALID_CELLS:
        raise ValueError(
            f"Rainfall valid cell count mismatch: expected "
            f"{EXPECTED_RAINFALL_VALID_CELLS}, got {valid_cells}"
        )
    if invalid_cells != EXPECTED_RAINFALL_INVALID_CELLS:
        raise ValueError(
            f"Rainfall invalid cell count mismatch: expected "
            f"{EXPECTED_RAINFALL_INVALID_CELLS}, got {invalid_cells}"
        )

    np.save(mask_path(processed_dir), reference_mask)
    write_rainfall_mask_artifacts(reference_mask, processed_dir)
    if logger:
        logger.info(
            "Rainfall mask verified and cached: valid=%s invalid=%s",
            valid_cells,
            invalid_cells,
        )
    return reference_mask


def load_or_create_rainfall_mask(
    rainfall_files: dict[int, YearFile],
    processed_dir: Path = PROCESSED_DIR,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Load cached rainfall mask, or verify and create it once."""

    path = mask_path(processed_dir)
    if path.exists():
        mask = np.load(path).astype(bool)
        expected_shape = VARIABLES[REFERENCE_VARIABLE].shape
        if mask.shape != expected_shape:
            raise ValueError(
                f"Cached rainfall mask has shape {mask.shape}, expected {expected_shape}"
            )
        if logger:
            logger.info("Loaded cached rainfall mask: %s", path)
        return mask
    return verify_and_cache_rainfall_mask(rainfall_files, processed_dir, logger)


def write_rainfall_mask_artifacts(
    valid_mask: np.ndarray,
    processed_dir: Path = PROCESSED_DIR,
) -> None:
    """Write rainfall mask CSV, NPY, and PNG metadata artifacts."""

    rainfall = VARIABLES[REFERENCE_VARIABLE]
    metadata_dir = metadata_dir_for(processed_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    lat_grid, lon_grid = np.meshgrid(
        rainfall.latitudes, rainfall.longitudes, indexing="ij"
    )
    pd.DataFrame(
        {
            "Latitude": lat_grid.ravel(),
            "Longitude": lon_grid.ravel(),
            "Valid": valid_mask.astype(np.uint8).ravel(),
        }
    ).to_csv(metadata_dir / "rainfall_mask.csv", index=False)
    np.save(metadata_dir / "rainfall_mask.npy", valid_mask.astype(bool))
    save_mask_plot(valid_mask, metadata_dir / "rainfall_mask.png")


def save_mask_plot(valid_mask: np.ndarray, output_path: Path) -> None:
    """Save a visual map of valid and invalid rainfall regions."""

    rainfall = VARIABLES[REFERENCE_VARIABLE]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.imshow(
        valid_mask,
        origin="lower",
        cmap="Greens",
        extent=[
            float(rainfall.longitudes[0]),
            float(rainfall.longitudes[-1]),
            float(rainfall.latitudes[0]),
            float(rainfall.latitudes[-1]),
        ],
        aspect="auto",
    )
    plt.title("Rainfall Valid Spatial Mask")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def verify_mask_identical_across_days(
    rainfall_files: dict[int, YearFile],
    valid_mask: np.ndarray,
) -> None:
    """Verify every daily rainfall mask in every file matches the cached mask."""

    rainfall = VARIABLES[REFERENCE_VARIABLE]
    for year in sorted(rainfall_files):
        data = read_grd_file(rainfall_files[year].path, rainfall)
        daily_masks = np.isfinite(data)
        if not np.all(daily_masks == valid_mask):
            raise ValueError(f"Rainfall mask differs across days in {year}")


def fill_valid_mask_nans_nearest(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Fill NaNs inside the valid mask using nearest finite cells per day."""

    result = data.astype(np.float32, copy=True)
    for day_index in range(result.shape[0]):
        day = result[day_index]
        fill_area = valid_mask & ~np.isfinite(day)
        if not fill_area.any():
            continue
        finite = np.isfinite(day)
        if not finite.any():
            raise ValueError(f"No finite values available to fill day {day_index}")
        nearest = ndimage.distance_transform_edt(~finite, return_distances=False, return_indices=True)
        day[fill_area] = day[tuple(index[fill_area] for index in nearest)]
    return result


def apply_rainfall_mask(
    data: np.ndarray,
    valid_mask: np.ndarray,
    fill_valid_nans: bool = False,
) -> np.ndarray:
    """Apply the permanent rainfall domain mask to a gridded time series."""

    if data.shape[1:] != valid_mask.shape:
        raise ValueError(
            f"Mask shape mismatch: data grid {data.shape[1:]}, mask {valid_mask.shape}"
        )
    result = data.astype(np.float32, copy=True)
    if fill_valid_nans:
        result = fill_valid_mask_nans_nearest(result, valid_mask)
    result[:, ~valid_mask] = np.nan
    return result


def assert_valid_cells_match_mask(
    data: np.ndarray,
    valid_mask: np.ndarray,
    label: str,
) -> None:
    """Verify every day has finite cells exactly where rainfall mask is valid."""

    finite = np.isfinite(data)
    expected = np.broadcast_to(valid_mask, finite.shape)
    if not np.array_equal(finite, expected):
        raise ValueError(f"{label} valid cells do not match rainfall mask after masking")
