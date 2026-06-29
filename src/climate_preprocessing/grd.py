"""Binary GRD file discovery and parsing."""

from __future__ import annotations

import calendar
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import DTYPE, SUPPORTED_DTYPES, VariableConfig


YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


@dataclass(frozen=True)
class YearFile:
    """A source file associated with a detected calendar year."""

    year: int
    path: Path


def detect_year(path: Path) -> int:
    """Extract a four-digit year from a file name."""

    match = YEAR_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not detect year from file name: {path}")
    return int(match.group(0))


def discover_year_files(dataset_root: Path, variable: VariableConfig) -> list[YearFile]:
    """Discover and validate all GRD files for a variable."""

    source_dir = dataset_root / variable.source_folder
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source folder: {source_dir}")

    files = sorted(
        [path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == ".grd"]
    )
    if not files:
        raise FileNotFoundError(f"No GRD files found in {source_dir}")

    by_year: dict[int, Path] = {}
    for path in files:
        year = detect_year(path)
        if year in by_year:
            raise ValueError(
                f"Duplicate {variable.key} GRD files for {year}: {by_year[year]} and {path}"
            )
        by_year[year] = path

    return [YearFile(year=year, path=by_year[year]) for year in sorted(by_year)]


def expected_days_for_year(year: int) -> int:
    """Return the calendar day count for a year."""

    return 366 if calendar.isleap(year) else 365


def detect_dtype_and_record_count(
    path: Path, variable: VariableConfig
) -> tuple[np.dtype, int]:
    """Detect the supported binary dtype and daily record count from file size."""

    cells_per_record = variable.shape[0] * variable.shape[1]
    file_size = path.stat().st_size
    matches: list[tuple[np.dtype, int]] = []

    for dtype in SUPPORTED_DTYPES:
        np_dtype = np.dtype(dtype)
        record_bytes = cells_per_record * np_dtype.itemsize
        if file_size % record_bytes == 0:
            records = file_size // record_bytes
            if records > 0:
                matches.append((np_dtype, int(records)))

    if not matches:
        raise ValueError(
            f"Unsupported datatype or incorrect file size for {path}: {file_size} bytes "
            f"does not match supported dtypes {SUPPORTED_DTYPES} and grid {variable.shape}"
        )
    if len(matches) > 1:
        raise ValueError(f"Ambiguous datatype detection for {path}: {matches}")

    return matches[0]


def detect_record_count(path: Path, variable: VariableConfig) -> int:
    """Detect daily records from file size and configured grid shape."""

    return detect_dtype_and_record_count(path, variable)[1]


def read_grd_file(
    path: Path,
    variable: VariableConfig,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Read one yearly GRD file into a ``(days, height, width)`` float32 array."""

    dtype, records = detect_dtype_and_record_count(path, variable)
    year = detect_year(path)
    expected_days = expected_days_for_year(year)
    if records != expected_days:
        raise ValueError(
            f"Incorrect record count for {path}: detected {records}, "
            f"calendar year {year} requires {expected_days}"
        )

    if logger:
        logger.info("Reading file %s", path)

    if dtype != np.dtype(DTYPE):
        raise ValueError(f"Unsupported datatype for {path}: detected {dtype}")

    data = np.fromfile(path, dtype=dtype)
    expected_values = records * variable.shape[0] * variable.shape[1]
    if data.size != expected_values:
        raise ValueError(
            f"Corrupted GRD file {path}: expected {expected_values} values, got {data.size}"
        )

    array = data.reshape((records, *variable.shape)).astype(np.float32, copy=False)
    for missing_value in variable.missing_values:
        array[array == np.float32(missing_value)] = np.nan
    return array
