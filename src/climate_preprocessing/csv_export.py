"""CSV export and concatenation utilities."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import PROCESSED_DIR, VariableConfig
from climate_preprocessing.dates import dates_for_year
from climate_preprocessing.grd import YearFile, read_grd_file


def write_year_csv(
    array: np.ndarray,
    year: int,
    variable: VariableConfig,
    output_path: Path,
    chunk_days: int = 16,
    drop_invalid: bool = False,
    valid_mask: np.ndarray | None = None,
) -> None:
    """Write one ``(days, height, width)`` array as long-format CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dates = dates_for_year(year)
    if len(dates) != array.shape[0]:
        raise ValueError(
            f"Date count mismatch for {year}: {len(dates)} dates vs {array.shape[0]} records"
        )

    lat_grid, lon_grid = np.meshgrid(
        variable.latitudes, variable.longitudes, indexing="ij"
    )
    flat_lats = lat_grid.ravel()
    flat_lons = lon_grid.ravel()
    cells = flat_lats.size
    keep_cells = np.ones(cells, dtype=bool)
    if drop_invalid:
        if valid_mask is None:
            raise ValueError("drop_invalid=True requires a rainfall valid_mask")
        if valid_mask.shape != variable.shape:
            raise ValueError(
                f"valid_mask shape {valid_mask.shape} does not match {variable.shape}"
            )
        keep_cells = valid_mask.ravel()
        flat_lats = flat_lats[keep_cells]
        flat_lons = flat_lons[keep_cells]
        cells = flat_lats.size

    first_chunk = True
    for start in range(0, array.shape[0], chunk_days):
        end = min(start + chunk_days, array.shape[0])
        chunk = array[start:end].reshape(end - start, -1)
        values = chunk[:, keep_cells].reshape((end - start) * cells)
        date_values = np.repeat(dates[start:end].strftime("%Y-%m-%d").to_numpy(), cells)
        frame = pd.DataFrame(
            {
                "Date": date_values,
                "Latitude": np.tile(flat_lats, end - start),
                "Longitude": np.tile(flat_lons, end - start),
                "Value": values,
            }
        )
        frame.to_csv(
            output_path,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False


def convert_variable_years_to_csv(
    year_files: list[YearFile],
    variable: VariableConfig,
    dataset_root: Path,
    processed_dir: Path,
    logger: logging.Logger,
    drop_invalid: bool = False,
    valid_mask: np.ndarray | None = None,
) -> None:
    """Convert all yearly GRD files for one variable into yearly CSV files."""

    del dataset_root
    logger.info("Started conversion for %s", variable.key)
    output_dir = processed_dir / variable.output_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    for year_file in tqdm(year_files, desc=f"CSV {variable.key}", unit="year"):
        output_path = output_dir / f"{year_file.year}.csv"
        array = read_grd_file(year_file.path, variable, logger)
        write_year_csv(
            array,
            year_file.year,
            variable,
            output_path,
            drop_invalid=drop_invalid,
            valid_mask=valid_mask,
        )
        logger.info("Completed year %s for %s", year_file.year, variable.key)


def combine_yearly_csvs(
    variable: VariableConfig,
    years: list[int],
    processed_dir: Path = PROCESSED_DIR,
    logger: logging.Logger | None = None,
) -> Path:
    """Concatenate yearly CSVs into one chronological combined CSV."""

    input_dir = processed_dir / variable.output_folder
    output_dir = processed_dir / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{variable.output_folder}_all.csv"

    with output_path.open("wb") as destination:
        wrote_header = False
        for year in tqdm(sorted(years), desc=f"Combine {variable.key}", unit="year"):
            path = input_dir / f"{year}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing yearly CSV: {path}")
            with path.open("rb") as source:
                if wrote_header:
                    source.readline()
                shutil.copyfileobj(source, destination, length=1024 * 1024 * 16)
            wrote_header = True

    if logger:
        logger.info("Combined CSV saved: %s", output_path)
    return output_path
