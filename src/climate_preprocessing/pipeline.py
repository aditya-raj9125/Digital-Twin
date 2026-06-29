"""End-to-end orchestration for the preprocessing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from config import (
    DATASET_ROOT,
    INCLUDE_MASK_CHANNEL,
    LOG_DIR,
    METADATA_DIR,
    PROCESSED_DIR,
    REFERENCE_VARIABLE,
    VARIABLES,
)
from climate_preprocessing.csv_export import (
    combine_yearly_csvs,
    convert_variable_years_to_csv,
)
from climate_preprocessing.grd import YearFile, discover_year_files
from climate_preprocessing.logging_utils import configure_logging
from climate_preprocessing.mask import verify_and_cache_rainfall_mask
from climate_preprocessing.metadata import write_metadata
from climate_preprocessing.tensor import create_stacked_tensors
from climate_preprocessing.verify import verify_pipeline


def _discover_all(dataset_root: Path) -> tuple[dict[str, dict[int, YearFile]], list[int]]:
    files_by_variable: dict[str, dict[int, YearFile]] = {}
    common_years: set[int] | None = None

    for key, variable in VARIABLES.items():
        year_files = discover_year_files(dataset_root, variable)
        files_by_variable[key] = {item.year: item for item in year_files}
        variable_years = set(files_by_variable[key])
        common_years = variable_years if common_years is None else common_years & variable_years

    if not common_years:
        raise ValueError("No common years found across all variables")

    sorted_common_years = sorted(common_years)
    for key, files in files_by_variable.items():
        extra_years = sorted(set(files) - set(sorted_common_years))
        if extra_years:
            raise ValueError(f"{key} has years not present in every variable: {extra_years}")

    return files_by_variable, sorted_common_years


def run_pipeline(
    dataset_root: Path = DATASET_ROOT,
    processed_dir: Path = PROCESSED_DIR,
    metadata_dir: Path = METADATA_DIR,
    log_dir: Path = LOG_DIR,
    include_mask_channel: bool = INCLUDE_MASK_CHANNEL,
    drop_invalid_rainfall_csv: bool = False,
) -> None:
    """Run the complete preprocessing pipeline."""

    logger = configure_logging(log_dir)
    logger.info("Pipeline started")
    dataset_root = Path(dataset_root)
    processed_dir = Path(processed_dir)
    metadata_dir = Path(metadata_dir)

    files_by_variable, years = _discover_all(dataset_root)
    logger.info("Detected years: %s", years)
    rainfall_mask = verify_and_cache_rainfall_mask(
        files_by_variable[REFERENCE_VARIABLE],
        processed_dir,
        logger,
    )

    for key, variable in VARIABLES.items():
        convert_variable_years_to_csv(
            list(files_by_variable[key].values()),
            variable,
            dataset_root,
            processed_dir,
            logger,
            drop_invalid=drop_invalid_rainfall_csv if key == REFERENCE_VARIABLE else False,
            valid_mask=rainfall_mask if key == REFERENCE_VARIABLE else None,
        )

    for key, variable in VARIABLES.items():
        combine_yearly_csvs(variable, years, processed_dir, logger)

    reference = VARIABLES[REFERENCE_VARIABLE]
    write_metadata(
        years,
        reference,
        rainfall_mask,
        metadata_dir,
        include_mask_channel=include_mask_channel,
    )
    logger.info("Metadata saved")

    create_stacked_tensors(
        files_by_variable,
        years,
        processed_dir,
        logger,
        include_mask_channel=include_mask_channel,
        rainfall_mask=rainfall_mask,
    )
    verify_pipeline(
        files_by_variable,
        years,
        processed_dir,
        metadata_dir,
        logger,
        rainfall_mask=rainfall_mask,
    )
    logger.info("Pipeline completed")


def get_discovered_years(dataset_root: Path = DATASET_ROOT) -> list[int]:
    """Return years common to all variables."""

    _, years = _discover_all(Path(dataset_root))
    return years
