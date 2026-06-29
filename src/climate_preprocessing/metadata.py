"""Metadata generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    CHANNEL_ORDER,
    MASK_CHANNEL_LABEL,
    METADATA_DIR,
    VARIABLES,
    VariableConfig,
)
from climate_preprocessing.dates import dates_for_years
from climate_preprocessing.mask import write_rainfall_mask_artifacts


def write_metadata(
    years: list[int],
    reference: VariableConfig,
    rainfall_mask,
    metadata_dir: Path = METADATA_DIR,
    include_mask_channel: bool = False,
) -> None:
    """Write dates, latitude, longitude, and channel metadata CSV files."""

    metadata_dir.mkdir(parents=True, exist_ok=True)

    dates = dates_for_years(years)
    pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Year": dates.year,
            "Month": dates.month,
            "Day": dates.day,
            "Day_of_Year": dates.dayofyear,
        }
    ).to_csv(metadata_dir / "dates.csv", index=False)

    pd.DataFrame({"Latitude": reference.latitudes}).to_csv(
        metadata_dir / "latitudes.csv", index=False
    )
    pd.DataFrame({"Longitude": reference.longitudes}).to_csv(
        metadata_dir / "longitudes.csv", index=False
    )
    channel_names = [VARIABLES[key].tensor_label for key in CHANNEL_ORDER]
    channel_indexes = [VARIABLES[key].channel_index for key in CHANNEL_ORDER]
    if include_mask_channel:
        channel_indexes.append(3)
        channel_names.append(MASK_CHANNEL_LABEL)

    pd.DataFrame(
        {
            "Channel": channel_indexes,
            "Name": channel_names,
        }
    ).to_csv(metadata_dir / "channels.csv", index=False)
    write_rainfall_mask_artifacts(rainfall_mask, metadata_dir.parent)
