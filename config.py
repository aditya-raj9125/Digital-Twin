"""Central configuration for the climate preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "Datasets"
PROCESSED_DIR = PROJECT_ROOT / "processed"
LOG_DIR = PROJECT_ROOT / "logs"
METADATA_DIR = PROCESSED_DIR / "metadata"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

DTYPE = np.float32
SUPPORTED_DTYPES = (np.float32,)
TRAIN_END_YEAR = 2024
VALIDATION_YEAR = 2025
RANDOM_SEED = 42
EXPECTED_RAINFALL_VALID_CELLS = 4964
EXPECTED_RAINFALL_INVALID_CELLS = 12451
INCLUDE_MASK_CHANNEL = False


@dataclass(frozen=True)
class VariableConfig:
    """Configuration for one climate variable."""

    key: str
    display_name: str
    source_folder: str
    output_folder: str
    shape: tuple[int, int]
    lat_start: float
    lat_step: float
    lon_start: float
    lon_step: float
    channel_index: int
    tensor_label: str
    missing_values: tuple[float, ...]

    @property
    def latitudes(self) -> np.ndarray:
        return (
            self.lat_start + np.arange(self.shape[0], dtype=np.float32) * self.lat_step
        ).astype(np.float32)

    @property
    def longitudes(self) -> np.ndarray:
        return (
            self.lon_start + np.arange(self.shape[1], dtype=np.float32) * self.lon_step
        ).astype(np.float32)


VARIABLES: dict[str, VariableConfig] = {
    "rainfall": VariableConfig(
        key="rainfall",
        display_name="Rainfall",
        source_folder="Rainfall Data",
        output_folder="rainfall",
        shape=(129, 135),
        lat_start=6.5,
        lat_step=0.25,
        lon_start=66.5,
        lon_step=0.25,
        channel_index=0,
        tensor_label="Rainfall",
        missing_values=(-999.0, 99.9),
    ),
    "max_temp": VariableConfig(
        key="max_temp",
        display_name="Maximum Temperature",
        source_folder="Max_Temp",
        output_folder="max_temp",
        shape=(31, 31),
        lat_start=7.5,
        lat_step=1.0,
        lon_start=67.5,
        lon_step=1.0,
        channel_index=1,
        tensor_label="Tmax",
        missing_values=(99.9,),
    ),
    "min_temp": VariableConfig(
        key="min_temp",
        display_name="Minimum Temperature",
        source_folder="Min_Temp",
        output_folder="min_temp",
        shape=(31, 31),
        lat_start=7.5,
        lat_step=1.0,
        lon_start=67.5,
        lon_step=1.0,
        channel_index=2,
        tensor_label="Tmin",
        missing_values=(99.9,),
    ),
}

REFERENCE_VARIABLE = "rainfall"
TEMPERATURE_VARIABLES = ("max_temp", "min_temp")
CHANNEL_ORDER = ("rainfall", "max_temp", "min_temp")
MASK_CHANNEL_LABEL = "Rainfall Valid Mask"
