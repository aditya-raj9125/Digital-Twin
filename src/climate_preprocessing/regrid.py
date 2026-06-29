"""Spatial harmonization utilities."""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from tqdm import tqdm

from config import VariableConfig


def interpolate_to_reference_grid(
    data: np.ndarray,
    source: VariableConfig,
    reference: VariableConfig,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Linearly interpolate a variable to the reference grid using xarray."""

    if data.ndim != 3:
        raise ValueError(f"Expected 3D data for interpolation, got shape {data.shape}")
    if data.shape[1:] != source.shape:
        raise ValueError(
            f"Interpolation input shape mismatch for {source.key}: "
            f"expected {source.shape}, got {data.shape[1:]}"
        )

    data_array = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(data.shape[0]),
            "lat": source.latitudes,
            "lon": source.longitudes,
        },
    )
    with tqdm(total=1, desc=f"Interpolate {source.key}", unit="year") as progress:
        interpolated = data_array.interp(
            lat=reference.latitudes,
            lon=reference.longitudes,
            method="linear",
        )
        progress.update(1)

    result = interpolated.to_numpy().astype(np.float32, copy=False)
    if result.shape[1:] != reference.shape:
        raise ValueError(
            f"Interpolation output dimensions incorrect for {source.key}: {result.shape}"
        )

    if logger:
        logger.info("Interpolation completed for %s", source.key)
    return result
