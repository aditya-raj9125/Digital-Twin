"""Date helpers."""

from __future__ import annotations

import pandas as pd


def dates_for_year(year: int) -> pd.DatetimeIndex:
    """Create daily dates for a full calendar year."""

    return pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")


def dates_for_years(years: list[int]) -> pd.DatetimeIndex:
    """Create a continuous daily date index for sorted years."""

    ranges = [dates_for_year(year) for year in sorted(years)]
    if not ranges:
        raise ValueError("No years supplied for date generation")
    return ranges[0].append(ranges[1:]) if len(ranges) > 1 else ranges[0]
