# AI-Powered Climate Digital Twin Preprocessing

Production-oriented preprocessing pipeline for IMD gridded climate datasets. It reads binary GRD rainfall, maximum temperature, and minimum temperature files, exports long-format CSVs, harmonizes temperature grids to the rainfall grid, and creates synchronized stacked tensors for future ConvLSTM or Transformer training.

No machine learning model is included.

## Installation

Use Python 3.12 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For direct package imports from the repository root:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
```

## Folder Structure

```text
Datasets/
  Rainfall Data/
  Max_Temp/
  Min_Temp/
src/
  climate_preprocessing/
processed/
logs/
metadata/
notebooks/
config.py
requirements.txt
```

The default input root is `Datasets/`. Source GRD files are never modified.

## Input Format

The pipeline automatically scans GRD files and extracts years from file names.

Rainfall:

- `float32`
- shape `129 x 135`
- missing values `-999.0` in the provided files, with `99.9` also treated as missing for compatibility
- reference grid

Maximum and minimum temperature:

- `float32`
- shape `31 x 31`
- missing value `99.9`

Daily record count is inferred from file size and checked against the calendar year, including leap years.

## Run

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
python -m climate_preprocessing
```

Optional paths:

```powershell
python -m climate_preprocessing --dataset-root Datasets --processed-dir processed --metadata-dir metadata --log-dir logs
```

## Outputs

Yearly long-format CSVs:

```text
processed/rainfall/2012.csv
processed/max_temp/2012.csv
processed/min_temp/2012.csv
```

Combined CSVs:

```text
processed/combined/rainfall_all.csv
processed/combined/max_temp_all.csv
processed/combined/min_temp_all.csv
```

Interpolated temperature grids:

```text
processed/interpolated/max_temp/2012.npz
processed/interpolated/min_temp/2012.npz
```

Stacked tensors:

```text
processed/stacked_tensor/train_tensor_2012_2024.npz
processed/stacked_tensor/validation_2025.npz
```

The tensor shape is:

```text
(days, 3, 129, 135)
```

Channel order:

```text
0 Rainfall
1 Tmax
2 Tmin
```

The training tensor contains years up to 2024 only. The validation tensor contains 2025 only.

Metadata:

```text
processed/metadata/dates.csv
processed/metadata/latitudes.csv
processed/metadata/longitudes.csv
processed/metadata/channels.csv
processed/metadata/rainfall_mask.csv
processed/metadata/rainfall_mask.npy
processed/metadata/rainfall_mask.png
```

Logs and verification images:

```text
logs/pipeline.log
processed/verification_plots/
```

## Rainfall Spatial Mask

The IMD rainfall product uses a rectangular `129 x 135` grid, but only part of that rectangle belongs to the analysed rainfall domain. In the provided files, `4,964` cells are valid and `12,451` cells are permanently marked as `-999.0`, which is approximately `71.5%` of the grid.

These permanently invalid cells are expected. They represent regions outside the IMD rainfall analysis domain, such as ocean, border areas, and areas outside the target rainfall product coverage. They are not random missing rainfall observations.

The pipeline verifies that the rainfall mask is identical across all years and all days before using it. It then caches the mask in `processed/metadata/rainfall_mask.npy`, exports a tabular version to `processed/metadata/rainfall_mask.csv`, and saves a visualization to `processed/metadata/rainfall_mask.png`.

After maximum and minimum temperature grids are interpolated from `31 x 31` to the rainfall grid, the same rainfall mask is applied to temperature as well. This guarantees that Rainfall, Tmax, and Tmin share the exact same spatial domain in the stacked tensor.

By default, tensors have three channels:

```text
0 Rainfall
1 Tmax
2 Tmin
```

Use `--include-mask-channel` to add channel `3`, where valid rainfall cells are `1` and invalid cells are `0`.

## Troubleshooting

- `Missing source folder`: confirm `Datasets/Rainfall Data`, `Datasets/Max_Temp`, and `Datasets/Min_Temp` exist.
- `Incorrect file size`: the GRD file does not match the configured grid shape or `float32` datatype.
- `Leap-year record mismatch`: the inferred record count does not match 365 or 366 days for the detected year.
- `Training tensor includes 2025`: stop and inspect configuration before using the output for training.
- `Rainfall mask differs`: stop preprocessing because the permanent rainfall domain mask is not consistent across files.
- Large CSV generation can require significant disk space because every day-grid cell pair is written as one row.
