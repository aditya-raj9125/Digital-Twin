# 🌏 AI-Powered Climate Digital Twin – Climate Data Preprocessing Pipeline

> A complete preprocessing framework for converting raw Indian Meteorological Department (IMD) gridded climate datasets into deep learning–ready spatiotemporal tensors for Climate Digital Twin applications.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Latest-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Latest-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

# Overview

This repository contains the complete preprocessing pipeline developed for the **AI-Powered Climate Digital Twin** project.

The objective is to transform raw daily climate observations from the **Indian Meteorological Department (IMD)** into structured spatiotemporal tensors suitable for training deep learning models such as **ConvLSTM**, **CNN-LSTM**, **Vision Transformers**, and other temporal forecasting architectures.

The preprocessing pipeline performs:

- Parsing IMD GRD datasets
- Spatial interpolation
- Temporal alignment
- Missing value handling
- Multi-variable tensor construction
- Regional tensor extraction (Kerala Pilot Region)
- Training / Validation dataset generation
- Metadata preservation

The final output is a compact tensor representation that can be directly consumed by deep learning frameworks.

---

# Project Goal

The primary objective of this preprocessing framework is to generate a clean, structured, and scalable dataset for forecasting:

- 🌧 Rainfall
- 🌡 Maximum Temperature (Tmax)
- 🌡 Minimum Temperature (Tmin)

using historical observations from IMD.

These tensors serve as the input dataset for building an AI-driven Climate Digital Twin capable of predicting future climate conditions.

---

# Pipeline Overview

```
                 IMD GRD Files
                       │
                       ▼
          Parse Binary GRD Datasets
                       │
                       ▼
            Daily Spatial Grids
                       │
                       ▼
          Coordinate Standardization
                       │
                       ▼
        Missing Value Identification
                       │
                       ▼
         Spatial / Temporal Interpolation
                       │
                       ▼
      Multi-variable Tensor Generation
                       │
                       ▼
         Regional Tensor Extraction
            (Kerala Pilot Region)
                       │
                       ▼
         Train / Validation Split
                       │
                       ▼
     Deep Learning Ready Tensors (.npz)
```

---

# Dataset Used

The preprocessing pipeline uses gridded datasets provided by the **India Meteorological Department (IMD).**

Variables processed:

| Variable | Unit |
|------------|------|
| Rainfall | mm/day |
| Tmax | °C |
| Tmin | °C |

Temporal Resolution

- Daily

Spatial Resolution

- 0.25°

Coverage

- India

---

# Repository Structure

```
Digital-Twin/

│
├── notebooks/
│      End-to-end preprocessing notebooks
│
├── src/
│      climate_preprocessing/
│          GRD parser
│          interpolation
│          tensor generation
│          utilities
│
├── processed/
│      Generated tensors
│
├── config.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

# Complete Preprocessing Workflow

## Stage 1 — GRD Parsing

Raw IMD binary GRD files are parsed into NumPy arrays.

Outputs

- Daily raster grids
- Latitude coordinates
- Longitude coordinates
- Observation dates

---

## Stage 2 — Coordinate Alignment

Different variables are aligned onto the same geographical grid.

This ensures that:

```
Rainfall(x,y)

Tmax(x,y)

Tmin(x,y)

refer to the exact same location.
```

---

## Stage 3 — Missing Value Processing

Raw climate datasets contain missing observations represented as NaN.

The pipeline:

- Detects missing cells
- Preserves ocean pixels
- Interpolates valid land observations
- Retains spatial consistency

Result

```
Raw Grid

██████ NaN ████

↓

Interpolation

██████████████
```

---

## Stage 4 — Multi-variable Tensor Construction

Daily observations are stacked into a single tensor.

Tensor format

```
(Time,
 Variables,
 Latitude,
 Longitude)
```

Specifically

```
(T, 3, H, W)

where

3 =
Rainfall
Tmax
Tmin
```

Example

```
tensor[120]

↓

Day 120

Rainfall Grid

Tmax Grid

Tmin Grid
```

---

# Tensor Dimensions

Original Tensor

```
(Time,
3 Variables,
Latitude,
Longitude)
```

Example

```
(4749,
3,
129,
135)
```

---

# Kerala Tensor Trimming

To accelerate model training and reduce computational cost, a regional tensor extraction stage is implemented.

Instead of using the entire Indian subcontinent, the tensor is cropped to the Kerala pilot region.

Geographical Extent

```
Latitude

8.0°N
↓

13.0°N
```

```
Longitude

74.5°E
↓

77.5°E
```

Result

Original Tensor

```
129 × 135
```

↓

Kerala Tensor

```
21 × 13
```

This reduces:

- Memory usage
- GPU utilization
- Training time

while preserving the complete spatial characteristics of Kerala.

---

# Kerala Extraction Workflow

```
India Tensor

129 × 135

──────────────

Select

Lat

8°–13°

Lon

74.5°–77.5°

──────────────

↓

Kerala Tensor

21 × 13
```

---

# Dataset Splitting

The dataset is separated chronologically.

Training Dataset

```
2012

↓

2024
```

Validation Dataset

```
2025
```

No temporal leakage occurs between training and validation.

---

# Output Files

Training Tensor

```
train_tensor_kerala.npz
```

Validation Tensor

```
validation_tensor_kerala.npz
```

Each tensor contains

```
tensor

dates

latitudes

longitudes

rainfall_mask
```

---

# Tensor Contents

```
tensor
```

Main climate tensor

Shape

```
(Time,
3,
21,
13)
```

---

```
dates
```

Daily timestamps

---

```
latitudes
```

Latitude coordinates

---

```
longitudes
```

Longitude coordinates

---

```
rainfall_mask
```

Boolean land mask used during model training to ignore invalid spatial locations.

---

# Why Tensor Trimming?

Training on the complete Indian tensor requires significantly higher computational resources.

The Kerala pilot tensor

✔ reduces GPU memory

✔ accelerates experimentation

✔ enables rapid prototyping

✔ preserves regional climatic behaviour

---

# End Product

The final dataset is optimized for deep learning models such as

- ConvLSTM
- CNN-LSTM
- Vision Transformer
- Spatiotemporal Transformers
- Sequence-to-Sequence Forecasting Networks

Example training input

```
30 Days

↓

Deep Learning Model

↓

Predict

Next 7 Days
```

---

# Example Tensor Flow

```
Day 1

Rainfall
Tmax
Tmin

↓

Day 2

↓

...

↓

Day 30

↓

Neural Network

↓

Forecast

Day 31–37
```

---

# Advantages

✔ Automated preprocessing

✔ Regional tensor extraction

✔ Missing value handling

✔ Metadata preservation

✔ Deep learning ready

✔ Memory efficient

✔ Easily extendable to additional climate variables

---

# Future Extensions

Potential enhancements include

- Relative Humidity
- Wind Speed
- Wind Direction
- Surface Pressure
- Solar Radiation
- Soil Moisture
- Satellite-derived products
- Multi-resolution tensors

---

# Applications

The generated tensors can be used for

- Climate Digital Twins
- Weather Forecasting
- Flood Prediction
- Heatwave Prediction
- Agricultural Forecasting
- Climate Risk Assessment
- Disaster Management
- AI-based Environmental Monitoring

---

# Citation

If you use this preprocessing framework in your research, please cite this repository appropriately.

---

# Author

**Aditya Raj**

AI-Powered Climate Digital Twin Project

Indian Meteorological Data Preprocessing Framework