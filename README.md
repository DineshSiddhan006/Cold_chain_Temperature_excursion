# Cold Chain Temperature Excursion Prediction Dataset

## Dataset Overview
This synthetic dataset was engineered for the warehouse operations ML use case:

- Prediction ID: WH-002
- Prediction Name: Cold Chain Temperature Excursion Prediction
- Total Records: 25,000
- Temporal Coverage: March 2025 → February 2026
- Time Structure: Continuous chronological series
- Intended Validation Strategy: 5-Fold TimeSeriesSplit

## Target Variable
`excursion_risk_score`
- Continuous float between 0.00 and 1.00
- Formatted to exactly 2 decimal places
- Represents the probability of a temperature excursion within a future 2-hour observation window

Special Rule:
- If `current_internal_temp` falls below `target_temp_min`
  OR exceeds `target_temp_max`,
  then `excursion_risk_score` is forced to exactly 1.00.

---

# Embedded Thermodynamic & Operational Rules

## 1. Strict SHC Mapping
| SHC | Min Temp | Max Temp |
|---|---|---|
| PIL | 2.00°C | 8.00°C |
| COL | 2.00°C | 8.00°C |
| FRO | -30.00°C | -18.00°C |
| CRT | 15.00°C | 25.00°C |

These thresholds are deterministic and never randomized.

---

## 2. Seasonal Desert Climate Dynamics
Summer months (May–September):
- Ambient temperatures scale between 38°C and 45°C
- Baseline excursion risk increases
- Compressor duty cycles rise
- Internal temperatures drift toward upper thresholds

Winter months:
- Ambient temperatures fall between 15°C and 22°C
- Risk baselines remain lower

---

## 3. Rainfall & Moisture Behavior
During rainy summer periods:
- Humidity scales between 85% and 100%
- Moisture accumulation increases defrost probability
- When doors remain open >300 seconds:
  - Defrost activation probability rises to 40%
  - Risk accelerates by 25%

---

## 4. Equipment Degradation
Older refrigeration assets (>7 years):
- Maintain elevated compressor duty cycles (>0.80)
- Demonstrate reduced cooling reserve capacity
- Reach thermal saturation faster under door stress

Catastrophic condition:
- Compressor duty cycle = 1.00
- Door open duration >600 sec
- Internal temperature spikes rapidly
- Risk rises above 0.90

---

## 5. Cargo Thermal Mass Buffer
Low cargo utilization (<0.20):
- Accelerates thermal drift
- Reduces thermal inertia
- Causes faster warming during door events

Frozen cargo (`FRO`) has slower thermal response due to sub-zero pallet mass buffering.

---

# Recommended ML Validation Strategy

## Why Random Splits Are Invalid
This dataset contains:
- Seasonal dependencies
- Sequential thermal drift
- Continuous operational cycles
- Equipment aging progression
- Weather-linked temporal autocorrelation

A random train/test split would leak future information into training folds.

---

# Correct Validation Method: 5-Fold TimeSeriesSplit

Recommended:
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(df):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

Behavior:
- Earlier periods train the model
- Later periods validate the model
- Temporal ordering is preserved
- No future leakage occurs

---

# Suggested ML Targets
Suitable models:
- XGBoost
- LightGBM
- CatBoost
- Temporal CNNs
- LSTM/GRU architectures
- Transformer-based time-series models

---

# Included Files
1. cold_chain_excursion_data.csv
2. README.md
