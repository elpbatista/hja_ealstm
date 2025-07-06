# Streamflow Modeling with LSTM – Progress Summary

## 1. Data Acquisition & Preparation

- **Streamflow Data**: Daily discharge data from station **HF004** at H.J. Andrews Experimental Forest.
- **Meteorological Data**: Retrieved using the **Open-Meteo Historical Weather API**, including:
  - Precipitation
  - Temperature
  - Additional variables as needed
- **Time Alignment**: Synchronized meteorological inputs and streamflow targets.
- **Data Formatting**: Saved as normalized sequences suitable for deep learning (`.npy` format).

---

## 2. Baseline LSTM Model

- Implemented a standard **LSTM regression model** to predict daily discharge.
- Trained using:
  - 270-day input sequences
  - Streamflow as target
  - 80/20 train-test split

### Model Enhancements

- **Input/Output Normalization** using `StandardScaler`
- **Learning Rate Decay** via `ReduceLROnPlateau`
- **Early Stopping** to prevent overfitting
- **Model Checkpointing** (best model saved)
- **Metrics Logged**:
  - MSE (Mean Squared Error)
  - NSE (Nash–Sutcliffe Efficiency)

### Results

- Achieved **NSE ≈ 0.64**, indicating good predictive skill.
- Visual inspection shows the model:
  - Captures seasonal timing and baseflow trends
  - Underpredicts extreme events, typical for standard LSTM

---

## 3. Output & Visualization

- Saved:
  - Predicted vs. Observed plots (denormalized)
  - Residuals (Observed – Predicted)
  - Final `.npy` arrays for postprocessing
- Results demonstrate:
  - Low bias overall
  - Residuals centered around zero
  - Some smoothing of high discharge events

---

## 4. Next Phase: EA-LSTM Upgrade

You are now transitioning to an **Entity-Aware LSTM** that:

- Integrates static catchment features (e.g., elevation, area)
- Adds day-of-year encoding to better capture seasonal variation
- Implements custom loss weighting to emphasize peak flows
