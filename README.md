# Rainfall-runoff modeling on H.J. Andrews using EA-LSTM

This project implements a rainfall–runoff modeling framework using deep learning, specifically an LSTM and EA-LSTM architecture, to model daily streamflow at the H.J. Andrews Experimental Forest.

The workflow includes:

- Data acquisition and preprocessing of streamflow and meteorological data  
- Sequence generation for LSTM input  
- Normalization and training of a baseline LSTM model  
- Visualization of predictions and residuals  
- Plans to upgrade to EA-LSTM with static attributes and seasonal encodings  

## Downloading Streamflow Data (HF004)

This project uses **daily streamflow summaries** from the H.J. Andrews Experimental Forest, available through the Environmental Data Initiative (EDI). Due to repository restrictions, this dataset must be **downloaded manually**.

Please follow these steps:

1. Visit the dataset page on EDI:  
   <https://portal.edirepository.org/nis/mapbrowse?packageid=knb-lter-and.4341.35>

2. Scroll to the list of **Data Entities** and locate:  
   **Daily streamflow summaries** (Entity 2, approximately 14.7 MiB)

3. Click the **Download** icon next to that entry.

4. Save the file as:  
   `HF004_daily_streamflow_entity2.csv`

5. Move the file to the following directory within this project:  
   `data/streamflow/HF004_daily_streamflow_entity2.csv`

### Citation

Johnson, S.L., Wondzell, S.M., & Rothacher, J.S. (2023). *Stream discharge in gaged watersheds at the H.J. Andrews Experimental Forest, 1949 to present ver 35*. Environmental Data Initiative. <https://doi.org/10.6073/pasta/62bd85397e23bc9c50c7f652c0cc58ea> (Accessed 2025-07-05)

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/elpbatista/hja-ealstm.git
cd hja-ealstm

# Install dependencies
poetry install
```

## Quick Start

1. **Manually download and place the HF004 streamflow CSV file as described above.**

2. **Prepare meteorological data and input sequences:**

```bash
poetry run python src/hja_ealstm/data/download_streamflow.py
poetry run python src/hja_ealstm/data/download_openmeteo_weather.py
poetry run python src/hja_ealstm/data/prepare_lstm_sequences.py
poetry run python src/hja_ealstm/data/generate_lstm_training_data.py
```

3. **Train the baseline LSTM model:**

```bash
poetry run python src/hja_ealstm/model/train_lstm_normalized.py
```

## Expected Results

After training, you should find:

- The best model checkpoint saved in the `models/` directory  
- Training logs with metrics such as MSE and NSE  
- Prediction and residual plots in the `results/` directory  
- `.npy` arrays of predictions for further analysis  

With the current configuration, the baseline model typically achieves:

- **NSE ≈ 0.64** on validation data  
- Accurate seasonal discharge trends  
- Moderate underestimation of peak flows

---

For more details on model architecture, data sources, or troubleshooting, see the comments in each script or open an issue on GitHub.
