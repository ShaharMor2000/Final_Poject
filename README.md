# Final Project: Traffic Speed Prediction and Flow Optimization

This project explores multiple approaches to predict roadway speed from traffic flow data and to optimize flow using learned models. It includes classic ML baselines, per-station models, LSTM-based predictors coupled with a genetic algorithm, and a graph-based spatiotemporal model (GCN + LSTM).

## What is inside
- **Model 1/2 (LSTM + GA):** Predict speed from flow history, detect anomalies with an autoencoder, then optimize flow using a genetic algorithm. Outputs are saved under `runs/model1_bilstm` or `runs/model2_lstm`.
- **Model 3 (GCN + LSTM):** Graph convolution for spatial dependencies + LSTM for temporal dependencies. Predicts speed from flow and saves outputs under `runs/model3_gcn`.
- **Classic ML baselines:** Several sklearn models trained on a 5-minute dataset with feature engineering.
- **Per-station models:** KNN and RandomForest per station with capacity-oriented analysis.

## Project layout
- `data/` raw and derived datasets (large CSVs)
- `runs/` model outputs (weights, logs, plots, summaries)
- `LSTM_GA.py` LSTM/BiLSTM + anomaly autoencoder + genetic algorithm pipeline
- `gcn_matrices_experiment.py` graph-based spatiotemporal model
- `data_models_5_minute_dataset.py` classic ML baselines
- `individual_station_models.py` per-station KNN/RandomForest models
- `create_search_dataframe.py` utility to generate a search dataframe

## Data files
Expected inputs (not all are included by default):
- `data/data.csv` (Flow, Speed, Occupancy)
- `data/metadata_extended.csv`, `data/metadata.txt`, `data/metadata_with_embeddings.csv`
- `data/train.csv`, `data/test.csv`
- `data/adj_matrix_directed.csv`
- `data/speed_search.csv`, `data/speed_search_with_speed.csv` (used by some scripts)

If `speed_search*.csv` are missing, generate or provide them before running the related scripts.

## Setup
Python 3.10 is recommended. Main dependencies (by script):
- LSTM + GA: `tensorflow==2.20.0`, `deap==1.4.3`, `numpy==2.2.6`, `pandas`, `scikit-learn`, `matplotlib`
- GCN: `tensorflow`, `haversine`, `numpy`, `pandas`, `matplotlib`
- Classic ML / per-station: `scikit-learn`, `xgboost`, `lightgbm`, `annoy`, `seaborn`

Example install (adjust as needed):
```bash
pip install tensorflow==2.20.0 deap==1.4.3 numpy==2.2.6 pandas scikit-learn matplotlib seaborn xgboost lightgbm annoy haversine
```

## Run instructions
All commands are run from the `finalProject/` folder.

### 1) LSTM + GA (speed prediction + flow optimization)
```bash
python LSTM_GA.py --run_dir runs/model2_lstm
```
Notes:
- Default uses the stacked LSTM. To use BiLSTM, enable the `Bidirectional(...)` layers in `build_speed_model`.
- Training is controlled by the `TRAINED` flag inside `LSTM_GA.py`.
- Outputs: `runs/.../weights/`, `logs/`, `outputs/`, `plots/`.

### 2) GCN + LSTM (spatiotemporal)
```bash
python gcn_matrices_experiment.py
```
Outputs: `runs/model3_gcn/weights/`, `logs/`, `outputs/`, `plots/`.

### 3) Classic ML baselines
```bash
python data_models_5_minute_dataset.py
```
Outputs include saved model pickles under `models/` and prediction CSVs.

### 4) Per-station models
```bash
python individual_station_models.py
```
Outputs: `data/capacity_individuals2test.csv`.

### 5) Utility: build search dataframe
```bash
python create_search_dataframe.py
```
Outputs: `data/speed_search2test.csv`.

## Outputs (typical)
- `runs/model*/weights/` model weights (`*.weights.h5`)
- `runs/model*/logs/` training history (`*.csv`)
- `runs/model*/outputs/` evaluation summaries, best individual
- `runs/model*/plots/` histograms and prediction plots

## Notes
- Large CSVs can require significant RAM; run on a machine with enough memory.
- GPU is optional but can speed up TensorFlow training.
- Some scripts are adapted from Colab notebooks; look at the top comments in each file for the original source.
