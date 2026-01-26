"""
You have 2 skipping points for training the 
model, after the comment:
# Uncomment the following if you need to train
the training line appears.

If you are already training the models, leave this comments.

Two speed prediction architectures were evaluated in this study.
Model 1 employs a Bidirectional LSTM (BiLSTM), which processes the input sequence in both forward and backward temporal directions, allowing the model to capture richer temporal dependencies.
Model 2 uses a standard stacked LSTM architecture, where multiple LSTM layers are applied sequentially in a single forward temporal direction.
Both models share the same data preprocessing pipeline, training procedure, evaluation metrics, and downstream optimization framework. The comparison focuses on the effect of bidirectionality on speed prediction performance.

Each speed prediction architecture is executed and stored in a separate run directory.
The BiLSTM-based model is saved under runs/model1_bilstm, while the stacked LSTM-based model is saved under runs/model2_lstm, with identical subdirectory structures for weights, logs, outputs, and plots.

tensorflow==2.20.0
deap==1.4.3
numpy==2.2.6
python3.10.11
GPU - NVIDIA GeForce RTX 4060
"""

import argparse
import random
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, RepeatVector, TimeDistributed
from sklearn.metrics import r2_score

from deap import base, creator, tools, algorithms

TRAINED = False
# ----------------------------
# Utilities
# ----------------------------
def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_dataset(flow: pd.DataFrame,
                   speed: pd.DataFrame,
                   occ: pd.DataFrame,
                   n_past: int,
                   n_future: int):
    """
    - X sample is flow window for each station: shape (N_ST, n_past)
    - y is speed at time i-1 for each station: shape (N_ST,)
    """
    x, y = [], []
    num_stations = len(flow.columns)

    for i in range(n_past + 1, len(flow) - n_future):
        total_flow = flow[(i - n_past):i].values.transpose().reshape(num_stations, n_past)
        # NOTE: although occupancy and speed windows are computed, only flow history
        # is used as model input in order to preserve the original experimental setup  
        x.append(total_flow)
        y.append(speed[i - 1:i].values.reshape(1, num_stations)[0])

    return np.array(x), np.array(y)


def compute_high_flow_low_speed(flow_df: pd.DataFrame, speed_df: pd.DataFrame, pct: float = 95.0):
    high_flow, low_speed = [], []
    for col in flow_df.columns:
        flow_ind = flow_df[col].values
        speed_ind = speed_df[col].values
        thr = np.percentile(flow_ind, pct)

        hf = flow_ind[flow_ind > thr]
        ls = speed_ind[flow_ind > thr]

        # Numerical safeguard: handle empty slices to prevent NaN values
        high_flow.append(float(np.mean(hf)) if hf.size else 0.0)
        low_speed.append(float(np.mean(ls)) if ls.size else 0.0)

    return high_flow, low_speed


def build_speed_model(n_st: int, n_past: int, out_dim: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(n_st, n_past)),
        # Bidirectional(LSTM(64, return_sequences=True)),
        # Bidirectional(LSTM(32, return_sequences=False)),
        LSTM(64, return_sequences=True),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(100),
        Dense(out_dim),
    ])
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def build_anomaly_autoencoder(n_st: int, n_past: int) -> tf.keras.Model:
    input_shape = (n_st, n_past)
    m = Sequential()
    m.add(Input(shape=input_shape))
    m.add(LSTM(256, return_sequences=True))
    m.add(LSTM(128, return_sequences=False))
    m.add(RepeatVector(n_st))
    m.add(LSTM(128, return_sequences=True))
    m.add(LSTM(256, return_sequences=True))
    m.add(TimeDistributed(Dense(n_past)))
    m.compile(
        optimizer="adam",
        loss="mae",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    return m


def make_evaluate(prev_2cols: np.ndarray,
                  max_flow: np.ndarray,
                  threshold: float,
                  speed_model: tf.keras.Model,
                  anomaly_model: tf.keras.Model):
    """
    prev_2cols: shape (N_ST, 2)
    Individual encodes the 3rd feature column for each station (flow decision).
    """
    n_st = prev_2cols.shape[0]

    def evaluate(individual):
        # individual -> (N_ST,)
        ind = np.asarray(individual, dtype=float)

        # Bound handling (fitness_flows): if above max_flow => contribute 0
        fitness_flows = np.where(ind > max_flow, 0.0, ind)

        # Build flows tensor: concatenate [ind] as first column + prev (2 cols) -> (N_ST, 3)
        arr = ind.reshape(n_st, 1)
        arr = np.concatenate((arr, prev_2cols), axis=1)  # (N_ST, 3)
        arr = np.round(arr).astype(int)

        flows = arr.reshape(1, n_st, 3)

        y_pred_speed = speed_model.predict(flows, verbose=0)  # (1, N_ST)
        y_pred_flow = anomaly_model.predict(flows, verbose=0)  # (1, N_ST, 3)

        mse = float(np.mean(np.square(y_pred_flow - flows), axis=(1, 2))[0])
        is_normal = threshold > mse

        # Indexing adjustment to align prediction output with number of stations
        fitness = float(np.sum(fitness_flows * y_pred_speed[0]))

        if not is_normal:
            # Penalize candidate solutions classified as anomalous based on reconstruction error
            fitness = float(((threshold / max(mse, 1e-9)) ** 2) * fitness)

        return fitness,

    return evaluate


def mate_individuals_factory(max_flow: np.ndarray):
    def mate_individuals(ind1, ind2):
        size = min(len(ind1), len(ind2))
        quart = int(size / 4)
        three_quart = int(size * 3 / 4)
        cxpoint = random.randint(quart, max(quart, three_quart))
        ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

        # Clamp to [0, max_flow[i]]
        for i in range(size):
            ind1[i] = max(0, min(ind1[i], float(max_flow[i])))
            ind2[i] = max(0, min(ind2[i], float(max_flow[i])))

        return ind1, ind2

    return mate_individuals


# def train_model(model, x_train, y_train, x_val, y_val, batch_size, num_epochs, output_filepath):
#     if TRAINED == False:
#         model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=batch_size, verbose=1)
#         model.save_weights(output_filepath)

# def train_model(model, x_train, y_train, x_val, y_val, batch_size, num_epochs, output_filepath, history_csv=None):
#     if TRAINED == False:
#         history = model.fit(
#             x_train, y_train,
#             validation_data=(x_val, y_val),
#             epochs=num_epochs,
#             batch_size=batch_size,
#             verbose=1
#         )
#         model.save_weights(output_filepath)

#         if history_csv is not None:
#             import pandas as pd, os
#             os.makedirs(os.path.dirname(history_csv) or ".", exist_ok=True)
#             pd.DataFrame(history.history).to_csv(history_csv, index=False)
def train_model(model, x_train, y_train, x_val, y_val,
                batch_size, num_epochs, output_filepath, history_csv=None):
    if TRAINED == False:
        callbacks = []

        # 1) Save model weights at each epoch to ensure reproducibility and recovery
        os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=output_filepath,        # e.g. "weights/lstm.weights.h5"
            save_weights_only=True,
            save_best_only=False,            # keep every epoch (no early stopping logic)
            save_freq="epoch",
            verbose=1
        )
        callbacks.append(checkpoint_cb)

        # 2) Log train+val loss to CSV during training
        if history_csv is not None:
            os.makedirs(os.path.dirname(history_csv) or ".", exist_ok=True)
            csv_cb = tf.keras.callbacks.CSVLogger(
                history_csv,
                append=False                  # set True only if you want to continue same file
            )
            callbacks.append(csv_cb)

        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks
        )

        # Final save (optional, but ok)
        model.save_weights(output_filepath)

        return history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", default="data/data.csv")
    parser.add_argument("--metadata_csv", default="data/metadata_extended.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_past", type=int, default=3)
    parser.add_argument("--n_future", type=int, default=1)
    parser.add_argument("--epochs_speed", type=int, default=1)
    parser.add_argument("--epochs_anom", type=int, default=1)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gen", type=int, default=20)
    # parser.add_argument("--run_dir", default="runs/model1_bilstm")
    parser.add_argument("--run_dir", default="runs/model2_lstm")
    # FIX - weights path should end with `.weights.h5`
    parser.add_argument("--weights_speed", default="speed.weights.h5")
    parser.add_argument("--weights_anom", default="anomaly.weights.h5")
    parser.add_argument("--best_out", default="best_individual.csv")
    args = parser.parse_args()

    set_seeds(args.seed)

    run_dir = args.run_dir
    weights_dir = os.path.join(run_dir, "weights")
    logs_dir = os.path.join(run_dir, "logs")
    outputs_dir = os.path.join(run_dir, "outputs")
    plots_dir = os.path.join(run_dir, "plots")

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    args.weights_speed = os.path.join(weights_dir, "speed.weights.h5")
    args.weights_anom = os.path.join(weights_dir, "anomaly.weights.h5")
    args.best_out = os.path.join(outputs_dir, "best_individual.csv")

    speed_history_csv = os.path.join(logs_dir, "speed_history.csv")
    anomaly_history_csv = os.path.join(logs_dir, "anomaly_history.csv")

    run_args_path = os.path.join(run_dir, "run_args.txt")
    with open(run_args_path, "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}={v}\n")

    # ----------------------------
    # Load data
    # ----------------------------
    data = pd.read_csv(args.data_csv)
    # data = data.iloc[: len(data) // 2]

    # Pivot to station-time matrices
    flow_df = data.pivot(index="Timestamp", columns="ID", values="Flow").fillna(0)
    speed_df = data.pivot(index="Timestamp", columns="ID", values="Speed").fillna(0)
    occ_df = data.pivot(index="Timestamp", columns="ID", values="Occupancy").fillna(0)

    # Drop stations that are all-zero in speed (as in user's code)
    constant_zeros = speed_df.columns[speed_df.isin([0]).all()]
    speed_df.drop(constant_zeros, axis=1, inplace=True)
    occ_df.drop(constant_zeros, axis=1, inplace=True)
    flow_df = flow_df.loc[:, ~flow_df.columns.isin(constant_zeros)]

    station_ids = list(flow_df.columns)
    N_ST = len(station_ids)
    print("N stations:", N_ST)

    # ----------------------------
    # Lanes per station (from metadata)
    # ----------------------------
    metadata = pd.read_csv(args.metadata_csv)

    lanes_map = dict(zip(metadata["ID"], metadata["Lanes"]))

    lanes_by_station = np.array(
        [lanes_map.get(sid, 1) for sid in station_ids],
        dtype=float
    )

    # High-flow / low-speed stats
    high_flow, low_speed = compute_high_flow_low_speed(flow_df, speed_df, pct=95.0)
    high_flow = np.array(high_flow, dtype=float)
    low_speed = np.array(low_speed, dtype=float)

    # ----------------------------
    # Train/val/test split (by index slices, like user)
    # ----------------------------
    # train_end = 3000
    # val_end = 4000

    flow_train = flow_df.iloc[:5000]
    flow_val   = flow_df.iloc[5000:6000]
    flow_test  = flow_df.iloc[6000:]

    occ_train = occ_df[station_ids].iloc[:5000]
    occ_val   = occ_df[station_ids].iloc[5000:6000]
    occ_test  = occ_df[station_ids].iloc[6000:]

    speed_train = speed_df[station_ids].iloc[:5000]
    speed_val   = speed_df[station_ids].iloc[5000:6000]
    speed_test  = speed_df[station_ids].iloc[6000:]


    # Create datasets
    X_train, y_train = create_dataset(flow_train, speed_train, occ_train, args.n_past, args.n_future)
    X_val, y_val = create_dataset(flow_val, speed_val, occ_val, args.n_past, args.n_future)
    X_test, y_test = create_dataset(flow_test, speed_test, occ_test, args.n_past, args.n_future)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    # ----------------------------
    # Speed model: train and save weights
    # ----------------------------
    print("Run Speed")
    speed_model = build_speed_model(n_st=N_ST, n_past=args.n_past, out_dim=N_ST)
    # Uncomment the following if you need to train
    # train_model(speed_model, X_train, y_train, X_val, y_val, 16, args.epochs_speed, args.weights_speed)
    train_model(speed_model, X_train, y_train, X_val, y_val, 16, args.epochs_speed, args.weights_speed,
            history_csv=speed_history_csv)

    speed_model.load_weights(args.weights_speed)
    # ---- Final evaluation (Speed) ----
    train_metrics = speed_model.evaluate(X_train, y_train, verbose=0, return_dict=True)
    val_metrics   = speed_model.evaluate(X_val, y_val, verbose=0, return_dict=True)
    test_metrics  = speed_model.evaluate(X_test, y_test, verbose=0, return_dict=True)
   
    # R^2 (computed manually)
    yhat_train = speed_model.predict(X_train, verbose=0)
    yhat_val   = speed_model.predict(X_val, verbose=0)
    yhat_test  = speed_model.predict(X_test, verbose=0)

    r2_train = r2_score(y_train, yhat_train, multioutput="uniform_average")
    r2_val   = r2_score(y_val,   yhat_val,   multioutput="uniform_average")
    r2_test  = r2_score(y_test,  yhat_test,  multioutput="uniform_average")

    summary_path = os.path.join(outputs_dir, "speed_eval_summary.csv")
    pd.DataFrame([
        {"split": "train", **train_metrics, "r2": r2_train},
        {"split": "val",   **val_metrics,   "r2": r2_val},
        {"split": "test",  **test_metrics,  "r2": r2_test},
    ]).to_csv(summary_path, index=False)

    print("Saved speed eval summary:", summary_path)
    # ----------------------------
    # Anomaly model: train and save weights
    # ----------------------------
    print("Run Anomaly")
    anomaly_model = build_anomaly_autoencoder(n_st=N_ST, n_past=args.n_past)
    # Uncomment the following if you need to train
    # train_model(anomaly_model, X_train, X_train, X_val, X_val, 16, args.epochs_anom, args.weights_anom)
    train_model(anomaly_model, X_train, X_train, X_val, X_val, 16, args.epochs_anom, args.weights_anom,
            history_csv=anomaly_history_csv)
    anomaly_model.load_weights(args.weights_anom)
    # ---- Final evaluation (Anomaly) ----
    anom_train = anomaly_model.evaluate(X_train, X_train, verbose=0, return_dict=True)
    anom_val   = anomaly_model.evaluate(X_val, X_val, verbose=0, return_dict=True)
    anom_test  = anomaly_model.evaluate(X_test, X_test, verbose=0, return_dict=True)

    anom_path = os.path.join(outputs_dir, "anomaly_eval_summary.csv")
    pd.DataFrame([
        {"split": "train", **anom_train},
        {"split": "val",   **anom_val},
        {"split": "test",  **anom_test},
    ]).to_csv(anom_path, index=False)

    print("Saved anomaly eval summary:", anom_path)

    # Fix - using X_val to initialize the anomaly threshold.
    # Threshold from reconstruction error on validation set
    reconstructed = anomaly_model.predict(X_val, verbose=0)
    mse_normal = np.mean(np.square(X_val - reconstructed), axis=(1, 2))
    threshold = float(np.mean(mse_normal) + 1.0 * np.std(mse_normal))
    print("Anomaly threshold:", threshold)

    # ----------------------------
    # GA setup
    # ----------------------------
    # max_flow per station (used both in eval and mate bounds)
    max_flow = np.array([float(np.max(flow_df[c].values)) for c in station_ids], dtype=float)

    # Choose a "prev" window from X_val (like user) and take first 2 columns -> (N_ST,2)
    if X_val.shape[0] < 1:
        raise RuntimeError("X_val is empty; cannot sample prev.")
    rand_idx = random.randint(0, X_val.shape[0] - 1)
    prev = X_val[rand_idx]  # (N_ST, n_past) == (N_ST,3)
    if prev.shape[1] < 3:
        raise RuntimeError(f"Expected prev to have 3 columns (n_past=3), got {prev.shape}.")
    prev_2cols = prev[:, 0:2].astype(float)  # (N_ST,2)

    # DEAP creator guards (important in notebooks)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, 1000)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=N_ST)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.register("evaluate", make_evaluate(prev_2cols, max_flow, threshold, speed_model, anomaly_model))
    toolbox.register("mate", mate_individuals_factory(max_flow))

    population = toolbox.population(n=args.pop)

    algorithms.eaMuPlusLambda(
        population,
        toolbox,
        mu=args.pop,
        lambda_=2 * args.pop,
        cxpb=0.7,
        mutpb=0.2,
        ngen=args.gen,
        stats=None,
        halloffame=None,
        verbose=True,
    )

    best_individual = tools.selBest(population, k=1)[0]
    print("Best Fitness:", best_individual.fitness.values)

    best_ind = np.asarray(best_individual, dtype=float).reshape(N_ST, 1)
    best_arr = np.concatenate((best_ind, prev_2cols), axis=1)  # (N_ST,3)
    best_flows = best_arr.reshape(1, N_ST, 3)

    best_speed = speed_model.predict(best_flows, verbose=0)[0]

    ############################ HCM Compare start ###################################
    print("########## Start HCM Compare ##########")
    flow_total = np.asarray(best_individual, dtype=float)  # (N_ST,)

    # If the flow is TOTAL across lanes, divide by lanes to get per-lane flow.
    # (assumes flow already per lane).

    #lanes = np.ones_like(flow_total)
    lanes = lanes_by_station
    q_per_lane = flow_total / np.maximum(lanes, 1.0)  # pc/h/ln (assumption)
    k = density_from_flow_speed(q_per_lane, best_speed)  # pc/km/ln
    los = hcm_los_basic_freeway_from_density_km(k)

    # Compare against LOS A
    is_A = (los == "A")
    print("HCM LOS A stations:", int(is_A.sum()), "/", len(is_A))
    print("Percent LOS A:", float(is_A.mean()) * 100)

    # If you specifically want the LOS A flow limit at 100 km/h:
    qmax_A_at_100 = 7.0 * 100.0  # 700 pc/h/ln
    print("LOS A max flow per lane at 100 km/h:", qmax_A_at_100)

    # Optional: show where you exceed LOS A given your predicted speeds
    qmax_A_each_station = 7.0 * best_speed  # q <= 7*v for LOS A
    violations = q_per_lane > qmax_A_each_station
    print("Stations violating LOS A (by q>7*v):", int(violations.sum()))

    print("########## Finish HCM Compare ##########")
    ############################ HCM compare end ###################################

    # Save best individual (first column only, i.e., the GA decision variable)
    pd.Series(best_individual, index=station_ids, name="best_flow_decision").to_csv(args.best_out)
    print("Saved:", args.best_out)

    # Quick summary stats
    print("Mean predicted speed (model units):", float(np.mean(best_speed)))
    print("Mean chosen flow decision:", float(np.mean(best_individual)))

    import matplotlib.pyplot as plt
    plt.hist(high_flow, 15, alpha=0.5, label='high flow')

    # Use only the GA-selected flow per station (1D) instead of the full 3D tensor
    # plt.hist(best_flows, 15, alpha=0.5, label='GA flow')
    plt.hist(best_flows[0, :, 0], 15, alpha=0.5, label='GA flow')

    plt.legend(loc='upper right')
    plt.title("Flow Histogram")
    plt.savefig(os.path.join(plots_dir, "histogramFlow.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.hist(low_speed, 15, alpha=0.5, label='high flow speed')
    plt.hist(best_speed, 15, alpha=0.5, label='GA flow speed')
    plt.legend(loc='upper right')
    plt.title("Speed Histogram")
    plt.savefig(os.path.join(plots_dir, "histogramSpeed.png"), dpi=200, bbox_inches="tight")
    plt.close()

    high_flow_throughput = [x*y for x, y in zip(high_flow, low_speed)]

    # best_throughput = [x*y for x, y in zip(best_flows, best_speed)]
    best_throughput = [x*y for x, y in zip(best_flows[0, :, 0], best_speed)]

    plt.hist(high_flow_throughput, 15, alpha=0.5, label='high flow throughput')
    plt.hist(best_throughput, 15, alpha=0.5, label='GA flow throughput')
    plt.legend(loc='upper right')
    plt.title("Speed Histogram")
    plt.savefig(os.path.join(plots_dir, "histogramTh.png"), dpi=200, bbox_inches="tight")
    plt.close()


def hcm_los_basic_freeway_from_density_km(density_pc_per_km_ln: np.ndarray) -> np.ndarray:
    """HCM basic freeway segment LOS using density in pc/km/ln."""
    d = np.asarray(density_pc_per_km_ln, dtype=float)
    los = np.full(d.shape, "F", dtype="<U1")
    los[(d >= 0) & (d <= 7)] = "A"
    los[(d > 7) & (d <= 11)] = "B"
    los[(d > 11) & (d <= 16)] = "C"
    los[(d > 16) & (d <= 22)] = "D"
    los[(d > 22) & (d <= 28)] = "E"
    los[d > 28] = "F"
    return los


def density_from_flow_speed(flow_pc_per_h_per_ln: np.ndarray, speed_km_per_h: np.ndarray) -> np.ndarray:
    """k = q / v, returns density pc/km/ln."""
    q = np.asarray(flow_pc_per_h_per_ln, dtype=float)
    v = np.asarray(speed_km_per_h, dtype=float)
    v = np.maximum(v, 1e-6)  # avoid divide-by-zero
    return q / v


if __name__ == "__main__":
    main()
