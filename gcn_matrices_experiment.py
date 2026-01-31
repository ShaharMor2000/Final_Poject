"""GCN - matrices experiment.ipynb

Original file is located at
    https://colab.research.google.com/drive/1Zu-7T7vZj7JoUAjGoUn1610O8jP0y9z3

(GCN) - Graph-based spatiotemporal forecasting

This version trains a graph-based model that combines:
(1) Graph convolution (message passing over sensor graph) for spatial dependencies.
(2) LSTM for temporal dependencies.
The model predicts SPEED from FLOW (Flow -> Speed).

All outputs are saved under runs/model3_gcn/ with the same structure used in Models 1-2:
weights/, logs/, outputs/, plots/.
"""

import typing
from dataclasses import dataclass
import os
import haversine
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import timeseries_dataset_from_array


def create_distance_matrix(a):
    # Get the coordinates of each station from the metadata dataframe.
    longitude = metadata.set_index("ID")["Longitude"].to_dict()
    latitude = metadata.set_index("ID")["Latitude"].to_dict()
    coordinates = {id: (latitude[id], longitude[id]) for id in a.columns}
    # Initialize the distance matrix.
    distance_matrix = np.zeros((len(a.columns), len(a.columns)))

    # Iterate over each pair of stations and calculate the distance.
    for i, index in enumerate(a.columns):
        for j, s_index in enumerate(a.columns):
            station_1_id = index
            station_2_id = s_index
            station_1_coordinates = coordinates[station_1_id]
            station_2_coordinates = coordinates[station_2_id]
            distance_matrix[i][j] = haversine.haversine(
                station_1_coordinates, station_2_coordinates
            )

    return distance_matrix


def create_tf_dataset(
    data_array: np.ndarray,
    target_array: np.ndarray,
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    shuffle=True,
    multi_horizon=True,
):
    """Creates tensorflow dataset from numpy array.

    This function creates a dataset where each element is a tuple `(inputs, targets)`.
    `inputs` is a Tensor
    of shape `(batch_size, input_sequence_length, num_routes, 1)` containing
    the `input_sequence_length` past values of the timeseries for each node.
    `targets` is a Tensor of shape `(batch_size, forecast_horizon, num_routes)`
    containing the `forecast_horizon`
    future values of the timeseries for each node.

    Args:
        data_array: np.ndarray with shape `(num_time_steps, num_routes)`
        input_sequence_length: Length of the input sequence (in number of timesteps).
        forecast_horizon: If `multi_horizon=True`, the target will be the values of the timeseries for 1 to
            `forecast_horizon` timesteps ahead. If `multi_horizon=False`, the target will be the value of the
            timeseries `forecast_horizon` steps ahead (only one value).
        batch_size: Number of timeseries samples in each batch.
        shuffle: Whether to shuffle output samples, or instead draw them in chronological order.
        multi_horizon: See `forecast_horizon`.

    Returns:
        A tf.data.Dataset instance.
    """

    inputs = timeseries_dataset_from_array(
        np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    target_offset = (
        input_sequence_length
        if multi_horizon
        else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = timeseries_dataset_from_array(
        target_array[target_offset:],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()


def compute_adjacency_matrix(
    route_distances: np.ndarray, sigma2: float, epsilon: float
):
    """Computes the adjacency matrix from distances matrix.

    It uses the formula in https://github.com/VeritasYin/STGCN_IJCAI-18#data-preprocessing to
    compute an adjacency matrix from the distance matrix.
    The implementation follows that paper.

    Args:
        route_distances: np.ndarray of shape `(num_routes, num_routes)`. Entry `i,j` of this array is the
            distance between roads `i,j`.
        sigma2: Determines the width of the Gaussian kernel applied to the square distances matrix.
        epsilon: A threshold specifying if there is an edge between two nodes. Specifically, `A[i,j]=1`
            if `np.exp(-w2[i,j] / sigma2) >= epsilon` and `A[i,j]=0` otherwise, where `A` is the adjacency
            matrix and `w2=route_distances * route_distances`

    Returns:
        A boolean graph adjacency matrix.
    """
    num_routes = route_distances.shape[0]
    # route_distances = route_distances / 10000.0
    w2, w_mask = (
        route_distances * route_distances,
        np.ones([num_routes, num_routes]) - np.identity(num_routes),
    )
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask


@dataclass
class GraphInfo:
    # edges = (src_nodes, dst_nodes) as python lists of ints
    edges: typing.Tuple[typing.List[int], typing.List[int]]
    num_nodes: int

    def to_dict(self) -> dict:
        src, dst = self.edges
        return {
            "edges": (list(src), list(dst)),
            "num_nodes": int(self.num_nodes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GraphInfo":
        src, dst = d["edges"]
        return cls(edges=(list(src), list(dst)), num_nodes=int(d["num_nodes"]))


class GraphConv(layers.Layer):
    def __init__(
        self,
        in_feat,
        out_feat,
        graph_info: GraphInfo,
        aggregation_type="mean",
        combination_type="concat",
        activation: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type

        self.weight = tf.Variable(
            initial_value=keras.initializers.glorot_uniform()(
                shape=(in_feat, out_feat), dtype="float32"
            ),
            trainable=True,
        )
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbour_representations: tf.Tensor):
        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)

        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )

        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_nodes_representation(self, features: tf.Tensor):
        """Computes each node's representation.

        The nodes' representations are obtained by multiplying the features tensor with
        `self.weight`. Note that
        `self.weight` has shape `(in_feat, out_feat)`.

        Args:
            features: Tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features: tf.Tensor):
        neighbour_representations = tf.gather(features, self.graph_info.edges[1])
        aggregated_messages = self.aggregate(neighbour_representations)
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        return self.activation(h)

    def call(self, features: tf.Tensor):
        """Forward pass.

        Args:
            features: tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)

class LSTMGC(layers.Layer):
    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        lstm_units: int,
        input_seq_len: int,
        output_seq_len: int,
        graph_info: GraphInfo,
        graph_conv_params: typing.Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "concat",
                "activation": None,
            }

        # Save init args for serialization
        self.in_feat = int(in_feat)
        self.out_feat = int(out_feat)
        self.lstm_units = int(lstm_units)
        self.input_seq_len = int(input_seq_len)
        self.output_seq_len = int(output_seq_len)
        self.graph_conv_params = dict(graph_conv_params)
        self.graph_info = graph_info

        # Sub-layers
        self.graph_conv = GraphConv(
            self.in_feat, self.out_feat, self.graph_info, **self.graph_conv_params
        )
        self.lstm = layers.LSTM(self.lstm_units, activation="relu")
        self.dense = layers.Dense(self.output_seq_len)

    def call(self, inputs):
        # inputs: (batch, input_seq_len, num_nodes, in_feat)

        # (num_nodes, batch, seq, in_feat)
        x = tf.transpose(inputs, [2, 0, 1, 3])

        # (num_nodes, batch, seq, out_feat)
        gcn_out = self.graph_conv(x)

        shape = tf.shape(gcn_out)
        num_nodes = shape[0]
        batch_size = shape[1]
        seq_len = shape[2]
        out_feat = shape[3]

        # (batch*num_nodes, seq, out_feat)
        gcn_out = tf.reshape(gcn_out, (batch_size * num_nodes, seq_len, out_feat))

        # (batch*num_nodes, lstm_units)
        lstm_out = self.lstm(gcn_out)

        # (batch*num_nodes, output_seq_len)
        dense_out = self.dense(lstm_out)

        # (batch, output_seq_len, num_nodes)
        dense_out = tf.reshape(dense_out, (num_nodes, batch_size, self.output_seq_len))
        return tf.transpose(dense_out, [1, 2, 0])

def train_model(
    model: keras.models.Model,
    learning_rate: float,
    num_epochs: int,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    output_path: str,
) -> None:
    model.compile(
        # FIX Change optimizer to AdamW from RMSprop
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
        loss=keras.losses.MeanAbsoluteError(),
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        # add restore best weights to early stopping
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    )

    model.save_weights(output_path)

if __name__ == "__main__":
    # Previous model (FLOW -> FLOW):
    # MODEL_PATH: str = "models/flow_similarity_matrix_gnn_model.weights.h5"
    # New model (FLOW -> SPEED):
    RUNS_DIR: str = os.path.join("runs", "model3_gcn")
    WEIGHTS_DIR: str = os.path.join(RUNS_DIR, "weights")
    LOGS_DIR: str = os.path.join(RUNS_DIR, "logs")
    OUTPUTS_DIR: str = os.path.join(RUNS_DIR, "outputs")
    PLOTS_DIR: str = os.path.join(RUNS_DIR, "plots")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    MODEL_PATH: str = os.path.join(
        WEIGHTS_DIR, "flow_to_speed_gnn_model.weights.h5"
    )
    METRICS_PATH: str = os.path.join(OUTPUTS_DIR, "metrics_summary.csv")
    DATA_CSV: str = os.path.join("data", "data.csv")
    METADATA_CSV: str = os.path.join("data", "metadata.txt")
    ADJ_MATRIX_CSV: str = os.path.join("data", "adj_matrix_directed.csv")

    # This is a spatio-temporal GNN: GraphConv for message passing + LSTM for time dynamics.
    # Input: Flow time-series per node (each station/sensor)
    # Target: Speed time-series per node
    # Output: Speed forecast with shape (batch, forecast_horizon, num_nodes)

    data = pd.read_csv(DATA_CSV)
    metadata = pd.read_csv(METADATA_CSV, sep="\t")
    adj = pd.read_csv(ADJ_MATRIX_CSV)

    flow_df = data.pivot(index="Timestamp", columns="ID", values="Flow")
    speed_df = data.pivot(index="Timestamp", columns="ID", values="Speed")

    flow_df = flow_df.fillna(0)
    speed_df = speed_df.fillna(0)

    constant_zeros = speed_df.columns[speed_df.isin([0]).all()]

    speed_df.drop(constant_zeros, axis=1, inplace=True)
    flow_df = flow_df.loc[:, ~flow_df.columns.isin(constant_zeros)]

    cols = []
    for i in range(1, len(adj.columns)):
        cols.append(int(adj.columns[i]))

    data = data[data["ID"].isin(cols)]

    adj = adj.drop(columns=[str(col) for col in cols if col not in flow_df.columns])
    columns = [str(col) for col in cols if col not in flow_df.columns]
    adj = adj[~adj["ID"].isin(int(col) for col in columns)]
    adj = adj.iloc[:, :-1]
    adj = adj.set_index("ID")
    adj = adj[:-1]

    missing_columns = [col for col in flow_df.columns if str(col) not in adj.columns]
    flow_df.drop(missing_columns, axis=1, inplace=True)
    speed_df.drop(missing_columns, axis=1, inplace=True)

    # Sanity check: after filtering, Flow and Speed must have the same number of columns
    if flow_df.shape[1] != speed_df.shape[1]:
        print("Warning: Flow and Speed columns count mismatch after filtering.")
    assert flow_df.shape[1] == speed_df.shape[1], (
        "Error: Flow and Speed must have the same number of columns after filtering."
    )
    distance_matrix = create_distance_matrix(flow_df)

    flow_train = flow_df[0:6000]
    flow_val = flow_df[6000:7000]
    flow_test = flow_df[7000:]

    speed_train = speed_df[0:6000]
    speed_val = speed_df[6000:7000]
    speed_test = speed_df[7000:]

    adj_similar = np.corrcoef(speed_train, rowvar=False)

    for i in range(len(adj_similar)):
        adj_similar[i][i] = 0

    # Originaly it was 64
    batch_size = 4
    # batch_size = 64
    input_sequence_length = 24
    forecast_horizon = 1
    multi_horizon = False

    # Old code (FLOW -> FLOW)
    # train_dataset = create_tf_dataset(
    #     flow_train,
    #     flow_train,
    #     input_sequence_length,
    #     forecast_horizon,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     multi_horizon=multi_horizon,
    # )
    # New: input = Flow, target = Speed (FLOW -> SPEED)
    train_dataset = create_tf_dataset(
        flow_train,
        speed_train,
        input_sequence_length,
        forecast_horizon,
        batch_size=batch_size,
        shuffle=False,
        multi_horizon=multi_horizon,
    )

    # Old code (FLOW -> FLOW)
    # val_dataset = create_tf_dataset(
    #     flow_val,
    #     flow_val,
    #     input_sequence_length,
    #     forecast_horizon,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     multi_horizon=multi_horizon,
    # )
    # New: input = Flow, target = Speed (FLOW -> SPEED)
    val_dataset = create_tf_dataset(
        flow_val,
        speed_val,
        input_sequence_length,
        forecast_horizon,
        batch_size=batch_size,
        shuffle=False,
        multi_horizon=multi_horizon,
    )

    # Old code (FLOW -> FLOW)
    # test_dataset = create_tf_dataset(
    #     flow_test,
    #     flow_test,
    #     input_sequence_length,
    #     forecast_horizon,
    #     batch_size=speed_test.shape[0],
    #     shuffle=False,
    #     multi_horizon=multi_horizon,
    # )
    # New: input = Flow, target = Speed (FLOW -> SPEED)
    test_dataset = create_tf_dataset(
        flow_test,
        speed_test,
        input_sequence_length,
        forecast_horizon,
        batch_size=speed_test.shape[0],
        shuffle=False,
        multi_horizon=multi_horizon,
    )

    sigma2 = 0.1
    epsilon = 0.5
    adjacency_matrix = compute_adjacency_matrix(distance_matrix, sigma2, epsilon)

    node_indices, neighbor_indices = np.where(adj_similar > 0.98)
    # Old code: number of nodes derived from the adjacency matrix.
    # graph = GraphInfo(
    #     edges=(node_indices.tolist(), neighbor_indices.tolist()),
    #     num_nodes=adj.shape[0],
    # )
    # New: number of nodes derived from the filtered Flow/Speed columns (safer).
    preferred_num_nodes = len(flow_df.columns)
    if adj.shape[0] != preferred_num_nodes:
        print(
            "Warning: adj node count doesn't match Flow/Speed columns; using Flow columns."
        )
    graph = GraphInfo(
        edges=(node_indices.tolist(), neighbor_indices.tolist()),
        num_nodes=preferred_num_nodes,
    )
    print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")

    in_feat = 1
    # epochs = 100
    epochs = 20
    input_sequence_length = 24
    forecast_horizon = 1
    multi_horizon = False
    out_feat = 10
    lstm_units = 64
    graph_conv_params = {
        "aggregation_type": "mean",
        "combination_type": "concat",
        "activation": None,
    }

    st_gcn = LSTMGC(
        in_feat,
        out_feat,
        lstm_units,
        input_sequence_length,
        forecast_horizon,
        graph,
        graph_conv_params,
    )
    inputs = layers.Input((input_sequence_length, graph.num_nodes, in_feat))
    outputs = st_gcn(inputs)

    model = keras.models.Model(inputs, outputs)
    train_model(
         model,
         learning_rate=3e-4,
         num_epochs=epochs,
         train_dataset=train_dataset,
         val_dataset=val_dataset,
         output_path=MODEL_PATH,
     )

    ####### START TEST ###########
    x_test, y_test = next(test_dataset.as_numpy_iterator())
    y_pred = model.predict(x_test)
    # Old code used fixed sizes (left commented to avoid deletion).
    # y_pred = y_pred.reshape((1328, 988))
    # y_test = y_test.reshape((1328, 988))
    # New: shape derived from actual model output.
    num_samples = y_pred.shape[0]  # number of test samples
    num_nodes = y_pred.shape[-1]  # number of nodes in output
    y_pred = y_pred.reshape((num_samples, num_nodes))
    y_test = y_test.reshape((num_samples, num_nodes))

    mape = []
    for i in range(0, y_pred.shape[1]):
        mape.append(mean_absolute_percentage_error(y_pred[:, i], y_test[:, i]))

    mape_score = np.mean(mape)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared:", r_squared)
    print("Mean Absolute Error:", mae)
    print("Mean Absolute Percentage Error:", mape_score)

    metrics_summary = pd.DataFrame(
        [
            {
                "mse": mse,
                "r2": r_squared,
                "mae": mae,
                "mape": mape_score,
                "num_samples": num_samples,
                "num_nodes": num_nodes,
            }
        ]
    )
    metrics_summary.to_csv(METRICS_PATH, index=False)

    mape = []
    for i in range(0, y_pred.shape[1]):
        mape.append(mean_absolute_percentage_error(y_pred[:, i], y_test[:, i]))

    plt.figure(figsize=(20, 6))
    plt.plot(y_pred[:, 900], label="Predicted", color="orange")
    plt.plot(y_test[:, 900], label="Actual", color="blue")
    plt.xlabel("Time Steps")
    # Update Y-axis label to reflect Speed.
    plt.ylabel("Speed")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, "prediction_vs_actual_node_900.png"))
    plt.show()

    ####### END TEST ###########
    errors = {}
    mse_arr = []
    r = {}
    r_arr = []

    n_outputs = y_test.shape[1]
    for i in range(n_outputs):
        y_test_ind = y_test[:, i]
        y_pred_ind = y_pred[:, i]

        mse = mean_squared_error(y_test_ind, y_pred_ind)
        r2 = r2_score(y_test_ind, y_pred_ind)

        mse_arr.append(mse)
        errors[i] = mse

        r_arr.append(r2)
        r[i] = r2

    count_high_r2 = sum(1 for i in r if r[i] > 0.9)
    print("Number of outputs with R^2 > 0.9:", count_high_r2)

    print("This model is FLOW -> SPEED: input Flow, target Speed.")

# Updated model path to reflect the FLOW → SPEED task (previous path kept as a comment).
# Switched datasets to use Flow as input and Speed as target for Train / Validation / Test.
# Added checks to ensure Flow and Speed columns remain aligned after filtering.
# Node count is now derived from Flow/Speed columns instead of the adjacency matrix.
# Evaluation reshaping is now dynamic (no hard-coded dimensions).
# Plots and logs were updated to clearly indicate Speed prediction.
