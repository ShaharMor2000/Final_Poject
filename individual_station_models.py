# -*- coding: utf-8 -*-
"""
Individual_station_models.py

Original notebook:
https://colab.research.google.com/drive/1U7-Oig1pOPq_gIBMyS2SHSvrxJOCn9Td

Models in this script:
- Per-station KNeighborsRegressor (n_neighbors=15)
- Per-station RandomForestRegressor (n_estimators=10, max_depth=6)

Preprocessing:
- Adds Hour and dayofweek from Timestamp
- One-hot encodes maxspeed
- StandardScaler on selected features
"""

import math
import statistics
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns  # kept from notebook
import pickle  # kept from notebook

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


TS_FMT = "%m/%d/%Y %H:%M:%S"


def extract_hour(timestamp: str) -> int:
    datetime_obj = datetime.strptime(timestamp, TS_FMT)
    return datetime_obj.hour


def extract_dayofweek(timestamp: str) -> str:
    datetime_obj = datetime.strptime(timestamp, TS_FMT)
    return datetime_obj.strftime("%A")


def encode(X_try: pd.DataFrame, var: str) -> pd.DataFrame:
    # Perform one-hot encoding on the Type column
    X_one_hot = pd.get_dummies(X_try[var], prefix="Type")

    # Concatenate the one-hot encoded features with the original dataframe
    X_encoded = pd.concat([X_try, X_one_hot], axis=1)

    # Drop the original Type column from the dataframe
    X_encoded.drop(var, axis=1, inplace=True)

    return X_encoded


class ChooseFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_cols, feature_cols_all):
        self.feature_cols = feature_cols
        self.feature_cols_all = feature_cols_all

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_selected = X.loc[:, self.feature_cols]

        for feature in self.feature_cols_all:
            X_selected[feature].fillna(0, inplace=True)

        # X_selected = encode(X_selected, 'Type')
        # X_selected = encode(X_selected, 'dayofweek')
        X_selected = encode(X_selected, "maxspeed")

        scaler = StandardScaler()
        X_selected = scaler.fit_transform(X_selected)

        return X_selected


def round_to_nearest_ten(number: float) -> int:
    if 85 < number < 125:
        return math.ceil(number / 10) * 10
    if number > 125:
        return 130
    if 85 > number:
        return 80
    return 0


def check_speed_range(flows, speeds, flow_speed_dict):
    results = []

    sorted_flows = sorted(flow_speed_dict.keys())
    min_flow_speed = flow_speed_dict[min(sorted_flows)]
    flow_speed_dict[0] = min_flow_speed
    sorted_flows = sorted(flow_speed_dict.keys())

    for i in range(len(flows)):
        flow = flows[i]
        speed = speeds[i]

        closest_lower_flow = None
        closest_upper_flow = None

        print(sorted_flows)

        for f in sorted_flows:
            if f <= flow:
                closest_lower_flow = f
            else:
                closest_upper_flow = f
                break

        if closest_lower_flow is not None and closest_upper_flow is not None:
            lower_speed = flow_speed_dict[closest_lower_flow]
            upper_speed = flow_speed_dict[closest_upper_flow]
            if lower_speed <= speed <= upper_speed:
                results.append(1)
            else:
                results.append(0)
        else:
            results.append(0)

    return results


def calculate_statistics(numbers):
    stats = {}
    stats["mean"] = statistics.mean(numbers)
    stats["median"] = statistics.median(numbers)
    stats["mode"] = statistics.mode(numbers)
    stats["standard_deviation"] = statistics.stdev(numbers)
    stats["variance"] = statistics.variance(numbers)
    stats["minimum"] = min(numbers)
    stats["maximum"] = max(numbers)
    return stats


def mean_of_absolute_values(lst):
    absolute_values = [abs(num) for num in lst]
    mean = sum(absolute_values) / len(absolute_values)
    return mean


def main():
    # ----------------------------
    # Load data
    # ----------------------------
    X_train = pd.read_csv("data/train.csv")
    X_test = pd.read_csv("data/test.csv")

    speed_search = pd.read_csv("data/speed_search.csv")
    flow_speed = speed_search  # keep original aliasing behavior

    capacity_df = pd.DataFrame()

    # ----------------------------
    # Time features
    # ----------------------------
    X_train["Hour"] = X_train["Timestamp"].apply(lambda x: extract_hour(x))
    X_test["Hour"] = X_test["Timestamp"].apply(lambda x: extract_hour(x))

    X_train["dayofweek"] = X_train["Timestamp"].apply(lambda x: extract_dayofweek(x))
    X_test["dayofweek"] = X_test["Timestamp"].apply(lambda x: extract_dayofweek(x))

    # Targets (kept)
    y_train = X_train[["Speed1"]]
    y_test = X_test[["Speed1"]]

    feature_cols_all = [
        "Flow",
        "Flow1",
        "p1_Flow",
        "p2_Flow",
        "p3_Flow",
        "p1_Flow1",
        "p2_Flow1",
        "p3_Flow1",
        "Hour",
        "Lanes",
        "maxspeed",
    ]

    extra_speed = ["p1_Speed1", "p2_Speed1", "p3_Speed1"]  # kept

    feature_cols = []
    feature_cols.append(feature_cols_all)

    # ----------------------------
    # HCM-based flow classes (same values)
    # ----------------------------
    flows = {}
    flows[120] = [840, 1320, 1840, 2200, 2400]
    flows[110] = [770, 1210, 1740, 2135, 2350]
    flows[100] = [700, 1100, 1600, 2065, 2300]
    flows[90] = [630, 990, 1440, 1955, 2250]

    rounded_flows = {}
    for key, value_list in flows.items():
        rounded_values = [math.ceil(val / 12) for val in value_list]
        rounded_flows[key] = rounded_values

    print("rounded_flows:", rounded_flows)

    # ----------------------------
    # Station class mapping
    # ----------------------------
    unique_ids = X_train["ID"].unique()
    station_class = {}

    for group_id in unique_ids:
        train_mask = X_train[X_train["ID"] == group_id]
        y_train_group = train_mask["Speed1"]

        average_speed = statistics.mean(y_train_group)
        average_speed = math.ceil(average_speed * 1.61)

        s_class = round_to_nearest_ten(average_speed)
        if s_class != 0:
            station_class[group_id] = s_class

    value_counts = Counter(station_class.values())
    for value, count in value_counts.items():
        print(f"Value: {value}, Count: {count}")

    for key, value in list(station_class.items()):
        if value == 130:
            station_class[key] = 120
        elif value == 80:
            station_class[key] = 90
        elif value is None:
            station_class[key] = 90

    # ----------------------------
    # Example usage block (kept)
    # ----------------------------
    flows_ex = [100, 150, 200]
    speeds_ex = [50, 60, 70]
    flow_speed_dict_ex = {120: 55, 180: 65, 220: 75}
    result_ex = check_speed_range(flows_ex, speeds_ex, flow_speed_dict_ex)
    print(result_ex)

    # ----------------------------
    # Per-station KNN models + capacity accuracy logic
    # ----------------------------
    mse_list = []
    r2_list = []
    hours = [5, 10, 15, 20]
    pipelines = {}

    for group_id in unique_ids:
        train_mask = X_train[X_train["ID"] == group_id]
        test_mask = X_test[X_test["ID"] == group_id]

        X_train_group = train_mask
        y_train_group = X_train_group["Speed1"]

        X_test_group = test_mask
        y_test_group = X_test_group["Speed1"]

        pipeline = Pipeline(
            steps=[
                ("choose_features", ChooseFeatures(feature_cols_all, feature_cols_all)),
                ("knn", KNeighborsRegressor(n_neighbors=15)),
            ]
        )

        s_class = station_class[group_id]

        new_rows = pd.concat([X_train_group.iloc[0]] * 5, axis=1).T.reset_index(drop=True)
        lanes = X_train_group.iloc[0]["Lanes"]

        pipeline.fit(X_train_group, y_train_group)

        flows_local = rounded_flows[s_class]

        columns_to_modify = ["Flow", "Flow1", "p1_Flow", "p2_Flow", "p3_Flow", "p1_Flow1", "p2_Flow1", "p3_Flow1"]
        for col in columns_to_modify:
            if "Flow1" in col:
                new_rows[col] = flows_local
            else:
                new_rows[col] = lanes * flows_local

        y_pred_speeds = pipeline.predict(new_rows)
        flow_speed_dict = {flow: speed for flow, speed in zip(flows_local, y_pred_speeds)}

        y_pred_group = pipeline.predict(X_test_group)
        x_test_flows = X_test_group["Flow1"]

        result = check_speed_range(x_test_flows, y_test_group, flow_speed_dict)
        print(result)

        mse = mean_squared_error(y_test_group, y_pred_group)
        r2 = r2_score(y_test_group, y_pred_group)

        mse_list.append(mse)
        r2_list.append(r2)

        pipelines[group_id] = pipeline

    print("Example pipeline:", pipelines.get(320660))

    # ----------------------------
    # Predict example for station 313114 (kept)
    # ----------------------------
    traffic_search = pd.read_csv("data/speed_search_with_speed.csv")
    X_predict_speed = traffic_search[traffic_search["ID"] == 313114]

    speed_pred = pipeline.predict(X_predict_speed)
    X_predict_speed["speed_pred"] = speed_pred

    flows = [772, 1214, 1692, 2024, 2208]
    fl = [int(flow / 12) for flow in flows]

    example = X_predict_speed[
        (X_predict_speed["ID"] == 313114)
        & (X_predict_speed["p1_Speed1"] == 70)
        & (X_predict_speed["Hour"] == 10)
    ]

    for n in fl:
        pred_speed = example[example["p1_Flow1"] == n]["speed_pred"]
        print("flow: " + str(n))
        print("pred_speed:" + str(pred_speed))
        speed = pred_speed * 1.60934
        print("speed" + str(speed))
        utility = n * pred_speed
        print(utility)

    print("r2_list length:", len(r2_list))
    print("mse_list length:", len(mse_list))
    print("mean abs r2:", mean_of_absolute_values(r2_list))

    print(calculate_statistics(mse_list))
    print(calculate_statistics(r2_list))

    # ----------------------------
    # Per-station RandomForest + capacity search (kept)
    # ----------------------------
    mse_list = []
    r2_list = []
    hours = [5, 10, 15, 20]

    unique_ids = X_train["ID"].unique()

    for group_id in unique_ids:
        train_mask = X_train[X_train["ID"] == group_id]
        test_mask = X_test[X_test["ID"] == group_id]

        X_train_group = train_mask
        y_train_group = X_train_group["Speed1"]
        X_test_group = test_mask
        y_test_group = X_test_group["Speed1"]

        pipeline = Pipeline(
            steps=[
                ("choose_features", ChooseFeatures(feature_cols_all, feature_cols_all)),
                ("random_forest", RandomForestRegressor(n_estimators=10, max_depth=6)),
            ]
        )

        pipeline.fit(X_train_group, y_train_group)
        y_pred_group = pipeline.predict(X_test_group)

        mse = mean_squared_error(y_test_group, y_pred_group)
        r2 = r2_score(y_test_group, y_pred_group)
        mse_list.append(mse)
        r2_list.append(r2)

        df = speed_search[speed_search["ID"] == group_id]

        for hour in hours:
            df_temp = df[df["Hour"] == hour]
            speed_pred = pipeline.predict(df_temp)

            df_temp["Speed1_pred"] = speed_pred
            df_temp["Flow1_Speed1"] = df_temp["Flow1"] * df_temp["Speed1_pred"]

            max_index = df_temp["Flow1_Speed1"].idxmax()
            max_row = df_temp.loc[max_index]

            max_index = df_temp["Flow1"].idxmax()
            max_row_flow = df_temp.loc[max_index]

            max_row["Max_Flow1"] = max_row_flow["Flow1"]

            capacity_df = capacity_df.append(max_row)

    print("MSE stats:", calculate_statistics(mse_list))
    print("R2 stats:", calculate_statistics(r2_list))

    flow_and_max_flow = pd.DataFrame().assign(
        Flow1=capacity_df["Flow1"],
        Max_Flow1=capacity_df["Max_Flow1"],
        Speed1=capacity_df["Speed1_pred"],
    )
    flow_and_max_flow["Max"] = flow_and_max_flow["Flow1"] == flow_and_max_flow["Max_Flow1"]
    print("Not max count:", len(flow_and_max_flow[flow_and_max_flow["Max"] == False]))

    print("Speed1 stats:", calculate_statistics(flow_and_max_flow["Speed1"]))

    capacity_df.to_csv("data/capacity_individuals2test.csv", index=False)
    print("Saved: data/capacity_individuals2test.csv")


if __name__ == "__main__":
    main()
