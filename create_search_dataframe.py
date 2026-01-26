# -*- coding: utf-8 -*-
"""
Create speed_search2test.csv from train/test.

Original logic preserved.
Only formatting + minimal fixes to make it runnable:
- define `hour` (4 values)
- assign Hour column with correct length after repeat
"""

from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd


TS_FMT = "%m/%d/%Y %H:%M:%S"


def extract_hour(timestamp: str) -> int:
    datetime_obj = datetime.strptime(timestamp, TS_FMT)
    return datetime_obj.hour


def extract_dayofweek(timestamp: str) -> str:
    datetime_obj = datetime.strptime(timestamp, TS_FMT)
    return datetime_obj.strftime("%A")


def main() -> None:
    # Load data
    metadata = pd.read_csv("data/metadata_with_embeddings.csv")
    X_train = pd.read_csv("data/train.csv")
    X_test = pd.read_csv("data/test.csv")

    # Time features
    X_train["Hour"] = X_train["Timestamp"].apply(extract_hour)
    X_test["Hour"] = X_test["Timestamp"].apply(extract_hour)

    X_train["dayofweek"] = X_train["Timestamp"].apply(extract_dayofweek)
    X_test["dayofweek"] = X_test["Timestamp"].apply(extract_dayofweek)

    # Find the highest flow for each id
    max_flows = X_train.groupby("ID")["Flow1"].max()

    # one row per station
    df = X_train.drop_duplicates(subset=["ID"])

    traffic_search = pd.DataFrame()

    # This replaces the "repeat 4 times" idea in a runnable way
    hour = np.array([5, 10, 15, 20], dtype=int)

    # Iterate over each station
    for _, station in df.iterrows():
        max_flow = max_flows[station.ID]
        lanes = station.Lanes

        station_df = df[df["ID"] == station.ID]

        # Repeat station rows (max_flow-1) times
        temp_df = pd.DataFrame(np.repeat(station_df.values, int(max_flow) - 1, axis=0))
        temp_df.columns = station_df.columns

        # Assign Flow1 = 1..max_flow-1
        temp_df["Flow1"] = np.repeat(range(1, int(max_flow)), 1)
        temp_df["Flow"] = temp_df["Flow1"].apply(lambda x: x * lanes)

        # Repeat each row 4 times
        temp_df = pd.DataFrame(np.repeat(temp_df.values, 4, axis=0))
        temp_df.columns = station_df.columns

        # IMPORTANT: Hour must match number of rows (= 4*(max_flow-1))
        temp_df["Hour"] = np.tile(hour, int(max_flow) - 1)

        traffic_search = pd.concat([traffic_search, temp_df], ignore_index=True)

    # Duplicate features for model input
    traffic_search["p1_Flow"] = traffic_search["Flow"]
    traffic_search["p2_Flow"] = traffic_search["Flow"]
    traffic_search["p3_Flow"] = traffic_search["Flow"]

    traffic_search["p1_Flow1"] = traffic_search["Flow1"]
    traffic_search["p2_Flow1"] = traffic_search["Flow1"]
    traffic_search["p3_Flow1"] = traffic_search["Flow1"]

    # Save
    traffic_search.to_csv("data/speed_search2test.csv", index=False)
    print(f"Saved {len(traffic_search):,} rows to data/speed_search2test.csv")


if __name__ == "__main__":
    main()
