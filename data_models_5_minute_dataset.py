# -*- coding: utf-8 -*-
"""
Data_models_5_minute_dataset.py

Original notebook:
https://colab.research.google.com/drive/1u_eilnK3tYDv4oJyNeTE871bcpH3zfcb


Models trained/evaluated in this script (via sklearn Pipelines):
1) MLPRegressor (Neural Network)  -> hidden_layer_sizes=(20,10,5)
2) KNeighborsRegressor (KNN)      -> n_neighbors=25 (and also a sweep: 25,50,75,100; plus fixed 25 saved)
3) SGDRegressor                   -> linear model trained with SGD (alpha=0.001, constant LR)
4) SVR (Support Vector Regression)-> RBF kernel
5) KNNRegressorAnnoy              -> Approximate KNN using Annoy (k=50)
6) DecisionTreeRegressor          -> max_depth=15
7) RandomForestRegressor          -> n_estimators=10 with max_depth sweep [5,8,10,12,15,18,20]
8) LightGBM Regressor (LGBMRegressor) -> (a) n_estimators sweep [50..1000] at max_depth=15
                                         (b) max_depth sweep [5..20] at n_estimators=100
                                         (c) final block: n_estimators=50, max_depth=15
9) XGBRegressor (XGBoost)         -> n_estimators=100, max_depth=15

All models use a Pipeline with ChooseFeatures() which selects columns,
one-hot encodes 'maxspeed', and applies StandardScaler.

"""
from __future__ import annotations

import os
import pickle
import statistics  
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns 

try:
    from annoy import AnnoyIndex
except ImportError as e:
    raise ImportError(
        "Missing dependency: annoy\n"
        "Install with: pip install annoy"
    ) from e

# sklearn
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    recall_score,
    precision_score,
)

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError(
        "Missing dependency: xgboost\n"
        "Install with: pip install xgboost"
    ) from e

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "Missing dependency: lightgbm\n"
        "Install with: pip install lightgbm"
    ) from e

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
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_selected = X.loc[:, self.feature_cols]

        # X_selected = encode(X_selected, 'Hour')
        # X_selected = encode(X_selected, 'dayofweek')
        X_selected = encode(X_selected, "maxspeed")

        scaler = StandardScaler()
        X_selected = scaler.fit_transform(X_selected)

        return X_selected


# ----------------------------
# Annoy KNN Regressor (same logic)
# ----------------------------
class KNNRegressorAnnoy(BaseEstimator, RegressorMixin):
    def __init__(self, k=5, n_trees=10):
        self.k = k
        self.n_trees = n_trees
        self.annoy_index = None

    def fit(self, X, y):
        dim = X.shape[1]
        self.annoy_index = AnnoyIndex(dim, "euclidean")
        for i, x in enumerate(X):
            self.annoy_index.add_item(i, x)
        self.annoy_index.build(self.n_trees)
        self.y_train = y
        return self

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i, x in enumerate(X):
            indices = self.annoy_index.get_nns_by_vector(x, self.k)
            y_pred[i] = np.mean(self.y_train[indices])
        return y_pred



def main() -> None:
    X_train = pd.read_csv("data/train.csv")
    X_test = pd.read_csv("data/test.csv")
    traffic_search = pd.read_csv("data/speed_search_with_speed.csv")

    # Time features (same logic)
    X_train["Hour"] = X_train["Timestamp"].apply(lambda x: extract_hour(x))
    X_test["Hour"] = X_test["Timestamp"].apply(lambda x: extract_hour(x))

    X_train["dayofweek"] = X_train["Timestamp"].apply(lambda x: extract_dayofweek(x))
    X_test["dayofweek"] = X_test["Timestamp"].apply(lambda x: extract_dayofweek(x))

    # Targets (same logic)
    y_train = X_train[["Speed1"]]
    y_test = X_test[["Speed1"]]

    # Feature columns (same overwritten assignments as original)
    feature_cols_all = ["Flow","Flow1","p1_Flow","p2_Flow","p3_Flow","p1_Flow1","p2_Flow1","p3_Flow1","Hour","Lanes","p1_Speed1","p2_Speed1","p3_Speed1","maxspeed","dayofweek","FFS"]

    feature_cols_all = ["Flow","Flow1","p1_Flow","p2_Flow","p3_Flow","p1_Flow1","p2_Flow1","p3_Flow1","Hour","Lanes","p1_Speed1","p2_Speed1","p3_Speed1","maxspeed","dayofweek","FFS"]

    feature_cols_all = ["Flow","Flow1","p1_Flow","p2_Flow","p3_Flow","p1_Flow1","p2_Flow1","p3_Flow1","Hour","Lanes","maxspeed"]

    feature_cols = []
    feature_cols.append(feature_cols_all)

    for feature in feature_cols_all:
        X_train[feature].fillna(0, inplace=True)
        X_test[feature].fillna(0, inplace=True)

    mean_speeds = X_train.loc[X_train["Hour"] == 0].groupby("ID")["Speed1"].mean()

    X_train["FFS"] = X_train["ID"].map(mean_speeds)
    X_test["FFS"] = X_test["ID"].map(mean_speeds)
    traffic_search["FFS"] = traffic_search["ID"].map(mean_speeds)

    # ----------------------------
    # Model 1: MLPRegressor 
    # ----------------------------
    score = dict()
    neighbors = [15]

    for fc in feature_cols:
        for n in neighbors:
            pipeline = Pipeline(
                steps=[
                    ("choose_features", ChooseFeatures(fc)),
                    ("mlp_regressor", MLPRegressor(hidden_layer_sizes=(20, 10, 5))),
                ]
            )

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = str(n) + "r2"
            mse = str(n) + "mse"

            score[mse] = mean_squared_error(y_test, y_pred)
            score[r2] = r2_score(y_test, y_pred)

            print("depth:" + str(n))
            print(score[r2], score[mse])

            # Save model
            os.makedirs("models", exist_ok=True)
            file_path = "models/" + "speed_data_set" + "mlp" + str(n) + "model.pkl"
            with open(file_path, "wb") as file:
                pickle.dump(pipeline, file)
            print("Model saved successfully.")


    # ----------------------------
    # Prepare X_predict_speed
    # ----------------------------
    X_predict_speed = pd.read_csv("data/speed_search_with_speed.csv")

    missing = set(set(X_train.maxspeed.unique()) - set(X_predict_speed.maxspeed.unique()))
    i = 0
    for value in missing:
        X_predict_speed.at[i, "maxspeed"] = value
        i += 1

    X_predict_speed["FFS"] = X_predict_speed["ID"].map(mean_speeds)
    X_predict_speed = X_predict_speed[X_predict_speed["ID"] == 313114]

    missing = set(set(X_train.maxspeed.unique()) - set(X_predict_speed.maxspeed.unique()))
    i = 0
    for value in missing:
        X_predict_speed.at[i, "maxspeed"] = value
        i += 1

    missing_days = set(set(X_train.dayofweek.unique()) - set(X_predict_speed.dayofweek.unique()))
    i = 0
    for value in missing_days:
        X_predict_speed.at[i, "dayofweek"] = value
        i += 1

    for feature in feature_cols_all:
        X_predict_speed[feature].fillna(0, inplace=True)

    # ----------------------------
    # KNN predict
    # ----------------------------
    pipeline = Pipeline(
        steps=[
            ("choose_features", ChooseFeatures(feature_cols[0])),
            ("KNN", KNeighborsRegressor(n_neighbors=25)),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred_speed = pipeline.predict(X_predict_speed)

    X_predict_speed["speed_pred"] = y_pred_speed

    flows = [772, 1214, 1692, 2024, 2208]
    fl = []
    for flow in flows:
        fl.append(int(flow / 12))

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

    X_predict_speed.to_csv("data/neural_network_predicted_speeds_dataframe.csv", index=False)

    # ----------------------------
    # SGDRegressor block 
    # ----------------------------
    score = dict()
    neighbors = [15]

    for fc in feature_cols:
        for n in neighbors:
            pipeline = Pipeline(
                steps=[
                    ("choose_features", ChooseFeatures(fc)),
                    ("SGD_regressor", SGDRegressor(alpha=0.001, learning_rate="constant", eta0=0.01, max_iter=1000, tol=1e-3)),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = str(n) + "r2"
            mse = str(n) + "mse"

            score[mse] = mean_squared_error(y_test, y_pred)
            score[r2] = r2_score(y_test, y_pred)

            print("depth:" + str(n))
            print(score[r2], score[mse])

    # ----------------------------
    # SVR block 
    # ----------------------------
    score = dict()
    neighbors = [15]

    for fc in feature_cols:
        for n in neighbors:
            pipeline = Pipeline(
                steps=[
                    ("choose_features", ChooseFeatures(fc)),
                    ("SVR", SVR(kernel="rbf")),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = str(n) + "r2"
            mse = str(n) + "mse"

            score[mse] = mean_squared_error(y_test, y_pred)
            score[r2] = r2_score(y_test, y_pred)

            print("depth:" + str(n))
            print(score[r2], score[mse])

    # ----------------------------
    # Annoy KNN block 
    # ----------------------------
    score = dict()
    neighbors = [15]

    for fc in feature_cols:
        for n in neighbors:
            pipeline = Pipeline(
                steps=[
                    ("choose_features", ChooseFeatures(fc)),
                    ("KNN", KNNRegressorAnnoy(k=50)),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = str(n) + "r2"
            mse = str(n) + "mse"

            score[mse] = mean_squared_error(y_test, y_pred)
            score[r2] = r2_score(y_test, y_pred)

            print("depth:" + str(n))
            print(score[r2], score[mse])

    try:
        _ = pipeline.get_feature_names_out()
    except Exception:
        pass

    # ----------------------------
    # KNN neighbors sweep
    # ----------------------------
    score = dict()
    neighbors = [25, 50, 75, 100]

    for fc in feature_cols:
        for n in neighbors:
            pipeline = Pipeline(
                steps=[
                    ("choose_features", ChooseFeatures(fc)),
                    ("KNN", KNeighborsRegressor(n_neighbors=n)),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = str(n) + "r2"
            mse = str(n) + "mse"

            score[mse] = mean_squared_error(y_test, y_pred)
            score[r2] = r2_score(y_test, y_pred)

            print("depth:" + str(n))
            print(score[r2], score[mse])

    # ----------------------------
    # KNN fixed 25 + save 
    # ----------------------------
    score = dict()
    neighbors = [25]

    for fc in feature_cols:
        for n in neighbors:
            pipeline = Pipeline(
                steps=[
                    ("choose_features", ChooseFeatures(fc)),
                    ("KNN", KNeighborsRegressor(n_neighbors=n)),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = str(n) + "r2"
            mse = str(n) + "mse"

            score[mse] = mean_squared_error(y_test, y_pred)
            score[r2] = r2_score(y_test, y_pred)

            print("depth:" + str(n))
            print(score[r2], score[mse])

            file_path = "data/" + "speed_data_set" + "knn" + str(n) + "model.pkl"
            with open(file_path, "wb") as file:
                pickle.dump(pipeline, file)
            print("Model saved successfully.")

    # ----------------------------
    # Decision Tree 
    # ----------------------------
    score = dict()
    neighbors = [15]

    for fc in feature_cols:
        for n in neighbors:
            pipeline = Pipeline(
                steps=[
                    ("choose_features", ChooseFeatures(fc)),
                    ("decision_tree", DecisionTreeRegressor(max_depth=15)),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = str(n) + "r2"
            mse = str(n) + "mse"

            score[mse] = mean_squared_error(y_test, y_pred)
            score[r2] = r2_score(y_test, y_pred)

            print("depth:" + str(n))
            print(score[r2], score[mse])

    # ----------------------------
    # Random Forest
    # ----------------------------
    score = dict()
    neighbors = [5, 8, 10, 12, 15, 18, 20]

    for fc in feature_cols:
        for n in neighbors:
            pipeline = Pipeline(
                steps=[
                    ("choose_features", ChooseFeatures(fc)),
                    ("random_forest", RandomForestRegressor(max_depth=n, n_estimators=10)),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = str(n) + "r2"
            mse = str(n) + "mse"

            score[mse] = mean_squared_error(y_test, y_pred)
            score[r2] = r2_score(y_test, y_pred)

            print("depth:" + str(n))
            print(score[r2], score[mse])

    # ----------------------------
    # LightGBM (n_estimators sweep) 
    # ----------------------------
    score = dict()
    neighbors = [50, 100, 150, 200, 250, 500, 1000]

    for fc in feature_cols:
        for n in neighbors:
            pipeline = Pipeline(
                steps=[
                    ("choose_features", ChooseFeatures(fc)),
                    ("random_forest", lgb.LGBMRegressor(n_estimators=n, max_depth=15, random_state=42)),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = str(n) + "r2"
            mse = str(n) + "mse"

            score[mse] = mean_squared_error(y_test, y_pred)
            score[r2] = r2_score(y_test, y_pred)

            print("depth:" + str(n))
            print(score[r2], score[mse])

    # ----------------------------
    # LightGBM (max_depth sweep) 
    # ----------------------------
    score = dict()
    neighbors = [5, 8, 10, 12, 15, 18, 20]

    for fc in feature_cols:
        for n in neighbors:
            pipeline = Pipeline(
                steps=[
                    ("choose_features", ChooseFeatures(fc)),
                    ("random_forest", lgb.LGBMRegressor(n_estimators=100, max_depth=n, random_state=42)),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = str(n) + "r2"
            mse = str(n) + "mse"

            score[mse] = mean_squared_error(y_test, y_pred)
            score[r2] = r2_score(y_test, y_pred)

            print("depth:" + str(n))
            print(score[r2], score[mse])

    # ----------------------------
    # XGBRegressor 
    # ----------------------------
    score = dict()
    neighbors = [15]

    for fc in feature_cols:
        for n in neighbors:
            pipeline = Pipeline(
                steps=[
                    ("choose_features", ChooseFeatures(fc)),
                    ("random_forest", XGBRegressor(n_estimators=100, max_depth=15, random_state=42)),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = str(n) + "r2"
            mse = str(n) + "mse"

            score[mse] = mean_squared_error(y_test, y_pred)
            score[r2] = r2_score(y_test, y_pred)

            print("depth:" + str(n))
            print(score[r2], score[mse])

    # ----------------------------
    # LightGBM (final block, n_estimators=50) 
    # ----------------------------
    score = dict()
    neighbors = [50]

    for fc in feature_cols:
        for n in neighbors:
            pipeline = Pipeline(
                steps=[
                    ("choose_features", ChooseFeatures(fc)),
                    ("random_forest", lgb.LGBMRegressor(n_estimators=n, max_depth=15, random_state=42)),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = str(n) + "r2"
            mse = str(n) + "mse"

            score[mse] = mean_squared_error(y_test, y_pred)
            score[r2] = r2_score(y_test, y_pred)

            print("depth:" + str(n))
            print(score[r2], score[mse])

    # ----------------------------
    # Predict on traffic_search 
    # ----------------------------
    X_predict_speed = traffic_search
    X_predict_speed = traffic_search

    missing = set(set(X_train.maxspeed.unique()) - set(X_predict_speed.maxspeed.unique()))
    i = 0
    for value in missing:
        X_predict_speed.at[i, "maxspeed"] = value
        i += 1

    speeds = pipeline.predict(X_predict_speed)
    X_predict_speed["Pred_speed"] = speeds

    flows = [770, 1210, 1740, 2135]
    flows = [708, 1113, 1600, 1964, 2162]
    flows = [772, 1214, 1692, 2024, 2208]

    fl = []
    for flow in flows:
        fl.append(int(flow / 12))

    example = X_predict_speed[
        (X_predict_speed["ID"] == 313114)
        & (X_predict_speed["p1_Speed1"] == 60)
        & (X_predict_speed["Hour"] == 10)
    ]

    for n in fl:
        pred_speed = example[example["p1_Flow1"] == n]["Pred_speed"]
        print("flow: " + str(n))
        print("pred_speed:" + str(pred_speed))
        speed = pred_speed * 1.60934
        print("speed" + str(speed))
        utility = n * pred_speed
        print(utility)


if __name__ == "__main__":
    main()
