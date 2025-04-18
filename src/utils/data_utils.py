# src/utils/data_utils.py

from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
import mlflow

def load_and_split_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    return df, X, y

def perform_feature_selection(X_train, y_train, k=5):
    with mlflow.start_run(run_name="feature_selection", nested=True):
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X_train, y_train)
        selected = X_train.columns[selector.get_support()]
        mlflow.log_param("selected_features", list(selected))
        return selected
