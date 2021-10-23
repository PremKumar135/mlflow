import argparse
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
# import argparse

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def main(alpha, l1_ratio):
     # Read the wine-quality csv file from the URL
    csv_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
    data = pd.read_csv(csv_url, sep=";")

    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    mlflow.sklearn.autolog()
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
        print(rmse, mae, r2)

        # tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme
        # if tracking_url_type !="file":
        #     mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        # else:
        #     mlflow.sklearn.log_model(lr, "model")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--alpha', default=0.5, type=float)
    args.add_argument('--l1ratio', default=0.5, type=float)
    parsed_args = args.parse_args()
    main(parsed_args.alpha, parsed_args.l1ratio)

    

