import xgboost as xgb
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Any

# Function to load and partition data for each client
def load_data(partition_id, num_partitions):
    train_path = os.getenv(f"CLIENT_{partition_id}_TRAINING_SET")
    test_path = os.getenv(f"CLIENT_{partition_id}_TESTING_SET")
    
    # Load data into pandas
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("LOADING TRAINING SET ", train_path, " LENGTH = ", len(train_df))
    print("LOADING TESTING SET ", test_path, " LENGTH = ", len(test_df))

    # Convert data to DMatrix format for XGBoost
    train_dmatrix = xgb.DMatrix(
        train_df.drop(columns="cardio"), label=train_df["cardio"]
    )
    valid_dmatrix = xgb.DMatrix(
        test_df.drop(columns="cardio"), label=test_df["cardio"]
    )
    
    num_train = train_df.shape[0]
    num_val = test_df.shape[0]
    
    return train_dmatrix, valid_dmatrix, num_train, num_val

# Replace keys (helper function for configuration)
def replace_keys(cfg):
    # Custom logic to adjust configuration if needed
    return cfg

# Function to aggregate evaluation metrics
def evaluate_metrics_aggregation(results):
    auc_values = [metrics["AUC"] for metrics in results]
    return {"mean_auc": sum(auc_values) / len(auc_values)}

# Function to set training and evaluation configuration
def config_func(rnd):
    return {"global_round": rnd}

def unflatten_dict(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Unflatten a dict with keys containing separators into a nested dict."""
    unflattened_dict: dict[str, Any] = {}
    separator: str = "."

    for key, value in flat_dict.items():
        parts = key.split(separator)
        d = unflattened_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return unflattened_dict