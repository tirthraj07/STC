from typing import Dict
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg, FedMedian, FedAdam, FedProx
from flwr.common.logger import console_handler, log
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from logging import INFO, ERROR
import os
from dotenv import load_dotenv
load_dotenv('../.env')

from utils.classification.random_forest_classification import RandomForestAlgo, load_data

# Return the current round
def fit_config(server_round: int) -> Dict:
    config = {
        "server_round": server_round,
    }
    return config


def evaluate(server_round, parameters, config):
    print(f"Received parameters for evaluation: {parameters}")

    n_estimators = int(parameters[0][0])
    max_depth = int(parameters[1][0]) if parameters[1][0] != 0 else None  # Convert back 0 to None
    min_samples_split = int(parameters[2][0])
    min_samples_leaf = int(parameters[3][0])

    print("Parameters before setting: ")
    print(f"n_estimators : {n_estimators}")
    print(f"max_depth : {max_depth}")
    print(f"min_samples_split : {min_samples_split}")
    print(f"min_samples_leaf : {min_samples_leaf}")
    
    model = RandomForestAlgo();
    model.set_params(parameters)
    
    x_train_1, y_train_1, _, _, X_global, y_global, = load_data(partition_id=1)
    x_train_2, y_train_2, _, _, X_global, y_global, = load_data(partition_id=2)
    x_train_3, y_train_3, _, _, X_global, y_global, = load_data(partition_id=3)

    model.train(x_train_1, y_train_1)
    model.train(x_train_2, y_train_2)
    model.train(x_train_3, y_train_3)

    y_pred = model.model.predict(X_global)

    accuracy = accuracy_score(y_global, y_pred)
    precision = precision_score(y_global, y_pred, average='weighted')
    recall = recall_score(y_global, y_pred, average='weighted')
    f1 = f1_score(y_global, y_pred, average='weighted')

    line = "-" * 21
    print(line)
    print(f"Accuracy : {accuracy:.8f}")
    print(f"Precision: {precision:.8f}")
    print(f"Recall   : {recall:.8f}")
    print(f"F1 Score : {f1:.8f}")
    print(line)


# Aggregate metrics and calculate weighted averages
def metrics_aggregate(results) -> Dict:
    if not results:
        return {}

    else:
        total_samples = 0  # Number of samples in the dataset

        # Collecting metrics
        aggregated_metrics = {
            "Accuracy": 0,
            "Precision": 0,
            "Recall": 0,
            "F1_Score": 0,
        }

        # Extracting values from the results
        for samples, metrics in results:
            for key, value in metrics.items():
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = 0
                else:
                    aggregated_metrics[key] += (value * samples)
            total_samples += samples

        # Compute the weighted average for each metric
        for key in aggregated_metrics.keys():
            aggregated_metrics[key] = round(aggregated_metrics[key] / total_samples, 6)

        return aggregated_metrics

# model = RandomForestAlgo()
# client_id = 1
# X_train, y_train, _, _, _, _, = load_data(partition_id=client_id)
# model.train(X_train, y_train)
# params = ndarrays_to_parameters(model.get_params());

start_server(
  server_address="0.0.0.0:8000",
  config=ServerConfig(num_rounds=10),
  strategy=FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=metrics_aggregate,
        fit_metrics_aggregation_fn=metrics_aggregate,
        evaluate_fn=evaluate,
        min_fit_clients=3,      
        min_evaluate_clients=3,  
        min_available_clients=3
    )
)