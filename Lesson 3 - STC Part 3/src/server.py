from typing import Dict
from flwr.server import ServerApp
from flwr.common import Context, Parameters
from flwr.server.strategy import FedXgbBagging
from flwr.server import ServerConfig, start_server
from logging import INFO, ERROR
from flwr.common.logger import console_handler, log
import os
from dotenv import load_dotenv
import xgboost as xgb
import pandas as pd

from utils.classification.xgboost_classification import load_data

# Load environment variables
load_dotenv('../env')

def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    loss_aggregated = (
        sum([metrics["LOSS"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": auc_aggregated, "LOSS":loss_aggregated}
    
    print(f"Global round AUC: {metrics_aggregated['AUC']}") 
    print(f"Global round LOSS: {metrics_aggregated['LOSS']}")
    
    # Print the AUC for each client
    for idx, client_metrics in enumerate(eval_metrics, start=1):
        client_num, metrics = client_metrics
        print(f"Client {idx} AUC: {metrics['AUC']}")


    return metrics_aggregated


def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config

parameters = Parameters(tensor_type="", tensors=[])

# Define the FedXgbBagging strategy
strategy = FedXgbBagging(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    on_evaluate_config_fn=config_func,
    on_fit_config_fn=config_func,
    initial_parameters=parameters,
    min_fit_clients=3,      
    min_evaluate_clients=3,  
    min_available_clients=3

)

start_server(
  server_address="0.0.0.0:8000",
  config=ServerConfig(num_rounds=10),
  strategy=strategy
)