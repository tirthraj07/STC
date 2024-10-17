from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg, FedMedian, FedAdam, FedProx
from flwr.common.logger import console_handler, log
from logging import INFO, ERROR
import os
from dotenv import load_dotenv
load_dotenv('../.env')

from utils.classification_factory import classification_algo_factory

CLASSIFICATION_ALGO = os.getenv('CLASSIFICATION_ALGO')
GLOBAL_TESTING_SET = os.getenv('GLOBAL_TESTING_SET')
CLIENT_1_TESTING_SET = os.getenv('CLIENT_1_TESTING_SET')
CLIENT_2_TESTING_SET = os.getenv('CLIENT_2_TESTING_SET')
CLIENT_3_TESTING_SET = os.getenv('CLIENT_3_TESTING_SET')
n_features=11

# print(CLASSIFICATION_ALGO)
# print(GLOBAL_TESTING_SET)
# print(CLIENT_1_TESTING_SET)
# print(CLIENT_2_TESTING_SET)
# print(CLIENT_3_TESTING_SET)


def evaluate(server_round, parameters, config):
    model = classification_algo_factory(CLASSIFICATION_ALGO, n_features)
    model.set_weights(parameters)

    _,accuracy_global = model.test(GLOBAL_TESTING_SET)
    _,accuracy_client1 = model.test(CLIENT_1_TESTING_SET)
    _,accuracy_client2 = model.test(CLIENT_2_TESTING_SET)
    _,accuracy_client3 = model.test(CLIENT_3_TESTING_SET)

    log(INFO, "test accuracy on global testset: %.4f", accuracy_global)
    log(INFO, "test accuracy on client1 testset: %.4f", accuracy_client1)
    log(INFO, "test accuracy on client2 testset: %.4f", accuracy_client2)
    log(INFO, "test accuracy on client3 testset: %.4f", accuracy_client3)


model = classification_algo_factory(CLASSIFICATION_ALGO,n_features)
params = ndarrays_to_parameters(model.get_weights())


# Start Flower server
start_server(
  server_address="0.0.0.0:8000",
  config=ServerConfig(num_rounds=10),
  strategy=FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        initial_parameters=params,
        evaluate_fn=evaluate,
        min_fit_clients=3,      
        min_evaluate_clients=3,  
        min_available_clients=3
    )
)