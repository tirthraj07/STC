from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg, FedMedian, FedAdam, FedProx
from flwr.common.logger import console_handler, log
from logging import INFO, ERROR
import os
from dotenv import load_dotenv

from utils.classification.logistic_regression import LogisticRegressionAlgo
load_dotenv('../.env')


CLASSIFICATION_ALGO = os.getenv('CLASSIFICATION_ALGO')
GLOBAL_TESTING_SET = os.getenv('GLOBAL_TESTING_SET')
CLIENT_1_TESTING_SET = os.getenv('CLIENT_1_TESTING_SET')
CLIENT_2_TESTING_SET = os.getenv('CLIENT_2_TESTING_SET')
CLIENT_3_TESTING_SET = os.getenv('CLIENT_3_TESTING_SET')

penalty = 'l2'
local_epochs = 100

def evaluate(server_round, parameters, config):
    model = LogisticRegressionAlgo(penalty=penalty, local_epochs=local_epochs)
    model.set_model_params(parameters) 

    accuracy_global = model.test(GLOBAL_TESTING_SET)
    accuracy_client1 = model.test(CLIENT_1_TESTING_SET)
    accuracy_client2 = model.test(CLIENT_2_TESTING_SET)
    accuracy_client3 = model.test(CLIENT_3_TESTING_SET)

    log(INFO, "Test accuracy on global test set: %.4f", accuracy_global)
    log(INFO, "Test accuracy on client1 test set: %.4f", accuracy_client1)
    log(INFO, "Test accuracy on client2 test set: %.4f", accuracy_client2)
    log(INFO, "Test accuracy on client3 test set: %.4f", accuracy_client3)



model = LogisticRegressionAlgo(penalty=penalty, local_epochs=local_epochs)
initial_parameters = ndarrays_to_parameters(model.get_model_params())


# Start Flower server
start_server(
  server_address="0.0.0.0:8000",
  config=ServerConfig(num_rounds=10),
  strategy=FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate,
        min_fit_clients=3,      
        min_evaluate_clients=3,  
        min_available_clients=3
    )
)