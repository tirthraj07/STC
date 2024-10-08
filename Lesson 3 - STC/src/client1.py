from utils.classification_factory import classification_algo_factory
import os
from flwr.client import NumPyClient, start_numpy_client
from flwr.common import NDArrays, Scalar
from typing import Dict

from dotenv import load_dotenv
load_dotenv('../.env')

CLASSIFICATION_ALGO = os.getenv('CLASSIFICATION_ALGO')
CLIENT_TRAINING_SET = os.getenv('CLIENT_1_TRAINING_SET')
CLIENT_TESTING_SET = os.getenv('CLIENT_1_TESTING_SET')
GLOBAL_TESTING_SET = os.getenv('GLOBAL_TESTING_SET')
CLIENT_NAME = 'client1'
n_features=11

class FlowerClient(NumPyClient):
    def __init__(self, model, trainset_location, testset_location):
        self.model = model
        self.trainset_location = trainset_location
        self.testset_location = testset_location
        self.round_number = 0

    # Train the model
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.train(self.trainset_location)
        num_samples = len(self.model.load_data(self.trainset_location)[1])
        # Plot the graph for the round and store it inside /results/client1/plots
        self.round_number += 1
        return self.model.get_weights(), num_samples, {}

    # Test the model
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Update model parameters
        self.model.set_weights(parameters)

        # Perform evaluation (assumes your model has a test method that returns loss and accuracy)
        loss, accuracy = self.model.test(self.testset_location)

        # Calculate the number of samples in the test set
        num_samples = len(self.model.load_data(self.testset_location)[1])

        # Return the required tuple (loss, num_samples, and metrics dictionary)
        return loss, num_samples, {"accuracy": accuracy}



model = classification_algo_factory(CLASSIFICATION_ALGO,n_features)
start_numpy_client(server_address="127.0.0.1:8000", client=FlowerClient(model, CLIENT_TRAINING_SET, CLIENT_TESTING_SET))    