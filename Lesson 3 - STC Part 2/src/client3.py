from utils.classification.logistic_regression import LogisticRegressionAlgo
import os
from flwr.client import NumPyClient, start_numpy_client
from flwr.common import NDArrays, Scalar
from typing import Dict
import warnings
from sklearn.metrics import log_loss


from dotenv import load_dotenv
load_dotenv('../.env')

CLASSIFICATION_ALGO = os.getenv('CLASSIFICATION_ALGO')
CLIENT_TRAINING_SET = os.getenv('CLIENT_3_TRAINING_SET')
CLIENT_TESTING_SET = os.getenv('CLIENT_3_TESTING_SET')
GLOBAL_TESTING_SET = os.getenv('GLOBAL_TESTING_SET')
CLIENT_NAME = 'client3'

class FlowerClient(NumPyClient):
    def __init__(self, model: LogisticRegressionAlgo, train_file_path: str, test_file_path: str):
        self.model = model
        
        self.X_train, self.y_train = self.model.load_data(train_file_path)
        self.X_test, self.y_test = self.model.load_data(test_file_path)
        self.round_number = 0

    def fit(self, parameters, config):
        self.model.set_model_params(parameters)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.model.fit(self.X_train, self.y_train)

        return self.model.get_model_params(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_model_params(parameters)

        loss = log_loss(self.y_test, self.model.model.predict_proba(self.X_test))
        accuracy = self.model.model.score(self.X_test, self.y_test)

        return loss, len(self.X_test), {"accuracy": accuracy}



penalty = 'l2'
local_epochs = 100
model = LogisticRegressionAlgo(penalty=penalty, local_epochs=local_epochs)


start_numpy_client(server_address="127.0.0.1:8000", client=FlowerClient(model, CLIENT_TRAINING_SET, CLIENT_TESTING_SET))    