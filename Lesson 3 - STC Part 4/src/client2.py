from utils.classification.random_forest_classification import RandomForestAlgo, load_data
import os
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.client import NumPyClient, start_numpy_client
from flwr.common import NDArrays, Scalar
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict

from dotenv import load_dotenv
load_dotenv('../.env')

CLIENT_TRAINING_SET = os.getenv('CLIENT_1_TRAINING_SET')
CLIENT_TESTING_SET = os.getenv('CLIENT_1_TESTING_SET')
GLOBAL_TESTING_SET = os.getenv('GLOBAL_TESTING_SET')
CLIENT_NAME = 'client1'

class FlowerClient(NumPyClient):
    def __init__(self, model:RandomForestAlgo, client_id):
        self.model = model
        self.client_id = client_id

    def get_parameters(self, config):
        print(f"Client {self.client_id} received the parameters.")
        return self.model.get_params()

    # Train the local model, return the model parameters to the server
    def fit(self, parameters, config):
        print(f"Received parameters for training: {parameters}")
        
        n_estimators = int(parameters[0][0])
        max_depth = int(parameters[1][0]) if parameters[1][0] != 0 else None  # Convert back 0 to None
        min_samples_split = int(parameters[2][0])
        min_samples_leaf = int(parameters[3][0])

        print("Parameters before setting: ")
        print(f"n_estimators : {n_estimators}")
        print(f"max_depth : {max_depth}")
        print(f"min_samples_split : {min_samples_split}")
        print(f"min_samples_leaf : {min_samples_leaf}")


        self.model.set_params(parameters)

        print("Parameters after setting: ")
        new_parameters = self.model.get_params()
        
        n_estimators = int(new_parameters[0][0])
        max_depth = int(new_parameters[1][0]) if new_parameters[1][0] != 0 else None  # Convert back 0 to None
        min_samples_split = int(new_parameters[2][0])
        min_samples_leaf = int(new_parameters[3][0])

        print(f"n_estimators : {n_estimators}")
        print(f"max_depth : {max_depth}")
        print(f"min_samples_split : {min_samples_split}")
        print(f"min_samples_leaf : {min_samples_leaf}")

        X_train, y_train, _, _, _, _, = load_data(partition_id=self.client_id)
        self.model.train(X_train, y_train)
        print(f"Training finished for round {config['server_round']}.")

        
        trained_params = self.model.get_params()
        print("Trained Parameters: ")

        n_estimators = int(trained_params[0][0])
        max_depth = int(trained_params[1][0]) if trained_params[1][0] != 0 else None  # Convert back 0 to None
        min_samples_split = int(trained_params[2][0])
        min_samples_leaf = int(trained_params[3][0])

        print(f"n_estimators : {n_estimators}")
        print(f"max_depth : {max_depth}")
        print(f"min_samples_split : {min_samples_split}")
        print(f"min_samples_leaf : {min_samples_leaf}")

        X_train, y_train, _, _, _, _, = load_data(partition_id=self.client_id)
        self.model.train(X_train, y_train)
        print(f"Training finished for round {config['server_round']}.")
    
        return trained_params, len(X_train), {}
    
    # Evaluate the local model, return the evaluation result to the server
    def evaluate(self, parameters, config):
        self.model.set_params(parameters)
        _, _, X_test, y_test, _, _, = load_data(partition_id=self.client_id)

        y_pred = self.model.model.predict(X_test)
        loss = log_loss(y_test, y_pred, labels=[0, 1])

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        line = "-" * 21
        print(line)
        print(f"Accuracy : {accuracy:.8f}")
        print(f"Precision: {precision:.8f}")
        print(f"Recall   : {recall:.8f}")
        print(f"F1 Score : {f1:.8f}")
        print(line)

        return loss, len(X_test), {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1_Score": f1}


model = RandomForestAlgo()
client_id = 2
X_train, y_train, _, _, _, _, = load_data(partition_id=client_id)
model.train(X_train, y_train)


start_numpy_client(server_address="127.0.0.1:8000", client=FlowerClient(model, client_id))    