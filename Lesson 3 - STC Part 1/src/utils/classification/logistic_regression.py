from .classification import ClassificationAlgo
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from flwr.common import NDArrays, Scalar
from typing import Dict

class LogisticRegressionAlgo(ClassificationAlgo):
    def __init__(self, CLIENT_TRAINING_SET):
        # Initialize the logistic regression model
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.is_trained = False  # Flag to check if model is trained
        self.train(CLIENT_TRAINING_SET)
        self.is_trained = True

    def train(self, location_of_training_dataset):
        X_train, y_train = self.load_data(location_of_training_dataset)
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")  # Debugging line
        self.model.fit(X_train, y_train)
        self.is_trained = True  # Mark as trained

    def test(self, location_of_testing_dataset):
        # Load test data
        X_test, y_test = self.load_data(location_of_testing_dataset)
        print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")  # Debugging line
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        return y_pred,accuracy  # Return predictions for further analysis if needed

    def get_weights(self):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        return self.model.coef_, self.model.intercept_.reshape(1, -1)  # Ensure intercept is 2D

    def set_weights(self, weights):
        # Set weights; weights should be a tuple of (coef, intercept)
        if len(weights) != 2:
            raise ValueError("Weights must be a tuple of (coef, intercept).")
        coef, intercept = weights
        self.model.coef_ = coef
        self.model.intercept_ = intercept.reshape(1, -1)  # Ensure intercept is 2D

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        coefficients = self.model.coef_
        intercept = self.model.intercept_.reshape(1, -1)  # Ensure intercept is a 2D array
        return [coefficients, intercept]
