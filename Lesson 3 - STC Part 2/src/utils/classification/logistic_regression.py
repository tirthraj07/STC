from .classification import ClassificationAlgo
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from flwr.common import NDArrays, Scalar
from typing import Dict


class LogisticRegressionAlgo(ClassificationAlgo):
    def __init__(self, penalty: str = 'l2', local_epochs: int = 100):
        self.penalty = penalty
        self.local_epochs = local_epochs
        self.model = LogisticRegression(
            penalty=penalty,
            max_iter=60,  # local epoch
            warm_start=True,  # prevent refreshing weights when fitting,
        )
        self.set_initial_params()

    def get_model(self, penalty: str, local_epochs: int):
        return LogisticRegression(
            penalty=penalty,
            max_iter=local_epochs,
            warm_start=True,
        )
    
    def get_model_params(self):
        if self.model.fit_intercept:
            params = [
                self.model.coef_,
                self.model.intercept_,
            ]
        else:
            params = [self.model.coef_]
        return params
    
    def set_model_params(self, params):
        self.model.coef_ = params[0]
        if self.model.fit_intercept:
            self.model.intercept_ = params[1]

    
    def set_initial_params(self):
        n_classes = 2  # Binary classification for 'cardio'
        n_features = 4  # Number of features in your dataset (age, ap_hi, ap_lo, cholesterol)
        self.model.classes_ = np.array([0, 1])  # Cardio is binary (0 or 1)

        self.model.coef_ = np.zeros((n_classes, n_features))
        if self.model.fit_intercept:
            self.model.intercept_ = np.zeros((n_classes,))
    
    def train(self, location_of_training_dataset):
        X_train, y_train = self.load_data(location_of_training_dataset)
        
        self.model.fit(X_train, y_train)
        
        train_accuracy = self.model.score(X_train, y_train)
        print(f"Training Accuracy: {train_accuracy:.4f}")
    
    def test(self, location_of_testing_dataset):
        X_test, y_test = self.load_data(location_of_testing_dataset)
        
        y_pred = self.model.predict(X_test)
        
        test_accuracy = accuracy_score(y_test, y_pred)
        return test_accuracy  # Returning the accuracy value
    
    def get_weights(self):
        return self.get_model_params(self.model)
    
    def set_weights(self, weights):
        self.set_model_params(self.model, weights)