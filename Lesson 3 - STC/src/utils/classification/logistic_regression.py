from .classification import ClassificationAlgo
from sklearn.linear_model import LogisticRegression
import numpy as np

class LogisticRegressionAlgo(ClassificationAlgo):
    def __init__(self, n_features):
        self.model = LogisticRegression()
        self.n_features = n_features
        # Initialize random weights manually, since the model is not trained yet
        self.initial_coef_ = np.random.rand(1, n_features)
        self.initial_intercept_ = np.random.rand(1)
    
    def train(self, location_of_training_dataset):
        # Load and preprocess training data
        X_train, y_train = self.load_data(location_of_training_dataset)
        self.model.fit(X_train, y_train)
    
    def test(self, location_of_testing_dataset):
        # Load and preprocess test data
        X_test, y_test = self.load_data(location_of_testing_dataset)
        return self.model.score(X_test, y_test)
    
    def get_weights(self):
        # Return model weights after fitting, or initialized weights if not trained
        if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
            return self.model.coef_, self.model.intercept_
        else:
            # If the model isn't trained yet, return the manually initialized weights
            return self.initial_coef_, self.initial_intercept_
    
    def set_weights(self, weights):
        # Logistic Regression in sklearn does not allow setting weights directly,
        # so you need to use your own attributes to store weights if needed
        self.initial_coef_, self.initial_intercept_ = weights
