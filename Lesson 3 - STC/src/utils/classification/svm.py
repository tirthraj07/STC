from .classification import ClassificationAlgo
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

class SVMAlgo(ClassificationAlgo):
    def __init__(self, n_features):
        # Linear kernel and class weight balancing
        self.model = SVC(kernel='linear', class_weight='balanced') 
        self.n_features = n_features
        # Manually initialize weights (random or zero)
        self.coef_ = np.random.rand(1, n_features)
        self.intercept_ = np.random.rand(1)
    
    def train(self, location_of_training_dataset):
        X_train, y_train = self.load_data(location_of_training_dataset)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)  # Scale the features
        self.model.fit(X_train_scaled, y_train)
    
    def test(self, location_of_testing_dataset):
        X_test, y_test = self.load_data(location_of_testing_dataset)
        scaler = StandardScaler()
        X_test_scaled = scaler.transform(X_test)  # Scale the test features
        return self.model.score(X_test_scaled, y_test)
    
    def get_weights(self):
        return self.model.coef_, self.model.intercept_
    
    def set_weights(self, weights):
        self.model.coef_, self.model.intercept_ = weights
