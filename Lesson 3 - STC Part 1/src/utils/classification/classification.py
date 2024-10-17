from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class ClassificationAlgo(ABC):
    
    @abstractmethod
    def train(self, location_of_training_dataset):
        pass
    
    @abstractmethod
    def test(self, location_of_testing_dataset):
        pass
    
    @abstractmethod
    def get_weights(self):
        pass
    
    @abstractmethod
    def set_weights(self, weights):
        pass

    def load_data(self, file_path):
        # Load the CSV file into a Pandas DataFrame
        df = pd.read_csv(file_path)
        
        # Separate features (X) and target (y)
        X = df.drop(columns=['cardio'])  # Drop the target column to get features
        y = df['cardio']  # Target column
        
        # Convert to NumPy arrays
        X = X.to_numpy()
        y = y.to_numpy()
        
        return X, y
