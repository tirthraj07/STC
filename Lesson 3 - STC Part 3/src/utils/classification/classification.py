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

