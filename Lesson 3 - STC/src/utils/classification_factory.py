from .classification.classification import ClassificationAlgo
from .classification.logistic_regression import LogisticRegressionAlgo
from .classification.neural_network import NeuralNetworkAlgo
from .classification.svm import SVMAlgo

def classification_algo_factory(algo_type: str, n_features) -> ClassificationAlgo:
    """
    Factory function to create an instance of a classification algorithm.
    
    Parameters:
    algo_type (str): The type of the classification algorithm ('logistic_regression', 'neural_network', 'svm').
    
    Returns:
    ClassificationAlgo: An instance of the requested classification algorithm class.
    """
    if algo_type == 'logistic_regression':
        return LogisticRegressionAlgo(n_features)
    elif algo_type == 'neural_network':
        return NeuralNetworkAlgo()
    elif algo_type == 'svm':
        return SVMAlgo(n_features)
    else:
        raise ValueError(f"Unsupported algorithm type: {algo_type}")

# Example usage:
# logistic_model = classification_algo_factory('logistic_regression')
# neural_net_model = classification_algo_factory('neural_network')
# svm_model = classification_algo_factory('svm')
