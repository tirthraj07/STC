from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple



def load_data(partition_id):
    train_path = os.getenv(f"CLIENT_{partition_id}_TRAINING_SET")
    test_path = os.getenv(f"CLIENT_{partition_id}_TESTING_SET")
    global_test_path = os.getenv('GLOBAL_TESTING_SET')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    global_df = pd.read_csv(global_test_path)

    X_train = train_df.drop(columns=['cardio'])
    y_train = train_df['cardio']

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    X_test = test_df.drop(columns=['cardio'])
    y_test = test_df['cardio']

    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    X_global = global_df.drop(columns=['cardio'])
    y_global = global_df['cardio']

    X_global = X_global.to_numpy()
    y_global = y_global.to_numpy()

    return X_train, y_train, X_test, y_test, X_global, y_global

class RandomForestAlgo:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def get_params(self) -> List[np.ndarray]:
        params = [
            np.array([self.model.n_estimators]),
            np.array([self.model.max_depth]) if self.model.max_depth is not None else np.array([0]),  # Handle NoneType
            np.array([self.model.min_samples_split]),
            np.array([self.model.min_samples_leaf]),
        ]
        print(params)
        return params



    # Set the parameters in the RandomForestClassifier
    def set_params(self, params: List[np.ndarray]) -> RandomForestClassifier:
        self.model.n_estimators = int(params[0][0])
        self.model.max_depth = int(params[1][0]) if params[1][0] != 0 else None  # Convert back 0 to None
        self.model.min_samples_split = int(params[2][0])
        self.model.min_samples_leaf = int(params[3][0])
    
    def train(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def test(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, int, Dict[str, float]]:
        y_pred_proba = self.model.predict_proba(X_test)
        y_pred = self.model.predict(X_test)
    
        loss = log_loss(y_test, y_pred_proba, labels=[0, 1])
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        return loss, len(y_test), {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        }
