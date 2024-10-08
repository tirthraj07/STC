from .classification import ClassificationAlgo
import tensorflow as tf

class NeuralNetworkAlgo(ClassificationAlgo):
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(12,)), # Update to 12 if you want to include an additional feature
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    def train(self, location_of_training_dataset):
        X_train, y_train = self.load_data(location_of_training_dataset)
        self.model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    def test(self, location_of_testing_dataset):
        # Load test data
        X_test, y_test = self.load_data(location_of_testing_dataset)
        
        # Evaluate the model on test data; evaluate returns [loss, accuracy]
        loss, accuracy = self.model.evaluate(X_test, y_test)
        
        # Return both loss and accuracy
        return loss, accuracy
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)
    