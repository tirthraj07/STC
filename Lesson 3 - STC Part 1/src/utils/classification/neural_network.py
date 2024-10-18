from .classification import ClassificationAlgo
import tensorflow as tf

class NeuralNetworkAlgo(ClassificationAlgo):
    def __init__(self):
        # Define a custom learning rate
        learning_rate = 0.005
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),  
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    def train(self, location_of_training_dataset):
        X_train, y_train = self.load_data(location_of_training_dataset)
        self.model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    def test(self, location_of_testing_dataset):
        X_test, y_test = self.load_data(location_of_testing_dataset)
        
        loss, accuracy = self.model.evaluate(X_test, y_test)
        
        return loss, accuracy
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)
