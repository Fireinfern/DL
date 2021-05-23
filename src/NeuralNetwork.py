from tensorflow.keras import Sequential
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.engine.training import Model

class Autoencoder(Model):
    def __init__(self, layers, activators, input_dim, do_prob):
        super(Autoencoder, self).__init__()
        self.hidden_layers = layers
        self.activators = activators
        self.input_dim = input_dim
        self.dropout = do_prob
        self.rng = np.random.RandomState(1234)
        tf.random.set_seed(1234)
    def DefineModel(self):
        self.encoder = Sequential()
        for i in range(len(self.hidden_layers) - 1):
            self.encoder.add(Dense(self.hidden_layers[i], activation=self.activators[i]))
            self.encoder.add(Dropout(self.dropout))
        self.encoder.add(Dense(self.hidden_layers[-1], activation=self.activators[-1]))
        self.decoder = Sequential()
        self.decoder.add(Dense(self.hidden_layers[-1], activation="elu"))
        self.decoder.add(Dense(self.input_dim, activation="elu"))
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded