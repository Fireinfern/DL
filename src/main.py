import sys
import os
from numpy.core.defchararray import decode
import tensorflow as tf
import numpy as np
import time
import NeuralNetwork

# Paths
DATASETSPATH = "F:\Programacion\Tensorflow\datasetExploration\Datasets"


def Normalizar(X, N):
    mean = N[0]
    std = N[1]
    for i in range(std.size):
        if std[i] == 0:
            std[i] = 1
    return (X-mean) / std, mean


optimizer = tf.keras.optimizers.Adam()

    # print(tf.__version__)
    # input_data = np.loadtxt(DATASETSPATH + "\\Input.txt", max_rows=10)
    # input_norm = np.loadtxt(DATASETSPATH + "\\InputNorm.txt")
    # print(input_data.shape)
    # print(input_norm.shape)
if __name__ == '__main__':
    input_data = np.loadtxt(DATASETSPATH + "\Input.txt", max_rows=2000)
    print(type(input_data))
    input_norm = np.loadtxt(DATASETSPATH + "\InputNorm.txt")
    normalized_input = (Normalizar(input_data, input_norm))[0]
    print(normalized_input.shape)
    # F encoder creation & training
    f_autoencoder = NeuralNetwork.Autoencoder([512,512], ["elu", "elu"], 419, 0.2)
    f_autoencoder.DefineModel()
    f_autoencoder.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())
    print("F encoder:")
    f_autoencoder.fit(normalized_input[:,:419], normalized_input[:,:419], epochs=10, shuffle=True)
    # G encoder creation & training
    g_autoencoder = NeuralNetwork.Autoencoder([128,128],["elu", "elu"],575-419,0.2)
    g_autoencoder.DefineModel()
    g_autoencoder.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())
    print("G encoder:")
    g_autoencoder.fit(normalized_input[:,419:575], normalized_input[:,419:575], epochs=10, shuffle=True)
    # I encoder creation & training
    i_autoencoder = NeuralNetwork.Autoencoder([512,512], ["elu", "elu"], 2609-575,0.2)
    i_autoencoder.DefineModel()
    i_autoencoder.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())
    print("I encoder:")
    i_autoencoder.fit(normalized_input[:,575:2609], normalized_input[:,575:2609], epochs=10, shuffle=True)
    # E encoder creation & training
    e_autoencoder = NeuralNetwork.Autoencoder([512,512],["elu","elu"], 4657-2609, 0.2)
    e_autoencoder.DefineModel()
    e_autoencoder.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())
    print("E encoder:")
    e_autoencoder.fit(normalized_input[:,2609:4657],normalized_input[:,2609:4657],epochs=10, shuffle=True)
    # decoded = NN.encoder(normalized_input[:,:419])
    # print(decoded)
    # print(NN.decoder(decoded))