"""
model/iris_model.py
-------------------
Defines the neural network model for Iris classification.
This is a simple 3-layer neural network built with TensorFlow/Keras.

In Federated Learning, every client and the aggregator share
the SAME model architecture — only the weights travel over the network.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_model(input_dim=4, num_classes=3):
    """
    Build and return a compiled Keras model.

    Architecture:
        Input (4 features) → Dense(16, ReLU) → Dense(8, ReLU) → Dense(3, Softmax)

    Args:
        input_dim  : number of input features (4 for Iris)
        num_classes: number of output classes  (3 for Iris)

    Returns:
        A compiled tf.keras.Model
    """
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(16, activation='relu', name='hidden_1'),
        keras.layers.Dense(8,  activation='relu', name='hidden_2'),
        keras.layers.Dense(num_classes, activation='softmax', name='output'),
    ], name='iris_classifier')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_weights(model):
    """Return model weights as a list of numpy arrays."""
    return model.get_weights()


def set_weights(model, weights):
    """Load a list of numpy arrays into the model."""
    model.set_weights(weights)


def weights_to_list(weights):
    """Convert numpy weight arrays → plain Python lists (for JSON serialisation)."""
    return [w.tolist() for w in weights]


def list_to_weights(weight_list):
    """Convert plain Python lists → numpy weight arrays."""
    return [np.array(w) for w in weight_list]
