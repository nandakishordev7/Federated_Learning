"""
utils/fed_avg.py
----------------
Implements Federated Averaging (FedAvg) — the core algorithm of
Federated Learning (McMahan et al., 2017).

How it works:
    1. Each client trains locally for E local epochs.
    2. Clients send their weight arrays to the aggregator.
    3. The aggregator computes a weighted average of all weights.
    4. The averaged (global) weights are sent back to every client.

No raw data ever moves — only the weight arrays (numbers).
"""

import numpy as np


def federated_averaging(client_weights: list, client_sizes: list) -> list:
    """
    Compute the weighted average of model weights from multiple clients.

    Args:
        client_weights : list of weight-lists, one per client
                         Each weight-list is [ numpy_array, numpy_array, ... ]
        client_sizes   : number of training samples per client (used for weighting)

    Returns:
        averaged_weights : list of numpy arrays — the new global model weights
    """
    total_samples = sum(client_sizes)

    # Start with zeros shaped like the first client's weights
    avg_weights = [np.zeros_like(w) for w in client_weights[0]]

    for weights, n in zip(client_weights, client_sizes):
        weight_factor = n / total_samples
        for i, layer_weights in enumerate(weights):
            avg_weights[i] += weight_factor * layer_weights

    return avg_weights


def simple_average(client_weights: list) -> list:
    """
    Plain (unweighted) average — all clients contribute equally.
    Useful when datasets are the same size.
    """
    avg_weights = [np.zeros_like(w) for w in client_weights[0]]
    n = len(client_weights)
    for weights in client_weights:
        for i, layer_weights in enumerate(weights):
            avg_weights[i] += layer_weights / n
    return avg_weights
