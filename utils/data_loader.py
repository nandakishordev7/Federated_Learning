"""
utils/data_loader.py
--------------------
Utility functions for loading a client's local CSV dataset.

Clients call load_client_data() to get their private
train / test tensors — no raw data ever leaves this function.
"""

import os
import numpy as np
import pandas as pd


def load_client_data(client_id: int, dataset_dir: str = "dataset"):
    """
    Load the train and test CSV files for a given client.

    Args:
        client_id   : integer 1, 2, or 3
        dataset_dir : path to the dataset folder

    Returns:
        (X_train, y_train, X_test, y_test) as numpy float32 arrays
    """
    train_path = os.path.join(dataset_dir, f"client_{client_id}_train.csv")
    test_path  = os.path.join(dataset_dir, f"client_{client_id}_test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Dataset not found: {train_path}\n"
            "Run  python dataset/split_data.py  first."
        )

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:,  -1].values.astype(np.int32)
    X_test  = test_df.iloc[:,  :-1].values.astype(np.float32)
    y_test  = test_df.iloc[:,   -1].values.astype(np.int32)

    return X_train, y_train, X_test, y_test
