"""
dataset/split_data.py
---------------------
Downloads the Iris dataset and splits it across 3 clients.

Key concept: In Federated Learning each client owns its own
LOCAL data that is NEVER shared with anyone else.
We simulate this by physically saving separate CSV files.

Usage:
    python dataset/split_data.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ── Configuration ──────────────────────────────────────────────
NUM_CLIENTS   = 3
RANDOM_STATE  = 42
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__))   # dataset/
# ───────────────────────────────────────────────────────────────


def split_and_save():
    print("=" * 55)
    print("  Federated Learning – Dataset Splitter")
    print("=" * 55)

    # 1. Load Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    print(f"\n[INFO] Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[INFO] Classes: {list(iris.target_names)}\n")

    # 2. Shuffle
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    # 3. Split into NUM_CLIENTS roughly equal parts
    splits = np.array_split(np.arange(len(X)), NUM_CLIENTS)

    for client_id, idx in enumerate(splits, start=1):
        X_client = X[idx]
        y_client = y[idx]

        # Further split each client slice into train / test
        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_client if len(set(y_client)) == 3 else None
        )

        # Save as CSV
        cols = feature_names + ['label']

        train_df = pd.DataFrame(
            np.column_stack([X_train, y_train]), columns=cols
        )
        test_df = pd.DataFrame(
            np.column_stack([X_test, y_test]),  columns=cols
        )

        train_path = os.path.join(OUTPUT_DIR, f"client_{client_id}_train.csv")
        test_path  = os.path.join(OUTPUT_DIR, f"client_{client_id}_test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path,  index=False)

        print(f"  [Client {client_id}]  train={len(X_train):>3} samples  "
              f"test={len(X_test):>3} samples")
        print(f"             saved → {train_path}")
        print(f"             saved → {test_path}\n")

    print("[DONE] Dataset split complete. Each client has its own private data.")
    print("=" * 55)


if __name__ == "__main__":
    split_and_save()
