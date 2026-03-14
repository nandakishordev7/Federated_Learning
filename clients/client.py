"""
clients/client.py
-----------------
A Federated Learning Client.

WHAT THIS DOES (step by step):
  1. Reads its client ID from the command-line argument (1, 2, or 3).
  2. Loads its PRIVATE local dataset (no other client can see this).
  3. Registers with the aggregator.
  4. For each training round:
       a. Fetches the current global model weights from the aggregator.
       b. Trains locally for LOCAL_EPOCHS epochs on its private data.
       c. Sends the updated weights to the aggregator (NOT the raw data).
  5. Evaluates the final global model on its local test set.

Usage (run each in a separate terminal):
    python clients/client.py 1
    python clients/client.py 2
    python clients/client.py 3
"""

import sys
import os
import json
import time

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import yaml

from model.iris_model  import build_model, get_weights, set_weights, weights_to_list, list_to_weights
from utils.data_loader import load_client_data
from utils.logger      import get_logger

# ── Parse client ID from command line ─────────────────────────
if len(sys.argv) < 2:
    print("Usage: python clients/client.py <client_id>   (1, 2, or 3)")
    sys.exit(1)

CLIENT_ID = int(sys.argv[1])
CLIENT_NAME = f"client_{CLIENT_ID}"

# ── Load configuration ─────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'client_config.yaml')
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

AGG_HOST     = cfg['client']['aggregator_host']
AGG_PORT     = cfg['client']['aggregator_port']
AGG_BASE_URL = f"http://{AGG_HOST}:{AGG_PORT}"
LOCAL_EPOCHS = cfg['local_training']['local_epochs']
BATCH_SIZE   = cfg['local_training']['batch_size']
INPUT_DIM    = cfg['model']['input_dim']
NUM_CLASSES  = cfg['model']['num_classes']
DATASET_DIR  = cfg['dataset']['dir']
LOG_DIR      = cfg['logging']['log_dir']

# ── Logger ─────────────────────────────────────────────────────
log = get_logger(CLIENT_NAME, LOG_DIR)


# ══════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════

def register_with_aggregator(n_samples: int):
    """Tell the aggregator we exist and how many local samples we have."""
    payload = {'client_id': CLIENT_NAME, 'n_samples': n_samples}
    resp = requests.post(f"{AGG_BASE_URL}/register", json=payload, timeout=10)
    resp.raise_for_status()
    log.info(f"Registered with aggregator  (n_samples={n_samples})")


def fetch_global_model():
    """Download the current global model weights + round number."""
    resp = requests.get(f"{AGG_BASE_URL}/get_global_model", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    weights = list_to_weights(data['weights'])
    rnd     = data['round']
    return weights, rnd


def submit_update(weights, rnd: int, n_samples: int):
    """Upload our locally-trained weights to the aggregator."""
    payload = {
        'client_id': CLIENT_NAME,
        'round':     rnd,
        'n_samples': n_samples,
        'weights':   weights_to_list(weights),
    }
    resp = requests.post(f"{AGG_BASE_URL}/submit_update", json=payload, timeout=30)
    resp.raise_for_status()


def wait_for_aggregator(max_retries: int = 30, delay: float = 2.0):
    """Block until the aggregator HTTP server is ready."""
    log.info(f"Waiting for aggregator at {AGG_BASE_URL} …")
    for attempt in range(max_retries):
        try:
            requests.get(f"{AGG_BASE_URL}/status", timeout=3)
            log.info("Aggregator is ready!")
            return
        except requests.exceptions.ConnectionError:
            time.sleep(delay)
    raise RuntimeError("Aggregator did not start in time. Is aggregator.py running?")


def wait_for_new_round(current_rnd: int, poll_interval: float = 1.5):
    """
    Poll the aggregator until the global model has advanced past current_rnd.
    This simulates the synchronisation barrier: a client must wait for the
    aggregator to finish FedAvg before fetching the next global model.
    """
    while True:
        resp = requests.get(f"{AGG_BASE_URL}/status", timeout=5)
        server_round = resp.json()['current_round']
        if server_round > current_rnd:
            return
        time.sleep(poll_interval)


# ══════════════════════════════════════════════════════════════
#  Main training loop
# ══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info(f"  Federated Learning Demo — CLIENT {CLIENT_ID}")
    log.info("=" * 60)

    # 1. Load private local data
    log.info(f"Loading private dataset for client {CLIENT_ID} …")
    X_train, y_train, X_test, y_test = load_client_data(CLIENT_ID, DATASET_DIR)
    log.info(f"Dataset loaded: {len(X_train)} train samples, {len(X_test)} test samples")

    # 2. Wait for aggregator to start
    wait_for_aggregator()

    # 3. Register
    register_with_aggregator(n_samples=len(X_train))

    # 4. Build local model (same architecture as global)
    local_model = build_model(INPUT_DIM, NUM_CLASSES)

    # 5. Federated training rounds
    prev_round = 0
    while True:
        # ── Wait for aggregator to start a new round ────────────
        log.info(f"  Waiting for aggregator to start next round …")
        wait_for_new_round(prev_round)

        # ── Fetch global model ───────────────────────────────────
        global_weights, server_round = fetch_global_model()
        set_weights(local_model, global_weights)
        log.info(f"  ↓ Received global model  (round {server_round})")

        # ── Local training ───────────────────────────────────────
        log.info(f"  Training locally for {LOCAL_EPOCHS} epochs …")
        history = local_model.fit(
            X_train, y_train,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,          # suppress per-epoch Keras output
        )
        train_acc  = history.history['accuracy'][-1]
        train_loss = history.history['loss'][-1]
        log.info(f"  Local training done  —  loss={train_loss:.4f}  acc={train_acc:.4f}")

        # ── Evaluate on local test set ───────────────────────────
        test_loss, test_acc = local_model.evaluate(X_test, y_test, verbose=0)
        log.info(f"  Local test eval       —  loss={test_loss:.4f}  acc={test_acc:.4f}")

        # ── Send update to aggregator ────────────────────────────
        updated_weights = get_weights(local_model)
        submit_update(updated_weights, server_round, len(X_train))
        log.info(f"  ↑ Submitted weight update for round {server_round}")

        prev_round = server_round

        # ── Check if training is complete ────────────────────────
        # Probe server round count from config
        from yaml import safe_load
        with open(CONFIG_PATH.replace('client_config', 'aggregator_config')) as f2:
            agg_cfg = safe_load(f2)
        num_rounds = agg_cfg['federated_learning']['num_rounds']

        if server_round >= num_rounds:
            log.info("")
            log.info("  All rounds complete! Fetching final global model …")
            break

    # 6. Final evaluation with the global model
    final_weights, _ = fetch_global_model()
    set_weights(local_model, final_weights)
    final_loss, final_acc = local_model.evaluate(X_test, y_test, verbose=0)

    log.info("")
    log.info("=" * 60)
    log.info(f"  CLIENT {CLIENT_ID} FINAL RESULTS")
    log.info(f"  Final global model — test loss : {final_loss:.4f}")
    log.info(f"  Final global model — test acc  : {final_acc:.4f}")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
