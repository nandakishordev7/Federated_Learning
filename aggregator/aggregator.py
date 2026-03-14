"""
aggregator/aggregator.py
------------------------
The Central Aggregator (Server) for our Federated Learning demo.

WHAT THIS DOES (step by step):
  1. Starts a Flask HTTP server on localhost:5000.
  2. Waits until all 3 clients have registered.
  3. Sends the initial global model weights to every client (round 0).
  4. Each round:
       a. Receives weight updates from all clients.
       b. Runs Federated Averaging (FedAvg).
       c. Broadcasts the new global model.
  5. After NUM_ROUNDS, saves the final global model.

API endpoints:
  POST /register          ← client says "I'm here"
  GET  /get_global_model  ← client fetches current global weights
  POST /submit_update     ← client uploads its local weight update
  GET  /status            ← anyone can check the server state

Usage:
    python aggregator/aggregator.py
"""

import sys
import os
import json
import threading
import time

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from flask import Flask, request, jsonify

from model.iris_model   import build_model, get_weights, set_weights, weights_to_list, list_to_weights
from utils.fed_avg      import federated_averaging
from utils.logger       import get_logger

# ── Load configuration ─────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'aggregator_config.yaml')
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

AGG_HOST    = cfg['aggregator']['host']
AGG_PORT    = cfg['aggregator']['port']
NUM_ROUNDS  = cfg['federated_learning']['num_rounds']
NUM_CLIENTS = cfg['federated_learning']['num_clients']
MIN_CLIENTS = cfg['federated_learning']['min_clients']
INPUT_DIM   = cfg['model']['input_dim']
NUM_CLASSES = cfg['model']['num_classes']
LOG_DIR     = cfg['logging']['log_dir']

# ── Logger & Flask app ─────────────────────────────────────────
log = get_logger('aggregator', LOG_DIR)
app = Flask(__name__)

# ── Shared server state (protected by a lock) ──────────────────
state_lock          = threading.Lock()
registered_clients  = {}          # { client_id: { 'n_samples': int } }
current_round       = 0           # which FL round we are in
global_model        = build_model(INPUT_DIM, NUM_CLASSES)
round_updates       = {}          # { client_id: { 'weights': [...], 'n_samples': int } }
round_event         = threading.Event()  # fires when all updates are in


# ══════════════════════════════════════════════════════════════
#  REST Endpoints
# ══════════════════════════════════════════════════════════════

@app.route('/register', methods=['POST'])
def register():
    """Client announces itself and its local dataset size."""
    data      = request.get_json()
    client_id = data['client_id']
    n_samples = data.get('n_samples', 0)

    with state_lock:
        registered_clients[client_id] = {'n_samples': n_samples}
        n_registered = len(registered_clients)

    log.info(f"Client '{client_id}' registered  (n_samples={n_samples})  "
             f"[{n_registered}/{NUM_CLIENTS} connected]")

    return jsonify({'status': 'ok', 'message': f'Registered as {client_id}'})


@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    """Return the current global model weights as JSON."""
    with state_lock:
        weights = get_weights(global_model)
        rnd     = current_round

    return jsonify({
        'round':   rnd,
        'weights': weights_to_list(weights)
    })


@app.route('/submit_update', methods=['POST'])
def submit_update():
    """
    Client submits its locally-trained weight update.
    Once all clients have submitted, FedAvg is triggered.
    """
    data      = request.get_json()
    client_id = data['client_id']
    rnd       = data['round']
    n_samples = data['n_samples']
    weights   = list_to_weights(data['weights'])

    with state_lock:
        round_updates[client_id] = {'weights': weights, 'n_samples': n_samples}
        n_updates = len(round_updates)
        log.info(f"  ← Received update from '{client_id}'  "
                 f"(round {rnd}, n_samples={n_samples})  "
                 f"[{n_updates}/{NUM_CLIENTS} updates]")

        if n_updates >= NUM_CLIENTS:
            round_event.set()   # wake up the aggregation thread

    return jsonify({'status': 'ok'})


@app.route('/status', methods=['GET'])
def status():
    with state_lock:
        return jsonify({
            'current_round':    current_round,
            'num_registered':   len(registered_clients),
            'num_updates_this_round': len(round_updates),
        })


# ══════════════════════════════════════════════════════════════
#  Aggregation Loop  (runs in a background thread)
# ══════════════════════════════════════════════════════════════

def aggregation_loop():
    global current_round, global_model, round_updates

    log.info("Aggregation loop started. Waiting for all clients to register …")

    # Wait for all clients to register
    while True:
        with state_lock:
            n = len(registered_clients)
        if n >= MIN_CLIENTS:
            break
        time.sleep(1)

    log.info(f"All {NUM_CLIENTS} clients registered. Starting federated training!")
    log.info("=" * 60)

    for rnd in range(1, NUM_ROUNDS + 1):
        with state_lock:
            current_round = rnd
            round_updates = {}      # reset for this round
        round_event.clear()

        log.info(f"")
        log.info(f"  ╔══════════════════════════════╗")
        log.info(f"  ║   ROUND {rnd:>2} / {NUM_ROUNDS:<2}  STARTING      ║")
        log.info(f"  ╚══════════════════════════════╝")

        # Wait for all clients to submit updates
        round_event.wait()

        # ── FedAvg ────────────────────────────────────────────
        with state_lock:
            all_weights = [v['weights']   for v in round_updates.values()]
            all_sizes   = [v['n_samples'] for v in round_updates.values()]

        log.info(f"  Running Federated Averaging over {len(all_weights)} clients …")
        new_weights = federated_averaging(all_weights, all_sizes)

        with state_lock:
            set_weights(global_model, new_weights)

        log.info(f"  ✔  Global model updated for round {rnd}")

        # Optional: evaluate on a small held-out set (just log the round)
        log.info(f"  Round {rnd} complete.")

    # ── Training finished ──────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info(f"  Federated Learning COMPLETE after {NUM_ROUNDS} rounds!")
    log.info("=" * 60)

    # Save final model
    save_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'global_model_final.keras')
    global_model.save(save_path)
    log.info(f"  Final global model saved → {save_path}")


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    log.info("=" * 60)
    log.info("  Federated Learning Demo — AGGREGATOR")
    log.info(f"  Listening on http://{AGG_HOST}:{AGG_PORT}")
    log.info(f"  Expecting {NUM_CLIENTS} clients  |  {NUM_ROUNDS} rounds")
    log.info("=" * 60)

    # Start aggregation loop in a background thread
    t = threading.Thread(target=aggregation_loop, daemon=True)
    t.start()

    # Start Flask server (main thread)
    app.run(host=AGG_HOST, port=AGG_PORT, debug=False, use_reloader=False)
