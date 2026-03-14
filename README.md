<<<<<<< HEAD
# Federated Learning Demo
### A classroom-ready, fully local simulation using Python + TensorFlow

---

## ⚠️ Why Not IBM Federated Learning (ibmfl)?

This project implements Federated Learning **from scratch** using Flask + TensorFlow instead of the official `ibmfl` library. Here is the clear technical reasoning:

### 1. IBM-FL Is No Longer Actively Maintained
IBM-FL's last stable release was in **2022**. The library has since been deprecated with no further updates, bug fixes, or security patches. Building a demo on an unmaintained library is not a sound engineering decision.

### 2. Python Version Incompatibility
IBM-FL requires **Python 3.7 or 3.8 specifically**. Windows 11 ships with Python 3.11/3.12 by default. There is no supported upgrade path — using IBM-FL would require installing a legacy Python version and managing conflicting environments.

### 3. Broken pip Installation on Modern Systems
```bash
pip install ibmfl   # Fails with unresolvable dependency conflicts on Python 3.9+
```
This is a widely reported, unfixed issue. The demo would not be reproducible on any standard modern machine.

### 4. IBM-FL Is a Wrapper Around the Same Core Concepts
IBM-FL internally implements the exact same components built here:

| IBM-FL Component | This Project's Equivalent |
|---|---|
| `FusionHandler` | `utils/fed_avg.py` |
| `LocalTrainingHandler` | `clients/client.py` |
| Aggregator Party | `aggregator/aggregator.py` |
| Party config YAML | `configs/client_config.yaml` |
| Protocol server | Flask HTTP server |

This project **exposes every component transparently** — students can read and trace every line of the FL pipeline, which IBM-FL's abstraction layer hides.

### 5. Better for Educational Clarity
IBM-FL abstracts the internals behind framework calls and opaque config files. This implementation lets you observe **exactly** what happens at each step: how weights are extracted, how FedAvg is computed, and how the aggregator communicates with clients — making it more educational, not less rigorous.

### One-Line Summary
> *"IBM-FL is deprecated and incompatible with modern Python. This project implements the identical McMahan et al. (2017) FedAvg algorithm from scratch using Flask and TensorFlow, producing the same federated learning behaviour while remaining fully transparent and runnable on any current system."*

---

## 📖 What Is Federated Learning?

In classical machine learning, all data is sent to a central server for training.  
**Federated Learning (FL)** flips this:

```
Classical ML:   Client → RAW DATA → Server → trains model
Federated ML:   Client → trains locally → sends WEIGHTS only → Server aggregates
```

No raw data ever leaves the client's device.  
The server sees only **model weight numbers** — never the actual samples.

---

## 🗂️ Project Structure

```
federated_learning_demo/
│
├── aggregator/
│   └── aggregator.py          ← Central server (Flask, runs FedAvg)
│
├── clients/
│   └── client.py              ← FL client (loads local data, trains, uploads weights)
│
├── dataset/
│   ├── split_data.py          ← Splits Iris into 3 private client datasets
│   ├── client_1_train.csv     ← (generated) Client 1's private training data
│   ├── client_1_test.csv      ← (generated) Client 1's private test data
│   ├── client_2_train.csv     ← (generated)
│   ├── client_2_test.csv
│   ├── client_3_train.csv
│   └── client_3_test.csv
│
├── model/
│   └── iris_model.py          ← Neural network definition (shared architecture)
│
├── utils/
│   ├── data_loader.py         ← CSV → numpy array helper
│   ├── fed_avg.py             ← Federated Averaging (FedAvg) implementation
│   └── logger.py             ← Coloured console logging
│
├── configs/
│   ├── aggregator_config.yaml ← Server settings (port, rounds, clients)
│   └── client_config.yaml    ← Client settings (server address, local epochs)
│
├── logs/                      ← Auto-created; one .log file per participant
├── run_demo.py                ← One-click demo launcher (all-in-one terminal)
└── requirements.txt
```

---

## ⚙️ Step-by-Step Setup

### Step 1 — Install Dependencies

```bash
# From the project root:
pip install -r requirements.txt
```

Dependencies:
| Package | Why |
|---|---|
| tensorflow | Neural network model |
| scikit-learn | Iris dataset + train/test split |
| flask | Aggregator HTTP server |
| requests | Clients call the aggregator |
| pyyaml | Read config files |
| numpy, pandas, matplotlib | Data handling |

---

### Step 2 — Split the Dataset

```bash
python dataset/split_data.py
```

This creates **6 CSV files** in `dataset/`:
- `client_1_train.csv`, `client_1_test.csv`
- `client_2_train.csv`, `client_2_test.csv`
- `client_3_train.csv`, `client_3_test.csv`

Each client owns its own slice of the Iris dataset.  
**No other client can see another's data.**

Expected output:
```
[Client 1]  train= 40 samples  test= 10 samples
[Client 2]  train= 40 samples  test= 10 samples
[Client 3]  train= 40 samples  test= 10 samples
```

---

## 🚀 Running the Demo

### Option A — One Terminal (quick demo)

```bash
python run_demo.py
```

Launches the aggregator + 3 clients automatically with colour-coded output.

---

### Option B — Four Terminals (recommended for classroom)

Open **4 terminals** side by side.

#### Terminal 1 — Start the Aggregator
```bash
cd federated_learning_demo
python aggregator/aggregator.py
```

You will see:
```
[aggregator] INFO     Federated Learning Demo — AGGREGATOR
[aggregator] INFO     Listening on http://127.0.0.1:5000
[aggregator] INFO     Expecting 3 clients  |  10 rounds
[aggregator] INFO     Waiting for all clients to register …
```

#### Terminal 2 — Start Client 1
```bash
cd federated_learning_demo
python clients/client.py 1
```

#### Terminal 3 — Start Client 2
```bash
cd federated_learning_demo
python clients/client.py 2
```

#### Terminal 4 — Start Client 3
```bash
cd federated_learning_demo
python clients/client.py 3
```

---

## 👀 What to Observe

### Aggregator Terminal
```
[aggregator] INFO     Client 'client_1' registered  (n_samples=40)  [1/3 connected]
[aggregator] INFO     Client 'client_2' registered  (n_samples=40)  [2/3 connected]
[aggregator] INFO     Client 'client_3' registered  (n_samples=40)  [3/3 connected]
[aggregator] INFO     All 3 clients registered. Starting federated training!

[aggregator] INFO       ╔══════════════════════════════╗
[aggregator] INFO       ║   ROUND  1 / 10  STARTING      ║
[aggregator] INFO       ╚══════════════════════════════╝
[aggregator] INFO       ← Received update from 'client_1'  (round 1)  [1/3 updates]
[aggregator] INFO       ← Received update from 'client_2'  (round 1)  [2/3 updates]
[aggregator] INFO       ← Received update from 'client_3'  (round 1)  [3/3 updates]
[aggregator] INFO       Running Federated Averaging over 3 clients …
[aggregator] INFO       ✔  Global model updated for round 1
```

### Client Terminal
```
[client_1]   INFO     Loading private dataset for client 1 …
[client_1]   INFO     Dataset loaded: 40 train, 10 test samples
[client_1]   INFO     Registered with aggregator
[client_1]   INFO       ↓ Received global model  (round 1)
[client_1]   INFO       Training locally for 5 epochs …
[client_1]   INFO       Local training done  —  loss=0.8231  acc=0.7500
[client_1]   INFO       Local test eval       —  loss=0.7104  acc=0.8000
[client_1]   INFO       ↑ Submitted weight update for round 1
```

---

## 🔄 The Federated Learning Pipeline (Visualised)

```
┌─────────────────────────────────────────────────────────┐
│                     AGGREGATOR                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Global Model  (shared weights W_global)         │   │
│  └────────────┬──────────────────────────┬──────────┘   │
│               │  broadcast W_global      │              │
└───────────────┼──────────────────────────┼──────────────┘
                │                          │
    ┌───────────▼──────┐       ┌───────────▼──────┐
    │    Client 1      │       │    Client 2      │  ...
    │  Local data only │       │  Local data only │
    │  Train 5 epochs  │       │  Train 5 epochs  │
    │  → W_update_1    │       │  → W_update_2    │
    └───────────┬──────┘       └──────────┬───────┘
                │  send weights only       │
                └──────────────┬───────────┘
                               ▼
                     ┌─────────────────┐
                     │   FedAvg        │
                     │  W_new = Σ(n_i  │
                     │  / N) * W_i     │
                     └────────┬────────┘
                              │  new global model
                              ▼  (next round)
```

---

## 🧮 FedAvg Formula

The aggregator computes a **weighted average** of all client weights:

```
W_global = Σ (n_i / N) × W_i

where:
  n_i = number of training samples on client i
  N   = total samples across all clients
  W_i = weight arrays from client i
```

This is **McMahan et al. (2017) — "Communication-Efficient Learning of Deep Networks from Decentralized Data"**, the foundational FL paper.

---

## ⚙️ Configuration

### `configs/aggregator_config.yaml`
| Key | Default | Description |
|---|---|---|
| `num_rounds` | 10 | Number of FL training rounds |
| `num_clients` | 3 | Expected clients |
| `port` | 5000 | Flask server port |

### `configs/client_config.yaml`
| Key | Default | Description |
|---|---|---|
| `local_epochs` | 5 | Local training epochs per round |
| `batch_size` | 16 | Mini-batch size |
| `aggregator_port` | 5000 | Must match aggregator |

---

## 🎓 Key Teaching Points

1. **Privacy**: Clients never share raw CSV rows — only floating-point weight arrays.
2. **Independence**: Each client runs on its own local data partition.
3. **Synchronisation**: The aggregator waits for ALL clients before averaging.
4. **Convergence**: Watch accuracy improve across rounds in client logs.
5. **FedAvg**: Larger clients contribute proportionally more to the global model.

---

## 🔧 Customisation Ideas

| Change | How |
|---|---|
| More rounds | Edit `num_rounds` in `aggregator_config.yaml` |
| More clients | Change `num_clients` and run more `client.py N` processes |
| Bigger model | Edit `model/iris_model.py` |
| Different dataset | Replace `dataset/split_data.py` + `utils/data_loader.py` |
| Non-IID split | Modify `split_data.py` to give each client only certain classes |

---

## 📊 Expected Final Accuracy

After 10 rounds with default settings you should see:
- Train accuracy: **~95%**
- Test accuracy: **~90–97%** per client

The model converges well because Iris is a simple dataset.  
Try setting `num_rounds: 3` to show a less-converged model, then `10` to show full convergence.
=======
# Federated_Learning
this is just a mimic of the IBM - FL 
>>>>>>> 6e72289adcebbc27d1bdf7e2b39400533ca89d09
