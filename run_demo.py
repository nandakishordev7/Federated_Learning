#!/usr/bin/env python3
"""
run_demo.py
-----------
Convenience launcher that runs the ENTIRE federated learning demo
in a single terminal using subprocesses (one aggregator + three clients).

This is great for a quick classroom demo.
For a multi-terminal demo use the manual instructions in README.md.

Usage:
    python run_demo.py
"""

import subprocess
import sys
import os
import time
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable


def stream_output(proc, prefix, colour_code):
    reset = "\033[0m"
    for line in iter(proc.stdout.readline, b''):
        text = line.decode('utf-8', errors='replace').rstrip()
        print(f"{colour_code}[{prefix}]{reset} {text}")


def main():
    print("\n" + "=" * 65)
    print("  Federated Learning Demo — Automated Launcher")
    print("=" * 65 + "\n")

    # Step 1: Split dataset
    print("[SETUP] Splitting Iris dataset across 3 clients …")
    result = subprocess.run(
        [PYTHON, os.path.join(BASE_DIR, 'dataset', 'split_data.py')],
        cwd=BASE_DIR, capture_output=False
    )
    if result.returncode != 0:
        print("[ERROR] Dataset split failed. See error above.")
        sys.exit(1)

    print("\n[SETUP] Starting aggregator …")
    time.sleep(1)

    # Step 2: Launch aggregator
    agg_proc = subprocess.Popen(
        [PYTHON, os.path.join(BASE_DIR, 'aggregator', 'aggregator.py')],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    threading.Thread(
        target=stream_output,
        args=(agg_proc, 'AGGREGATOR', '\033[94m'),  # blue
        daemon=True
    ).start()

    time.sleep(3)  # Give the Flask server time to start

    # Step 3: Launch 3 clients
    colours = ['\033[92m', '\033[93m', '\033[95m']  # green, yellow, magenta
    client_procs = []
    for i in range(1, 4):
        print(f"[SETUP] Starting client {i} …")
        p = subprocess.Popen(
            [PYTHON, os.path.join(BASE_DIR, 'clients', 'client.py'), str(i)],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        threading.Thread(
            target=stream_output,
            args=(p, f'CLIENT {i}', colours[i - 1]),
            daemon=True
        ).start()
        client_procs.append(p)
        time.sleep(0.5)

    print("\n" + "=" * 65)
    print("  All processes started. Watch the federated training rounds!")
    print("  Press Ctrl+C to stop early.")
    print("=" * 65 + "\n")

    # Wait for all clients to finish
    for p in client_procs:
        p.wait()

    print("\n[DONE] All clients finished. Stopping aggregator.")
    agg_proc.terminate()
    agg_proc.wait()
    print("[DONE] Demo complete!")


if __name__ == '__main__':
    main()
