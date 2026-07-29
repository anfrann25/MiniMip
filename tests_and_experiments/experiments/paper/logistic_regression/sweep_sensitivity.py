"""
sweep_sensitivity.py
--------------------------------------------------------------------
Sensitivity analysis for the Exaflow staleness-aware caching client.

Sweeps eps_w (and optionally tau_max) and reports, per configuration:
  - total FULL_UPDATE  (summed over all clients)
  - total EMPTY_UPDATE (summed over all clients)
  - % of rounds skipped
  - mean final test accuracy across clients

Drop this file next to AutomateTest.py and run it the same way.
It reuses your existing dataset loading, splitting, server, and client
plumbing; the ONLY change is that the Cached client is now constructed
with the eps_w / tau_max values under test.

NOTE: This imports helpers from your AutomateTest module. If your file
is named differently, change the `import AutomateTest as A` line below.
--------------------------------------------------------------------
"""

import time
import numpy as np
from multiprocessing import Process, Manager
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- reuse everything already defined in your main experiment file ---
import AutomateTest as A          # <-- rename if your file differs
from AutomateTest import (
    load_dataset, stratified_split, start_server, available_clients,
)
from mini_mip_system.client.grpc_agg_client import GRPCClient
from library.under_development.stat_models.federated_logreg_cached import (
    FederatedLogisticRegressionClientCached,
)

# ============ CONFIG ============
DATASET     = "heart"                       # change per run: heart / water / lung / ics3d
EPS_GRID    = [0.001, 0.005, 0.01, 0.05, 0.1]
TAU_GRID    = [20]                           # add more (e.g. [5, 10, 20, 50]) for a 2-D sweep
NUM_CLIENTS = 3
NUM_EPOCHS  = 100

# teardown safety knobs (raise CLIENT_JOIN_TIMEOUT for big datasets like lung/ics3d)
CLIENT_JOIN_TIMEOUT  = 600   # seconds to wait for a client before force-killing
SOCKET_RELEASE_PAUSE = 3     # seconds between configs so the server port frees up
# ================================


def run_client_sweep(client_id, x_train, y_train, x_test, y_test,
                     eps_w, tau_max, comm_dict, acc_dict,
                     aggregation_server="localhost:50051"):
    client = GRPCClient(
        client_id, available_clients, operation_id=0,
        aggregation_server=aggregation_server,
    )
    # the one line that matters: inject the swept hyperparameters
    model = FederatedLogisticRegressionClientCached(
        client, eps_w=eps_w, tau_max=tau_max,
    )
    model.fit(x_train, y_train, num_epochs=NUM_EPOCHS, X_val=x_test, y_val=y_test)

    comm_dict[client_id] = model.communication_log
    preds = model.predict(x_test)
    acc_dict[client_id] = accuracy_score(y_test, preds)


def run_one_config(eps_w, tau_max, X_splits, y_splits, X_test, y_test):
    manager = Manager()
    comm_dict = manager.dict()
    acc_dict = manager.dict()

    server_process = Process(target=start_server, args=(available_clients,))
    server_process.start()
    time.sleep(2)

    procs = []
    for cid in range(1, NUM_CLIENTS + 1):
        p = Process(target=run_client_sweep, args=(
            cid, X_splits[cid - 1], y_splits[cid - 1], X_test, y_test,
            eps_w, tau_max, comm_dict, acc_dict,
        ))
        p.start()
        procs.append(p)

    # join clients with a timeout so a single hung client can't freeze the sweep
    for p in procs:
        p.join(timeout=CLIENT_JOIN_TIMEOUT)
        if p.is_alive():
            print(f"  [warn] client pid {p.pid} still alive after "
                  f"{CLIENT_JOIN_TIMEOUT}s -> terminating")
            p.terminate()
            p.join(timeout=10)
            if p.is_alive():
                p.kill()

    # tear the server down defensively
    server_process.terminate()
    server_process.join(timeout=10)
    if server_process.is_alive():
        print("  [warn] server still alive after terminate -> kill")
        server_process.kill()
        server_process.join(timeout=5)

    # give the OS a moment to release the listening socket before the next
    # config reopens it (this is the usual cause of a hang on the 2nd config)
    time.sleep(SOCKET_RELEASE_PAUSE)

    total_full = sum(log.count("FULL_UPDATE") for log in comm_dict.values())
    total_empty = sum(log.count("EMPTY_UPDATE") for log in comm_dict.values())
    total = total_full + total_empty
    pct_skip = 100 * total_empty / total if total else 0.0
    mean_acc = float(np.mean(list(acc_dict.values()))) if acc_dict else 0.0
    return total_full, total_empty, pct_skip, mean_acc


if __name__ == "__main__":
    X, y = load_dataset(DATASET)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_splits, y_splits = stratified_split(X_train, y_train, num_clients=NUM_CLIENTS)

    print(f"\n################  SWEEP on '{DATASET}'  ################")
    print(f"{'tau_max':>8} {'eps_w':>8} {'FULL':>6} {'EMPTY':>6} "
          f"{'skip%':>7} {'accuracy':>9}")
    print("-" * 50)

    rows = []
    for tau in TAU_GRID:
        for eps in EPS_GRID:
            print(f"  running config: tau_max={tau} eps_w={eps} ...",
                  flush=True)
            full, empty, skip, acc = run_one_config(
                eps, tau, X_splits, y_splits, X_test, y_test
            )
            rows.append((tau, eps, full, empty, skip, acc))
            print(f"{tau:>8} {eps:>8} {full:>6} {empty:>6} "
                  f"{skip:>6.1f}% {acc:>9.4f}", flush=True)

    print("\n=== LaTeX-ready rows (eps_w & FULL & EMPTY & skip% & acc) ===")
    for tau, eps, full, empty, skip, acc in rows:
        print(f"{eps} & {full} & {empty} & {skip:.1f}\\% & {acc:.4f} \\\\ \\hline")