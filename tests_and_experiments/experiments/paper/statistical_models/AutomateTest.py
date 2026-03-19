import os
import time
from multiprocessing import Process, Manager

import numpy as np
import matplotlib.pyplot as plt

from mini_mip_system.client.grpc_agg_client import GRPCClient
from mini_mip_system.server.grpc_agg_server import serve
from server import available_clients

from library.under_development.stat_models.federated_cachedStatistical import FederatedStatsClientCached


DO_PLOTS = True
METRIC = "mean"   # mean, variance, std, rms, covariance, pearson, regression
SAVE_DIR = f"experiment_results/federated_stats/{METRIC}" #dir gia save ta plots
os.makedirs(SAVE_DIR, exist_ok=True)


#dataset
def make_client_data(client_id: int, n_samples=200000):
    rng = np.random.default_rng(42 * (client_id + 1))

    x = rng.normal(loc=100 + 5 * client_id, scale=10.0, size=n_samples)

    # δεύτερο vector για pairwise metrics
    y = 2.5 * x + rng.normal(loc=0.0, scale=5.0, size=n_samples)

    return x, y


#plots
def plot_execution_times(results_dict, save_path=None):
    names = list(results_dict.keys())
    values = list(results_dict.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}", ha="center", va="bottom")

    plt.ylabel("Execution time (ms)")
    plt.title("Execution time per client")
    plt.xticks(rotation=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_estimate_history(history_dict, metric_name, save_path=None):
    plt.figure(figsize=(10, 6))

    for name, hist in history_dict.items():
        if len(hist) == 0:
            continue

        if metric_name == "regression":
            slope_hist = [v[0] for v in hist]
            plt.plot(slope_hist, label=f"{name} slope")
        else:
            plt.plot(hist, label=name)

    plt.xlabel("Round")
    plt.ylabel("Estimate")
    plt.title(f"{metric_name} estimate per round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_aggregation_counts(agg_counts, save_path=None):
    names = list(agg_counts.keys())
    values = list(agg_counts.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom")

    plt.ylabel("Aggregation count")
    plt.title("How many rounds actually aggregated")
    plt.xticks(rotation=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


#kopia server.py
def start_server(available_clients):
    print("Starting asyncio server...")
    import asyncio
    asyncio.run(serve(available_clients=available_clients))



#kopia client.py
def run_client(metric_name, client_id, results_dict, history_dict, final_dict, agg_count_dict,
               aggregation_server="localhost:50051"):

    x, y = make_client_data(client_id)

    client = GRPCClient(
        client_id,
        available_clients,
        operation_id=99,
        aggregation_server=aggregation_server
    )

    model = FederatedStatsClientCached(
        client,
        warmup_rounds=3,
        tau_max=10,
        eps=1e-3,
        aggregation_interval=5
    )

    start_time = time.time()

    if metric_name == "mean":
        result = model.fit_mean(x, num_rounds=30)

    elif metric_name == "variance":
        result = model.fit_variance(x, num_rounds=30)

    elif metric_name == "std":
        result = model.fit_std(x, num_rounds=30)

    elif metric_name == "rms":
        result = model.fit_rms(x, num_rounds=30)

    elif metric_name == "covariance":
        result = model.fit_covariance(x, y, num_rounds=30)

    elif metric_name == "pearson":
        result = model.fit_pearson(x, y, num_rounds=30)

    elif metric_name == "regression":
        result = model.fit_regression(x, y, num_rounds=30)

    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000

    filtered_global_hist = [val for m, val in model.global_history if m == metric_name]
    filtered_agg_flags = [flag for m, flag in model.did_aggregate if m == metric_name]

    results_dict[f"client_{client_id}"] = elapsed_ms
    history_dict[f"client_{client_id}"] = filtered_global_hist
    final_dict[f"client_{client_id}"] = result
    agg_count_dict[f"client_{client_id}"] = int(sum(filtered_agg_flags))

    print(f"[Client {client_id}] metric={metric_name}")
    print(f"[Client {client_id}] result={result}")
    print(f"[Client {client_id}] time={elapsed_ms:.2f} ms")
    print(f"[Client {client_id}] aggregations={int(sum(filtered_agg_flags))}/{len(filtered_agg_flags)}")


#main loop pou trexei clients k server parallhla
def run_metric_group(metric_name, client_ids):
    manager = Manager()

    results = manager.dict()
    histories = manager.dict()
    finals = manager.dict()
    agg_counts = manager.dict()

    server_process = Process(target=start_server, args=(available_clients,))
    server_process.start()
    time.sleep(2)

    processes = []

    for cid in client_ids:
        p = Process(
            target=run_client,
            args=(metric_name, cid, results, histories, finals, agg_counts)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    server_process.terminate()
    server_process.join()

    print(f"{metric_name} server stopped.\n")

    return dict(results), dict(histories), dict(finals), dict(agg_counts)


#teliko print
def print_summary(metric_name, results, finals, agg_counts):
    print(f"\n=== SUMMARY FOR {metric_name.upper()} ===")
    print("Execution times:", results)
    print("Final values:", finals)
    print("Aggregation counts:", agg_counts)

    if metric_name != "regression":
        vals = list(finals.values())
        print(f"Mean final result across clients: {np.mean(vals):.6f}")
    else:
        slopes = [v[0] for v in finals.values()]
        intercepts = [v[1] for v in finals.values()]
        print(f"Mean slope across clients: {np.mean(slopes):.6f}")
        print(f"Mean intercept across clients: {np.mean(intercepts):.6f}")


# Main gia run prepei na peirazw metrics!
if __name__ == "__main__":
    client_ids = [0, 1, 2]

    results, histories, finals, agg_counts = run_metric_group(METRIC, client_ids)

    print_summary(METRIC, results, finals, agg_counts)

    if DO_PLOTS:
        plot_execution_times(
            results,
            save_path=os.path.join(SAVE_DIR, "execution_times.jpg")
        )

        plot_estimate_history(
            histories,
            METRIC,
            save_path=os.path.join(SAVE_DIR, "estimate_history.jpg")
        )

        plot_aggregation_counts(
            agg_counts,
            save_path=os.path.join(SAVE_DIR, "aggregation_counts.jpg")
        )