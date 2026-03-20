import os
import time
from multiprocessing import Process, Manager

import numpy as np
import matplotlib.pyplot as plt

from mini_mip_system.client.grpc_agg_client import GRPCClient
from mini_mip_system.server.grpc_agg_server import serve
from server import available_clients

from library.under_development.stat_models.federated_cachedStatistical import FederatedStatsClientCached
from library.under_development.stat_models.federated_Statistical import FederatedStatsClient


DO_PLOTS = True
METRIC = "variance"   # mean, variance, std, rms, covariance, pearson, regression
SAVE_DIR = f"experiment_results/federated_stats_compare/{METRIC}"
os.makedirs(SAVE_DIR, exist_ok=True)


# -------------------- dataset --------------------

def make_client_data(client_id: int, n_samples=200000):
    rng = np.random.default_rng(42 * (client_id + 1))

    x = rng.normal(loc=100 + 5 * client_id, scale=10.0, size=n_samples)
    y = 2.5 * x + rng.normal(loc=0.0, scale=5.0, size=n_samples)

    return x, y


# -------------------- plots --------------------

def plot_model_execution_time_comparison(cached_results, nocache_results, save_path=None):
    model_names = ["cached", "no_cache"]
    values = [
        np.mean(list(cached_results.values())),
        np.mean(list(nocache_results.values()))
    ]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(model_names, values)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom")

    plt.ylabel("Mean execution time (ms)")
    plt.title("Model comparison: execution time")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_aggregation_counts_comparison(agg_counts_by_mode, save_path=None):
    modes = list(agg_counts_by_mode.keys())
    client_names = sorted(agg_counts_by_mode[modes[0]].keys())

    x = np.arange(len(client_names))
    width = 0.35

    plt.figure(figsize=(10, 6))

    for i, mode in enumerate(modes):
        values = [agg_counts_by_mode[mode].get(c, 0) for c in client_names]
        bars = plt.bar(x + (i - 0.5) * width, values, width=width, label=mode)

        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{int(h)}",
                ha="center",
                va="bottom"
            )

    plt.xticks(x, client_names, rotation=20)
    plt.ylabel("Aggregation count")
    plt.title("Aggregation count per client: cached vs no_cache")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_model_aggregation_comparison(cached_agg_counts, nocache_agg_counts, save_path=None):
    model_names = ["cached", "no_cache"]
    values = [
        np.mean(list(cached_agg_counts.values())),
        np.mean(list(nocache_agg_counts.values()))
    ]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(model_names, values)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom")

    plt.ylabel("Mean aggregation count")
    plt.title("Model comparison: aggregation count")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_model_final_value_comparison(metric_name, cached_finals, nocache_finals, save_path=None):
    if metric_name == "regression":
        cached_slope = np.mean([v[0] for v in cached_finals.values()])
        nocache_slope = np.mean([v[0] for v in nocache_finals.values()])

        cached_intercept = np.mean([v[1] for v in cached_finals.values()])
        nocache_intercept = np.mean([v[1] for v in nocache_finals.values()])

        labels = ["cached_slope", "no_cache_slope", "cached_intercept", "no_cache_intercept"]
        values = [cached_slope, nocache_slope, cached_intercept, nocache_intercept]
    else:
        labels = ["cached", "no_cache"]
        values = [
            np.mean(list(cached_finals.values())),
            np.mean(list(nocache_finals.values()))
        ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.6f}", ha="center", va="bottom")

    plt.ylabel("Final value")
    plt.title(f"Model comparison: final {metric_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_model_history_comparison(metric_name, cached_histories, nocache_histories, save_path=None):
    plt.figure(figsize=(10, 6))

    if metric_name == "regression":
        cached_curves = [np.array([v[0] for v in hist]) for hist in cached_histories.values() if len(hist) > 0]
        nocache_curves = [np.array([v[0] for v in hist]) for hist in nocache_histories.values() if len(hist) > 0]
    else:
        cached_curves = [np.array(hist) for hist in cached_histories.values() if len(hist) > 0]
        nocache_curves = [np.array(hist) for hist in nocache_histories.values() if len(hist) > 0]

    if len(cached_curves) > 0:
        min_len_cached = min(len(c) for c in cached_curves)
        cached_matrix = np.array([c[:min_len_cached] for c in cached_curves])
        cached_mean_curve = np.mean(cached_matrix, axis=0)
        plt.plot(cached_mean_curve, label="cached", linewidth=2)

    if len(nocache_curves) > 0:
        min_len_nocache = min(len(c) for c in nocache_curves)
        nocache_matrix = np.array([c[:min_len_nocache] for c in nocache_curves])
        nocache_mean_curve = np.mean(nocache_matrix, axis=0)
        plt.plot(nocache_mean_curve, label="no_cache", linewidth=2)

    plt.xlabel("Round")
    plt.ylabel("Estimate")
    plt.title(f"Model comparison: {metric_name} estimate history")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# -------------------- server --------------------

def start_server(available_clients):
    print("Starting asyncio server...")
    import asyncio
    asyncio.run(serve(available_clients=available_clients))


# -------------------- client runner --------------------

def build_model(mode, client):
    if mode == "cached":
        return FederatedStatsClientCached(
            client,
            warmup_rounds=3,
            tau_max=10,
            eps=1e-3,
            aggregation_interval=5
        )

    if mode == "no_cache":
        return FederatedStatsClient(client)

    raise ValueError(f"Unknown mode: {mode}")


def run_selected_metric(model, metric_name, x, y, num_rounds=30):
    if metric_name == "mean":
        return model.fit_mean(x, num_rounds=num_rounds)

    elif metric_name == "variance":
        return model.fit_variance(x, num_rounds=num_rounds)

    elif metric_name == "std":
        return model.fit_std(x, num_rounds=num_rounds)

    elif metric_name == "rms":
        return model.fit_rms(x, num_rounds=num_rounds)

    elif metric_name == "covariance":
        return model.fit_covariance(x, y, num_rounds=num_rounds)

    elif metric_name == "pearson":
        return model.fit_pearson(x, y, num_rounds=num_rounds)

    elif metric_name == "regression":
        return model.fit_regression(x, y, num_rounds=num_rounds)

    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def run_client(
    mode,
    metric_name,
    client_id,
    results_dict,
    history_dict,
    final_dict,
    agg_count_dict,
    aggregation_server="localhost:50051"
):
    x, y = make_client_data(client_id)

    client = GRPCClient(
        client_id,
        available_clients,
        operation_id=99,
        aggregation_server=aggregation_server
    )

    model = build_model(mode, client)

    start_time = time.time()
    result = run_selected_metric(model, metric_name, x, y, num_rounds=30)
    end_time = time.time()

    elapsed_ms = (end_time - start_time) * 1000

    filtered_global_hist = [val for m, val in model.global_history if m == metric_name]
    filtered_agg_flags = [flag for m, flag in model.did_aggregate if m == metric_name]

    key = f"client_{client_id}"
    results_dict[key] = elapsed_ms
    history_dict[key] = filtered_global_hist
    final_dict[key] = result
    agg_count_dict[key] = int(sum(filtered_agg_flags))

    print(f"[{mode}][Client {client_id}] metric={metric_name}")
    print(f"[{mode}][Client {client_id}] result={result}")
    print(f"[{mode}][Client {client_id}] time={elapsed_ms:.2f} ms")
    print(f"[{mode}][Client {client_id}] aggregations={int(sum(filtered_agg_flags))}/{len(filtered_agg_flags)}")


# -------------------- experiment group --------------------

def run_metric_group(mode, metric_name, client_ids):
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
            args=(mode, metric_name, cid, results, histories, finals, agg_counts)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    server_process.terminate()
    server_process.join()

    print(f"{metric_name} server stopped for mode={mode}.\n")

    return dict(results), dict(histories), dict(finals), dict(agg_counts)


# -------------------- summaries --------------------

def print_summary(mode, metric_name, results, finals, agg_counts):
    print(f"\n=== SUMMARY FOR {metric_name.upper()} | MODE={mode.upper()} ===")
    print("Execution times:", results)
    print("Final values:", finals)
    print("Aggregation counts:", agg_counts)

    if metric_name != "regression":
        vals = list(finals.values())
        print(f"Mean final result across clients: {np.mean(vals):.6f}")
        print(f"Mean execution time across clients: {np.mean(list(results.values())):.2f} ms")
        print(f"Mean aggregation count across clients: {np.mean(list(agg_counts.values())):.2f}")
    else:
        slopes = [v[0] for v in finals.values()]
        intercepts = [v[1] for v in finals.values()]
        print(f"Mean slope across clients: {np.mean(slopes):.6f}")
        print(f"Mean intercept across clients: {np.mean(intercepts):.6f}")
        print(f"Mean execution time across clients: {np.mean(list(results.values())):.2f} ms")
        print(f"Mean aggregation count across clients: {np.mean(list(agg_counts.values())):.2f}")


def print_comparison(metric_name, cached_results, cached_finals, cached_agg,
                     nocache_results, nocache_finals, nocache_agg):
    print(f"\n=== COMPARISON FOR {metric_name.upper()} ===")

    cached_time_mean = np.mean(list(cached_results.values()))
    nocache_time_mean = np.mean(list(nocache_results.values()))

    cached_agg_mean = np.mean(list(cached_agg.values()))
    nocache_agg_mean = np.mean(list(nocache_agg.values()))

    print(f"Cached mean execution time: {cached_time_mean:.2f} ms")
    print(f"No-cache mean execution time: {nocache_time_mean:.2f} ms")
    print(f"Time difference (no_cache - cached): {nocache_time_mean - cached_time_mean:.2f} ms")

    print(f"Cached mean aggregation count: {cached_agg_mean:.2f}")
    print(f"No-cache mean aggregation count: {nocache_agg_mean:.2f}")
    print(f"Aggregation difference: {nocache_agg_mean - cached_agg_mean:.2f}")

    if metric_name != "regression":
        cached_vals = list(cached_finals.values())
        nocache_vals = list(nocache_finals.values())

        print(f"Cached mean final metric: {np.mean(cached_vals):.6f}")
        print(f"No-cache mean final metric: {np.mean(nocache_vals):.6f}")
        print(f"Absolute difference: {abs(np.mean(cached_vals) - np.mean(nocache_vals)):.6f}")
    else:
        cached_slopes = [v[0] for v in cached_finals.values()]
        nocache_slopes = [v[0] for v in nocache_finals.values()]
        cached_intercepts = [v[1] for v in cached_finals.values()]
        nocache_intercepts = [v[1] for v in nocache_finals.values()]

        print(f"Cached mean slope: {np.mean(cached_slopes):.6f}")
        print(f"No-cache mean slope: {np.mean(nocache_slopes):.6f}")
        print(f"Slope abs diff: {abs(np.mean(cached_slopes) - np.mean(nocache_slopes)):.6f}")

        print(f"Cached mean intercept: {np.mean(cached_intercepts):.6f}")
        print(f"No-cache mean intercept: {np.mean(nocache_intercepts):.6f}")
        print(f"Intercept abs diff: {abs(np.mean(cached_intercepts) - np.mean(nocache_intercepts)):.6f}")


# -------------------- main --------------------

if __name__ == "__main__":
    client_ids = [0, 1, 2]

    cached_results, cached_histories, cached_finals, cached_agg_counts = run_metric_group(
        "cached", METRIC, client_ids
    )

    nocache_results, nocache_histories, nocache_finals, nocache_agg_counts = run_metric_group(
        "no_cache", METRIC, client_ids
    )

    print_summary("cached", METRIC, cached_results, cached_finals, cached_agg_counts)
    print_summary("no_cache", METRIC, nocache_results, nocache_finals, nocache_agg_counts)

    print_comparison(
        METRIC,
        cached_results, cached_finals, cached_agg_counts,
        nocache_results, nocache_finals, nocache_agg_counts
    )

    if DO_PLOTS:
        plot_model_execution_time_comparison(
            cached_results,
            nocache_results,
            save_path=os.path.join(SAVE_DIR, "model_execution_time_comparison.jpg")
        )

        plot_model_aggregation_comparison(
            cached_agg_counts,
            nocache_agg_counts,
            save_path=os.path.join(SAVE_DIR, "model_aggregation_comparison.jpg")
        )

        plot_model_final_value_comparison(
            METRIC,
            cached_finals,
            nocache_finals,
            save_path=os.path.join(SAVE_DIR, "model_final_value_comparison.jpg")
        )

        plot_model_history_comparison(
            METRIC,
            cached_histories,
            nocache_histories,
            save_path=os.path.join(SAVE_DIR, "model_history_comparison.jpg")
        )