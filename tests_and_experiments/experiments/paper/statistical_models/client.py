import numpy as np
import time

from mini_mip_system.client.grpc_agg_client import GRPCClient
from server import available_clients
from library.under_development.stat_models.federated_cachedStatistical import FederatedStatsClientCached


def start_client(*, aggregation_server="localhost:50051", client_id: int, metric="mean"):
    rng = np.random.default_rng(42 * (client_id + 1))

    # local x
    values = rng.normal(loc=100 + 5 * client_id, scale=10.0, size=1_000_000)

    # δεύτερο vector για pairwise metrics
    y = 2.5 * values + rng.normal(loc=0.0, scale=5.0, size=values.shape[0])

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

    start = time.time()

    if metric == "mean":
        result = model.fit_mean(values, num_rounds=30)

    elif metric == "variance":
        result = model.fit_variance(values, num_rounds=30)

    elif metric == "std":
        result = model.fit_std(values, num_rounds=30)

    elif metric == "rms":
        result = model.fit_rms(values, num_rounds=30)

    elif metric == "covariance":
        result = model.fit_covariance(values, y, num_rounds=30)

    elif metric == "pearson":
        result = model.fit_pearson(values, y, num_rounds=30)

    elif metric == "regression":
        result = model.fit_regression(values, y, num_rounds=30)

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    end = time.time()

    metric_agg_flags = [flag for m, flag in model.did_aggregate if m == metric]
    metric_global_hist = [val for m, val in model.global_history if m == metric]
    metric_local_hist = [val for m, val in model.local_history if m == metric]

    print(f"\n[Client {client_id}] metric={metric}")

    if metric == "mean":
        local_val = float(values.mean())
        print(f"[Client {client_id}] local_mean={local_val:.6f}")
        print(f"[Client {client_id}] global_mean_est={result:.6f}")
        print(f"[Client {client_id}] cached_predict={model.predict('mean'):.6f}")

    elif metric == "variance":
        local_val = float(np.var(values))
        print(f"[Client {client_id}] local_variance={local_val:.6f}")
        print(f"[Client {client_id}] global_variance_est={result:.6f}")
        print(f"[Client {client_id}] cached_predict={model.predict('variance'):.6f}")

    elif metric == "std":
        local_val = float(np.std(values))
        print(f"[Client {client_id}] local_std={local_val:.6f}")
        print(f"[Client {client_id}] global_std_est={result:.6f}")
        print(f"[Client {client_id}] cached_predict={model.predict('std'):.6f}")

    elif metric == "rms":
        local_val = float(np.sqrt(np.mean(values ** 2)))
        print(f"[Client {client_id}] local_rms={local_val:.6f}")
        print(f"[Client {client_id}] global_rms_est={result:.6f}")
        print(f"[Client {client_id}] cached_predict={model.predict('rms'):.6f}")

    elif metric == "covariance":
        local_val = float(np.mean((values - values.mean()) * (y - y.mean())))
        print(f"[Client {client_id}] local_covariance={local_val:.6f}")
        print(f"[Client {client_id}] global_covariance_est={result:.6f}")
        print(f"[Client {client_id}] cached_predict={model.predict('covariance'):.6f}")

    elif metric == "pearson":
        local_std_x = float(np.std(values))
        local_std_y = float(np.std(y))
        if local_std_x == 0 or local_std_y == 0:
            local_val = 0.0
        else:
            local_cov = float(np.mean((values - values.mean()) * (y - y.mean())))
            local_val = float(local_cov / (local_std_x * local_std_y))

        print(f"[Client {client_id}] local_pearson={local_val:.6f}")
        print(f"[Client {client_id}] global_pearson_est={result:.6f}")
        print(f"[Client {client_id}] cached_predict={model.predict('pearson'):.6f}")

    elif metric == "regression":
        local_var_x = float(np.var(values))
        if local_var_x == 0:
            local_slope, local_intercept = 0.0, float(np.mean(y))
        else:
            local_cov = float(np.mean((values - values.mean()) * (y - y.mean())))
            local_slope = float(local_cov / local_var_x)
            local_intercept = float(np.mean(y) - local_slope * np.mean(values))

        cached_reg = model.predict("regression")
        print(f"[Client {client_id}] local_regression=(slope={local_slope:.6f}, intercept={local_intercept:.6f})")
        print(f"[Client {client_id}] global_regression=(slope={result[0]:.6f}, intercept={result[1]:.6f})")
        if cached_reg is not None:
            print(f"[Client {client_id}] cached_predict=(slope={cached_reg[0]:.6f}, intercept={cached_reg[1]:.6f})")

    print(f"[Client {client_id}] did_aggregate_count={sum(metric_agg_flags)}/{len(metric_agg_flags)}")
    print(f"[Client {client_id}] last_local={metric_local_hist[-1]}")
    print(f"[Client {client_id}] last_global={metric_global_hist[-1]}")
    print(f"[Client {client_id}] time_ms={(end - start) * 1000:.2f}")