import numpy as np
from library.under_development.stat_models.federated_metric_cached import FederatedMetricClientCached
from mini_mip_system.client.grpc_agg_client import GRPCClient
from server import available_clients
import time

def start_client(*, aggregation_server="localhost:50051", client_id):
    rng = np.random.default_rng(42 * (client_id + 1))

    # "dataset" για mean: 1D values
    # κάθε client έχει λίγο διαφορετική κατανομή
    values = rng.normal(loc=20 + client_id, scale=2.0, size=10000)

    client: GRPCClient = GRPCClient(
        client_id, available_clients, operation_id=0,
        aggregation_server=aggregation_server
    )

    model = FederatedMetricClientCached(client)

    start_time = time.time()
    global_mean = model.fit_mean(values, num_rounds=100)
    end_time = time.time()

    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"[Client {client_id}] Global mean estimate: {global_mean:.4f}")
    print("Time (ms):", elapsed_time_ms)
