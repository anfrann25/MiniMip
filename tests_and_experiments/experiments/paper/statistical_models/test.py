import numpy as np
import time

from mini_mip_system.client.grpc_agg_client import GRPCClient
from server import available_clients
from library.under_development.stat_models.federated_cachedStatistical import FederatedStatsClientCached

def start_client(*, aggregation_server="localhost:50051", client_id: int):
    rng = np.random.default_rng(42 * (client_id + 1))
    values = rng.normal(loc=100 + 5*client_id, scale=10.0, size=1000000)

    client = GRPCClient(
        client_id,
        available_clients,
        operation_id=99,  # βάλε άλλο από το logistic regression για ασφάλεια
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
    gmean = model.fit_mean(values, num_rounds=30)
    end = time.time()

    # Debug prints
    local_mean = float(values.mean())
    print(f"[Client {client_id}] local_mean={local_mean:.4f}  global_mean_est={gmean:.4f}")
    print(f"[Client {client_id}] did_aggregate_count={sum(model.did_aggregate)}/{len(model.did_aggregate)}")
    print(f"[Client {client_id}] time_ms={(end-start)*1000:.2f}")
