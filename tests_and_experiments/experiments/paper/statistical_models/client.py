import numpy as np
import time
from mini_mip_system.client.grpc_agg_client import GRPCClient
from server import available_clients
from library.under_development.stat_models.federated_cachedStatistical import FederatedStatsClientCached

def start_client(*, aggregation_server="localhost:50051", client_id):
    rng = np.random.default_rng(42 * (client_id + 1))
    values = rng.normal(loc=100 + 5*client_id, scale=10.0, size=10000)

    client = GRPCClient(client_id, available_clients, operation_id=0, aggregation_server=aggregation_server)
    model = FederatedStatsClientCached(client)

    start = time.time()
    gmean = model.fit_mean(values, num_rounds=100)
    end = time.time()

    print(f"[Client {client_id}] Global mean: {gmean}")
    print("Time (ms):", (end-start)*1000)
