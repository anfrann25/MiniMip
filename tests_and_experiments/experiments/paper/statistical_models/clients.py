from sklearn.datasets import make_classification

from library.under_development.stat_models.logistic_regression_saga_solver import FederatedLogisticRegressionClientSaSo
from library.under_development.stat_models.federated_logreg_cached import FederatedLogisticRegressionClientCached
from mini_mip_system.client.grpc_agg_client import GRPCClient
from server import available_clients
import time


import time
from sklearn.datasets import make_classification

from library.under_development.stat_models.logistic_regression_saga_solver import FederatedLogisticRegressionClientSaSo
from library.under_development.stat_models.federated_logreg_cached import FederatedLogisticRegressionClientCached
from mini_mip_system.client.grpc_agg_client import GRPCClient
from server import available_clients


def start_client(*, aggregation_server="localhost:50051", client_id):
    # Δημιουργία dataset
    x, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42 * client_id
    )

    # Δημιουργία GRPC client
    client = GRPCClient(
        client_id,
        available_clients,
        operation_id=0,
        aggregation_server=aggregation_server
    )

    # -------------------------------
    # 1. SaSo MODEL
    # -------------------------------
    model_saso = FederatedLogisticRegressionClientSaSo(client)

    start_saso = time.time()
    model_saso.fit(x, y, num_epochs=100)
    end_saso = time.time()

    time_saso_ms = (end_saso - start_saso) * 1000
    print(f"[SaSo] Time: {time_saso_ms:.2f} ms")

    # -------------------------------
    # 2. Cached MODEL
    # -------------------------------
    model_cached = FederatedLogisticRegressionClientCached(client)

    start_cached = time.time()
    model_cached.fit(x, y, num_epochs=100)
    end_cached = time.time()

    time_cached_ms = (end_cached - start_cached) * 1000
    print(f"[Cached] Time: {time_cached_ms:.2f} ms")

    #compare
    print("Cache model: ", time_cached_ms)
    print("SaSo model: ", time_saso_ms)
    if time_saso_ms < time_cached_ms:
        print("Faster model: SaSo")
    elif time_cached_ms < time_saso_ms:
        print("Faster model: Cached")
    else:
        print("Both models have the same speed!")

    return time_saso_ms, time_cached_ms


# -------------------------------
# Run example
# -------------------------------
if __name__ == "__main__":
    start_client(client_id=1)
    start_client(client_id=2)
    start_client(client_id=3)
