from sklearn.datasets import make_classification

from library.under_development.stat_models.logistic_regression_saga_solver import FederatedLogisticRegressionClientSaSo
from library.under_development.stat_models.federated_logreg_cached import FederatedLogisticRegressionClientCached
from mini_mip_system.client.grpc_agg_client import GRPCClient
from server import available_clients
import time



def start_client(*,aggregation_server="localhost:50051",client_id):
    x, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, n_classes=2,
                               random_state=42*client_id)
    client: GRPCClient = GRPCClient(client_id, available_clients, operation_id=0, aggregation_server=aggregation_server)
    # Creating the federated and the centralized versions of the same dataset
    #model:FederatedLogisticRegressionClientSaSo = FederatedLogisticRegressionClientSaSo(client)
    model:FederatedLogisticRegressionClientCached = FederatedLogisticRegressionClientCached(client)
    # Timing starts
    start_time = time.time()
    model.fit(x, y, num_epochs=100)
    # Timing ends
    end_time = time.time()
    # Calculate elapsed time in milliseconds
    elapsed_time_ms = (end_time - start_time) * 1000
    print("Time: ", elapsed_time_ms)