from sklearn.datasets import make_classification

from library.under_development.stats.bivariate_statistics import PearsonCorrelation
from mini_mip_system.client.grpc_agg_client import GRPCClient
from server import available_clients
import time



def start_client(*,aggregation_server="localhost:50051",client_id):
    rand_data,_ = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, n_classes=2,
                               random_state=42*client_id)
    x = rand_data[:, 0]  # First column
    y = rand_data[:, 1]  # Second column

    client: GRPCClient = GRPCClient(client_id, available_clients, operation_id=0, aggregation_server=aggregation_server)
    # Creating the federated and the centralized versions of the same dataset
    model:PearsonCorrelation = PearsonCorrelation(client)
    # Timing starts
    start_time = time.time()
    model.compute(x, y)
    # Timing ends
    end_time = time.time()
    # Calculate elapsed time in milliseconds
    elapsed_time_ms = (end_time - start_time) * 1000
    print(elapsed_time_ms)

if __name__ == "__main__":
    start_client(client_id=0)