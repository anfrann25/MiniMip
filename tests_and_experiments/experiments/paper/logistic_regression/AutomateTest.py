import warnings
import time
from multiprocessing import Process, Manager
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import pandas as pd

# Import federated models and grpc client
from mini_mip_system.client.grpc_agg_client import GRPCClient
from library.under_development.stat_models.logistic_regression_saga_solver import FederatedLogisticRegressionClientSaSo
from library.under_development.stat_models.federated_logreg_cached import FederatedLogisticRegressionClientCached
from mini_mip_system.server.grpc_agg_server import serve
from server import available_clients

# Import custom dataset
from tests_and_experiments.datasets import iris as data
from tests_and_experiments.datasets import heart_disease as data1

# -------------------------------
# Ignore convergence warnings
# -------------------------------
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -------------------------------
# Client function
# -------------------------------
def run_client(model_type, client_id, x_train, y_train, x_test, y_test, results_dict, acc_dict, aggregation_server="localhost:50051"):
    client = GRPCClient(
        client_id,
        available_clients,
        operation_id=0,
        aggregation_server=aggregation_server
    )

    if model_type == "SaSo":
        model = FederatedLogisticRegressionClientSaSo(client)
    elif model_type == "Cached":
        model = FederatedLogisticRegressionClientCached(client)
    else:
        raise ValueError("Unknown model type")

    start_time = time.time()
    model.fit(x_train, y_train, num_epochs=100, X_val=x_test, y_val=y_test)
    end_time = time.time()

    elapsed_ms = (end_time - start_time) * 1000
    print(f"[Client {client_id} - {model_type}] Time: {elapsed_ms:.2f} ms")
    results_dict[f"{model_type}_client_{client_id}"] = elapsed_ms
    acc_dict[f"{model_type}_client_{client_id}"] = model.accuracy_history

    # Compute accuracy
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    print(f"[Client {client_id} - {model_type}] Test Accuracy: {acc:.4f}")
    acc_dict[f"{model_type}_client_{client_id}"] = model.accuracy_history

# -------------------------------
# Server function
# -------------------------------
def start_server(available_clients):
    print("Starting asyncio server...")
    import asyncio
    asyncio.run(serve(available_clients=available_clients))

# -------------------------------
# Function to run a model group
# -------------------------------
def run_model_group(model_type, client_ids, x_test, y_test):
    manager = Manager()
    results = manager.dict()
    accuracies = manager.dict()

    # Start server
    server_process = Process(target=start_server, args=(available_clients,))
    server_process.start()
    time.sleep(2)  # wait for server to be ready

    # Start clients
    processes = []
    for cid in client_ids:
        p = Process(target=run_client, args=(
            model_type,
            cid,
            X_splits[cid - 1],  # train split per client
            y_splits[cid - 1],
            X_test,  # shared test set
            y_test,
            results,
            accuracies
        ))

        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    server_process.terminate()
    server_process.join()
    print(f"{model_type} server stopped.\n")

    return dict(results), dict(accuracies)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # Load custom Iris dataset
    # iris_data = data.IrisDataset()
    # df = iris_data.get_dataset()


    ins_data = data1.HeartDisease()
    df = ins_data.get_dataset()

    print(df.columns)
    # Δες τις πρώτες 5 γραμμές
    print(df.head())

    # Δες τις στήλες και τον τύπο τους
    print(df.info())

    # Δες βασικές στατιστικές
    print(df.describe())
    # Target column
    target_col = 'HeartDisease'

    cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    X = pd.get_dummies(df.drop('HeartDisease', axis=1), columns=cat_cols, drop_first=True).values
    y = df['HeartDisease'].values

    from sklearn.model_selection import StratifiedKFold


    def stratified_split(X, y, num_clients=3):
        skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)
        X_splits, y_splits = [], []
        for _, idx in skf.split(X, y):
            X_splits.append(X[idx])
            y_splits.append(y[idx])
        return X_splits, y_splits




    # Features και target
    # X = df.drop("DEATH_EVENT", axis=1).values  # DEATH_EVENT είναι binary target (0 ή 1)
    # y = df["DEATH_EVENT"].values


    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create splits
    X_splits, y_splits = stratified_split(X, y, num_clients=3)

    client_ids = [1, 2, 3]

    # -------------------------------
    # Run SaSo clients
    # -------------------------------
    print("=== Running SaSo clients ===")
    saso_results, saso_acc = run_model_group("SaSo", client_ids, X_test, y_test)
    print("SaSo results:", saso_results)
    print("SaSo accuracy:", saso_acc, "\n")

    # -------------------------------
    # Run Cached clients
    # -------------------------------
    print("=== Running Cached clients ===")
    cached_results, cached_acc = run_model_group("Cached", client_ids, X_test, y_test)
    print("Cached results:", cached_results)
    print("Cached accuracy:", cached_acc, "\n")

    # -------------------------------
    # Plot average execution time
    # -------------------------------
    mean_saso = np.mean(list(saso_results.values()))
    mean_cached = np.mean(list(cached_results.values()))

    models = ['SaSo', 'Cached']
    mean_times = [mean_saso, mean_cached]
    colors = ['skyblue', 'salmon']

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, mean_times, color=colors)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 10, f"{height:.0f} ms", ha='center', va='bottom')
    ax.set_ylabel("Average execution time (ms)")
    ax.set_title("Average execution time per model")
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # Plot average accuracy
    # -------------------------------
    mean_saso_acc = np.mean(list(saso_acc.values()))
    mean_cached_acc = np.mean(list(cached_acc.values()))

    mean_acc = [mean_saso_acc, mean_cached_acc]
    colors = ['skyblue', 'salmon']

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, mean_acc, color=colors)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{height:.4f}", ha='center', va='bottom')
    ax.set_ylabel("Average Test Accuracy")
    ax.set_title("Average Accuracy per Model")
    plt.tight_layout()
    plt.show()


    # -------------------------------
    # Line plot Accuracy ανά epoch
    # -------------------------------
    # -------------------------------
    # Line plot Accuracy ανά epoch (global)
    # -------------------------------
    plt.figure(figsize=(10, 6))

    # Πάρε την πρώτη (ή οποιαδήποτε) γιατί όλες είναι ίδιες
    global_acc = list(cached_acc.values())[0]

    plt.plot(range(1, len(global_acc) + 1), global_acc, label="Global Cached Accuracy", linestyle="-", color="green")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Global Accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()
