import warnings
import time
from multiprocessing import Process, Manager
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
import pandas as pd

# Import federated models and grpc client
from mini_mip_system.client.grpc_agg_client import GRPCClient
from library.under_development.stat_models.logistic_regression_saga_solver import FederatedLogisticRegressionClientSaSo
from library.under_development.stat_models.federated_logreg_cached import FederatedLogisticRegressionClientCached
from mini_mip_system.server.grpc_agg_server import serve
from server import available_clients

# Import custom datasets
from tests_and_experiments.datasets import iris as iris_data
from tests_and_experiments.datasets import   heart_disease as heartd_data
from tests_and_experiments.datasets import smoking as smoking_data
from tests_and_experiments.datasets import water as water_data
from tests_and_experiments.datasets import LungCancer as lungcancer_data
# -------------------------------
# Ignore convergence warnings
# -------------------------------
warnings.filterwarnings("ignore", category=ConvergenceWarning)

cached_acc = {
    "Cached_client_1": [],
    "Cached_client_2": [],
    "Cached_client_3": []
}

saso_acc = {
    "SaSo_client_1": [],
    "SaSo_client_2": [],
    "SaSo_client_3": []
}


def plot_average_execution(saso_results, cached_results):
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

    return


    # -------------------------------
    # Plot average accuracy
    # -------------------------------
def plot_average_accuracy(saso_acc, cached_acc):
    mean_saso_acc = np.mean(list(saso_acc.values()))
    mean_cached_acc = np.mean(list(cached_acc.values()))

    models = ['SaSo', 'Cached']
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

    return

    # -------------------------------
    # Line plot Accuracy per Epoch
    # -------------------------------
def plot_accuracy_per_epoch(cached_acc):
    plt.figure(figsize=(10, 6))

    for cid in [1, 2, 3]:
        history = cached_acc.get(f"Cached_client_{cid}", [])

        if len(history) == 0:
            print(f"⚠ Προσοχή: Client {cid} δεν έχει accuracy history!")
            continue

        plt.plot(range(1, len(history) + 1), history, label=f"Client {cid}")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch (Cached Federated Logistic Regression)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return

#Define plots kai dataset
DO_PLOTS = True     # ή False
DATASET = "water"


def load_dataset(name):
    if name == "heart":
        heart_data = heartd_data.HeartDisease()
        df = heart_data.get_dataset()

        cat_cols = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
        X = pd.get_dummies(df.drop('HeartDisease', axis=1), columns=cat_cols, drop_first=True).values
        y = df['HeartDisease'].values
        return X, y

    elif name == "iris":
        iris_dat = iris_data.IrisDataset()
        df = iris_dat.get_dataset()
        X = df.drop('target', axis=1).values
        y = df['target'].values
        return X, y

    elif name == "smoking":
        smoking_dat = smoking_data.SmokingDataset()
        df = smoking_dat.get_dataset()
        target_col = "smoking"

        if "ID" in df.columns:
            df = df.drop(columns=["ID"])

        # categorical columns (από αυτά που είδαμε)
        cat_cols = ["gender", "oral", "tartar"]

        X = pd.get_dummies(df.drop(target_col, axis=1),columns=cat_cols,drop_first=True).values
        y = df[target_col].values
        return X, y
    elif name == "water":
        water_dat = water_data.WaterQuality()
        df = water_dat.get_dataset()

        target_col = "Potability"
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(df.drop(target_col, axis=1))

        y = df[target_col].values

        return X, y
    elif name == "lung":
        lungcancer_dat = lungcancer_data.LungCancer()
        df = lungcancer_dat.get_dataset()

        target_col = "survived"

        if "id" in df.columns:
            df = df.drop(columns=["id"])

        date_cols = []
        for c in ["diagnosis_date", "end_treatment_date"]:
            if c in df.columns:
                date_cols.append(c)

        for c in date_cols:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df[c] = (df[c] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")

        # 3) X/y
        y = df[target_col].values
        X_df = df.drop(columns=[target_col])

        # 4) One-hot για categorical
        cat_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

        # 5) Impute NaNs (με median) γιατί dates/coerce + missing values
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X_df)

        return X, y
    else:
        raise ValueError("Unknown dataset")

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
# Server function - start server
# -------------------------------
def start_server(available_clients):
    print("Starting asyncio server...")
    import asyncio
    asyncio.run(serve(available_clients=available_clients))

# -------------------------------
# Function to run a model group - run saso or cached
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

# split datasets gia accuracy
from sklearn.model_selection import StratifiedKFold

def stratified_split(X, y, num_clients=3):
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)
    X_splits, y_splits = [], []
    for _, idx in skf.split(X, y):
        X_splits.append(X[idx])
        y_splits.append(y[idx])
    return X_splits, y_splits

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    X, y = load_dataset(DATASET)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

    if DO_PLOTS:
        plot_average_execution(saso_results, cached_results)
        plot_average_accuracy(saso_acc, cached_acc)
        plot_accuracy_per_epoch(cached_acc)






