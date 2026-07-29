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
import os

# Import federated models and grpc client
from mini_mip_system.client.grpc_agg_client import GRPCClient
from library.under_development.stat_models.logistic_regression_saga_solver import FederatedLogisticRegressionClientSaSo
from library.under_development.stat_models.federated_logreg_cached import FederatedLogisticRegressionClientCached
from mini_mip_system.server.grpc_agg_server import serve
from server import available_clients

# Import custom datasets
from tests_and_experiments.datasets import iris as iris_data
from tests_and_experiments.datasets import heart_disease as heartd_data
from tests_and_experiments.datasets import smoking as smoking_data
from tests_and_experiments.datasets import water as water_data
from tests_and_experiments.datasets import LungCancer as lungcancer_data
from tests_and_experiments.datasets import csec_iot as ics3d_data
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

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, CompositeVideoClip, AudioFileClip, CompositeAudioClip


def create_text_image(text, size=(500, 100), fontsize=50, color=(0, 0, 0)):
    """Δημιουργεί μια διάφανη εικόνα με κείμενο χρησιμοποιώντας PIL."""
    # Δημιουργία διάφανης εικόνας (RGBA)
    img = Image.new('RGBA', size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Προσπάθεια φόρτωσης γραμματοσειράς Arial - αν δεν υπάρχει, παίρνει την default
    try:
        # Στα Windows η Arial είναι συνήθως διαθέσιμη
        font = ImageFont.truetype("arial.ttf", fontsize)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), text, font=font, fill=color)
    return np.array(img)


def create_federated_video(cached_comm,
                           bg_path="background.png",
                           arrow_path="arrow.png",
                           bg_audio_path="bgaudio.mp3",
                           update_sfx_path="arroaudio.mp3",
                           output_name="federated_training.mp4"):
    # 1. Έλεγχος αν υπάρχουν όλα τα απαραίτητα αρχεία
    required_files = [bg_path, arrow_path, bg_audio_path, update_sfx_path]
    for f in required_files:
        if not os.path.exists(f):
            print(f"❌ Λείπει το αρχείο: {f}")
            return

    duration_per_epoch = 0.4
    total_epochs = 100
    total_duration = total_epochs * duration_per_epoch

    # 2. Φόρτωση και Διόρθωση Φόντου (Ζυγές διαστάσεις για Windows)
    background = ImageClip(bg_path).with_duration(total_duration)
    w, h = background.size
    final_w = w if w % 2 == 0 else w - 1
    final_h = h if h % 2 == 0 else h - 1
    background = background.cropped(x1=0, y1=0, x2=final_w, y2=final_h)

    # Ορισμός θέσεων
    server_pos = (final_w // 2 - 50, 80)
    hospital_positions = {
        "Cached_client_1": (120, 700),
        "Cached_client_2": (450, 700),
        "Cached_client_3": (780, 700)
    }

    overlay_clips = [background]
    audio_clips = []

    # 3. Σταθερός Τίτλος Server
    server_label_img = create_text_image("CENTRAL SERVER", size=(400, 80), fontsize=40, color=(50, 50, 50))
    server_label = (ImageClip(server_label_img)
                    .with_duration(total_duration)
                    .with_position((final_w // 2 - 150, 20)))
    overlay_clips.append(server_label)

    # 4. Υπολογισμός Τρεχούμενου Συνόλου Updates
    updates_per_epoch = [0] * total_epochs
    for client_id, log in cached_comm.items():
        for epoch, status in enumerate(log):
            if status == "FULL_UPDATE":
                updates_per_epoch[epoch] += 1
    cumulative_updates = np.cumsum(updates_per_epoch)

    # 5. Background Audio Loop
    try:
        bg_audio = AudioFileClip(bg_audio_path).with_duration(total_duration).multiply_volume(0.2)
        audio_clips.append(bg_audio)
    except Exception as e:
        print(f"⚠️ Σφάλμα στον ήχο φόντου: {e}")

    # 6. Δημιουργία Δυναμικών Στοιχείων ανά Epoch (Κείμενα, Βέλη, SFX)
    print("⏳ Προετοιμασία Frames και Ήχων...")

    for e in range(total_epochs):
        t_start = e * duration_per_epoch

        # --- Epoch Counter ---
        epoch_img = create_text_image(f"Epoch: {e + 1}", size=(350, 80), fontsize=50, color=(200, 0, 0))
        epoch_clip = (ImageClip(epoch_img)
                      .with_start(t_start)
                      .with_duration(duration_per_epoch)
                      .with_position((40, 40)))
        overlay_clips.append(epoch_clip)

        # --- Updates Counter ---
        upd_text = f"Total Updates: {cumulative_updates[e]}"
        upd_img = create_text_image(upd_text, size=(500, 80), fontsize=40, color=(0, 0, 150))
        upd_clip = (ImageClip(upd_img)
                    .with_start(t_start)
                    .with_duration(duration_per_epoch)
                    .with_position((40, 110)))
        overlay_clips.append(upd_clip)

    # 7. Κίνηση Βελών και Update SFX
    for client_id, log in cached_comm.items():
        if client_id not in hospital_positions: continue
        start_pos = hospital_positions[client_id]

        for epoch, status in enumerate(log):
            if status == "FULL_UPDATE":
                t_start = epoch * duration_per_epoch
                move_duration = duration_per_epoch * 0.9

                # Κινούμενο Βέλος
                arrow = (ImageClip(arrow_path)
                         .with_start(t_start)
                         .with_duration(move_duration)
                         .with_position(lambda t, sp=start_pos, ep=server_pos, dur=move_duration:
                                        (sp[0] + (ep[0] - sp[0]) * (t / dur),
                                         sp[1] + (ep[1] - sp[1]) * (t / dur))))
                overlay_clips.append(arrow)

                # Ήχος Ping
                try:
                    sfx = AudioFileClip(update_sfx_path).with_start(t_start)
                    audio_clips.append(sfx)
                except:
                    pass

    # 8. Σύνθεση και Rendering
    print(f"🎬 Ξεκινάει το rendering του {output_name}...")
    final_video = CompositeVideoClip(overlay_clips, size=(final_w, final_h))

    if audio_clips:
        final_video.audio = CompositeAudioClip(audio_clips)

    final_video.write_videofile(
        output_name,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile='temp-audio.m4a',
        remove_temp=True,
        ffmpeg_params=['-pix_fmt', 'yuv420p']
    )
    print(f"✅ Ολοκληρώθηκε επιτυχώς! Το αρχείο {output_name} είναι έτοιμο.")
# --- Πώς το καλείς στο Main ---
# create_federated_video(cached_comm, "background.png")

# -------------------------------
# Plot clients
# -------------------------------
def plot_communication_timeline_paper(comm_histories, save_path=None):
    if not comm_histories:
        print("⚠ No communication logs found to plot.")
        return

    client_names = sorted(list(comm_histories.keys()))
    num_clients = len(client_names)
    num_epochs = len(comm_histories[client_names[0]])

    # Δημιουργούμε έναν πίνακα (Matrix) με 0 και 1
    grid = np.zeros((num_clients, num_epochs))

    for idx, name in enumerate(client_names):
        for epoch in range(num_epochs):
            if comm_histories[name][epoch] == "FULL_UPDATE":
                grid[idx, epoch] = 1
            else:
                grid[idx, epoch] = 0

    # Στήσιμο του Heatmap Plot
    fig, ax = plt.subplots(figsize=(12, 4))

    # Αφαιρέθηκε το edgecolor από εδώ για να μη χτυπάει σφάλμα
    cax = ax.imshow(grid, aspect='auto', cmap='Blues')

    # Ρυθμίσεις αξόνων
    ax.set_xticks(np.arange(0, num_epochs, 10))
    ax.set_xticklabels(np.arange(0, num_epochs, 10))
    ax.set_yticks(range(num_clients))

    # Ομορφιά στα ονόματα των clients
    clean_labels = [name.replace("Cached_", "").replace("_", " ").title() for name in client_names]
    ax.set_yticklabels(clean_labels)

    ax.set_xlabel("Epoch / Communication Round")
    ax.set_ylabel("Decentralized Edge Nodes")
    ax.set_title("Exaflow Framework: Communication Timeline Matrix")

    # --- ΚΛΕΔΙ ΓΙΑ ΤΟ ΠΛΕΓΜΑ (Grid) ---
    # Βάζουμε διακριτικές γραμμές ανάμεσα στα κουτάκια χωρίς να σκάει ο κώδικας
    ax.set_xticks(np.arange(-0.5, num_epochs, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_clients, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)  # Κρύβουμε τα μικρά ticks

    # Προσθήκη custom επεξήγησης (Legend)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=plt.cm.Blues(180), edgecolor='gray', label='Full Update (Tensor Payload Sent)'),
        Patch(facecolor=plt.cm.Blues(0), edgecolor='lightgray', label='Empty Update (Skipped via Caching)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1.18), ncol=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Το Communication Heatmap αποθηκεύτηκε επιτυχώς στο: {save_path}")
    else:
        plt.show()

def plot_communication_timeline(comm_histories, save_path=None):
    plt.figure(figsize=(12, 6))
    client_names = list(comm_histories.keys())

    for i, name in enumerate(client_names):
        log = comm_histories[name]
        epochs = np.arange(1, len(log) + 1)

        # Points for Full Updates
        full_idx = [e for e, val in zip(epochs, log) if val == "FULL_UPDATE"]
        plt.scatter(full_idx, [i] * len(full_idx), marker='|', s=100, color='blue',
                    label='Full Update' if i == 0 else "")

        # Dotted line for skips
        plt.hlines(i, 1, len(log), colors='gray', linestyles='--', alpha=0.2)

    plt.yticks(range(len(client_names)), client_names)
    plt.xlabel("Epoch")
    plt.title("Client Communication Timeline (Full Updates vs Skips)")
    plt.grid(axis='x', alpha=0.3)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# -------------------------------
# Plot final test accuracy
# -------------------------------
def plot_final_accuracy(saso_final, cached_final, save_path = None):

    mean_saso = np.mean(list(saso_final.values()))
    mean_cached = np.mean(list(cached_final.values()))

    models = ['SaSo', 'Cached']
    mean_acc = [mean_saso, mean_cached]
    colors = ['skyblue', 'salmon']

    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(models, mean_acc, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                height + 0.005,
                f"{height:.4f}",
                ha='center')

    ax.set_ylabel("Final Test Accuracy")
    ax.set_title("Final Accuracy Comparison")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# -------------------------------
# Accuracy comparison per epoch
# -------------------------------
def plot_accuracy_comparison(saso_hist, cached_hist, save_path = None):

    plt.figure(figsize=(10,6))

    saso_mean = np.mean(np.array(list(saso_hist.values())), axis=0)
    cached_mean = np.mean(np.array(list(cached_hist.values())), axis=0)

    plt.plot(saso_mean, label="SaSo", linewidth=2)
    plt.plot(cached_mean, label="Cached", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch Comparison")

    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_average_execution(saso_results, cached_results, save_path = None):
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
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return


    # -------------------------------
    # Plot average accuracy
    # -------------------------------
def plot_average_accuracy(saso_acc, cached_acc, save_path = None):
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
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return

    # -------------------------------
    # Line plot Accuracy per Epoch
    # -------------------------------
def plot_accuracy_per_epoch(cached_acc, save_path = None):
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
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return

#Define plots kai dataset
import os
DO_PLOTS = True     # ή False
DATASET = "ics3d"
SAVE_DIR = "experiment_results/" + DATASET
os.makedirs(SAVE_DIR, exist_ok=True)


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
    elif name == "ics3d":
        ics_dat = ics3d_data.ICS3D()
        df = ics_dat.get_dataset()

        target_col = "Attack_label"

        # sample πρώτα για να μη σκάει
        df = df.sample(8000, random_state=42)

        # αφαιρούμε target και multiclass label
        drop_cols = [target_col, "Attack_type"]

        # high-cardinality / text-like columns που εκτοξεύουν το one-hot
        heavy_cols = [
            "ip.src_host", "ip.dst_host",
            "arp.dst.proto_ipv4", "arp.src.proto_ipv4",
            "http.file_data", "http.request.uri.query",
            "http.referer", "http.request.full_uri",
            "tcp.options", "tcp.payload",
            "dns.qry.name",
            "mqtt.msg_decoded_as", "mqtt.msg",
            "mqtt.protoname", "mqtt.topic"
        ]

        cols_to_drop = [c for c in drop_cols + heavy_cols if c in df.columns]

        y = df[target_col].values
        X_df = df.drop(columns=cols_to_drop)

        # categorical columns
        cat_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

        # numeric conversion
        X_df = X_df.apply(pd.to_numeric, errors="coerce")

        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X_df)
        print("Dataset loaded")
        return X, y
    else:
        raise ValueError("Unknown dataset")

# -------------------------------
# Client function
# -------------------------------
def run_client(model_type, client_id, x_train, y_train, x_test, y_test,
               results_dict, history_dict, final_acc_dict,comm_dict,
               aggregation_server="localhost:50051"):

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

    # Καταγραφή του log αν είναι ο Cached client
    if hasattr(model, "communication_log"):
        comm_dict[f"{model_type}_client_{client_id}"] = model.communication_log

    end_time = time.time()

    elapsed_ms = (end_time - start_time) * 1000

    results_dict[f"{model_type}_client_{client_id}"] = elapsed_ms

    history_dict[f"{model_type}_client_{client_id}"] = model.accuracy_history

    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)

    final_acc_dict[f"{model_type}_client_{client_id}"] = acc

    print(f"[Client {client_id} - {model_type}] Time: {elapsed_ms:.2f} ms")
    print(f"[Client {client_id} - {model_type}] Test Accuracy: {acc:.4f}")


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
    histories = manager.dict()
    final_acc = manager.dict()
    comm_dict = manager.dict() # <--- Προσθήκη

    server_process = Process(target=start_server, args=(available_clients,))
    server_process.start()
    time.sleep(2)

    processes = []

    for cid in client_ids:
        p = Process(target=run_client, args=(
            model_type,
            cid,
            X_splits[cid - 1],
            y_splits[cid - 1],
            X_test,
            y_test,
            results,
            histories,
            final_acc,
            comm_dict # <--- Προσθήκη ορίσματος
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    server_process.terminate()
    server_process.join()

    print(f"{model_type} server stopped.\n")

    return dict(results), dict(histories), dict(final_acc), dict(comm_dict) # <--- Επιστροφή και του comm_dict

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

    # σωστό train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # split ΜΟΝΟ του training set
    X_splits, y_splits = stratified_split(X_train, y_train, num_clients=3)

    client_ids = [1, 2, 3]

    print("=== Running SaSo clients ===")

    saso_results, saso_hist, saso_final, cached_comm = run_model_group(
        "SaSo", client_ids, X_test, y_test
    )

    print("=== Running Cached clients ===")
    # Προσθέτουμε το cached_comm στην αριστερή πλευρά
    cached_results, cached_hist, cached_final, cached_comm = run_model_group(
        "Cached", client_ids, X_test, y_test
    )

    if DO_PLOTS:
        plot_average_execution(saso_results, cached_results, save_path=os.path.join(f"{SAVE_DIR}/execution_time.jpg"))

        plot_final_accuracy(saso_final, cached_final, save_path= os.path.join(f"{SAVE_DIR}/final_accuracy.jpg"))

        plot_accuracy_comparison(saso_hist, cached_hist, save_path= os.path.join(f"{SAVE_DIR}/accuracy_comparison.jpg"))

        plot_accuracy_per_epoch(cached_hist, save_path= os.path.join(f"{SAVE_DIR}/accuracy_per_epoch.jpg"))
        # Νέο γράφημα για το Communication Timeline των Cached clients
        plot_communication_timeline(cached_comm, save_path=os.path.join(f"{SAVE_DIR}/communication_timeline.jpg"))

        plot_communication_timeline_paper(cached_comm, save_path=os.path.join(f"{SAVE_DIR}/communication_timeline_paper.jpg"))

        #create_federated_video(cached_comm, "background.png")






