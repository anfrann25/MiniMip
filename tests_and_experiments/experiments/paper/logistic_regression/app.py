import streamlit as st
import time
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

st.set_page_config(page_title="Federated Learning Dashboard", layout="wide")

st.title("🏥 Federated Learning: Communication Efficiency Dashboard")
st.markdown("---")

# --- SIDEBAR: Ρυθμίσεις ---
st.sidebar.header("Παράμετροι Αλγορίθμου")
eps_w = st.sidebar.slider("Variation Threshold (eps_w)", 0.0, 0.5, 0.1, step=0.01)
tau_max = st.sidebar.slider("Max Delay (tau_max)", 1, 50, 20)
start_btn = st.sidebar.button("🚀 Έναρξη Εκπαίδευσης")

# --- LAYOUT: 2 Στήλες ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Οπτικοποίηση Δικτύου")
    status_text = st.empty()
    network_image = st.empty()  # Εδώ θα μπαίνει η εικόνα με τα νοσοκομεία

with col2:
    st.subheader("Στατιστικά Επικοινωνίας")
    metric_comm = st.empty()
    metric_savings = st.empty()
    chart_acc = st.empty()

# --- SIMULATION LOGIC ---
if start_btn:
    # Φόρτωση εικόνας φόντου
    base_img = Image.open("background.png.png").convert("RGBA")

    total_epochs = 100
    updates_count = 0
    accuracy_data = []

    for epoch in range(1, total_epochs + 1):
        # Εδώ "κλέβουμε" λίγο: Προσομοιώνουμε το αν θα γίνει update
        # (Στην πτυχιακή σου θα καλείς το train() του μοντέλου σου εδώ)
        will_update = True if epoch <= 5 or (epoch % 15 == 0) else False

        # Σχεδίαση πάνω στην εικόνα
        temp_img = base_img.copy()
        draw = ImageDraw.Draw(temp_img)

        if will_update:
            updates_count += 1
            # Σχεδιάζουμε πράσινους κύκλους πάνω από τα νοσοκομεία αν μιλάνε
            draw.ellipse([120, 650, 170, 700], fill="green", outline="white")
            draw.ellipse([450, 650, 500, 700], fill="green", outline="white")
            draw.ellipse([780, 650, 830, 700], fill="green", outline="white")
            status_text.warning(f"Epoch {epoch}: 📡 Στάλθηκαν Full Updates!")
        else:
            status_text.info(f"Epoch {epoch}: ⚡ Χρήση Cached Weights (Skip)")

        # Ανανέωση Dashboard
        network_image.image(temp_img, use_column_width=True)

        savings = 100 - (updates_count / epoch * 100)
        metric_comm.metric("Συνολικά Updates", f"{updates_count}")
        metric_savings.metric("Εξοικονόμηση Bandwidth", f"{savings:.1f}%")

        # Fake accuracy chart
        accuracy_data.append(0.6 + (0.3 * (1 - np.exp(-epoch / 20))))
        chart_acc.line_chart(accuracy_data)

        time.sleep(0.1)  # Ταχύτητα simulation

    st.success("✅ Η εκπαίδευση ολοκληρώθηκε!")