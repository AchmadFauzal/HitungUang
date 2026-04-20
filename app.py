import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
from PIL import Image

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Deteksi Uang Rupiah",
    page_icon="💰",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("best_model.pt")

model = load_model()

# =========================
# MAPPING NOMINAL
# =========================
nominal_map = {
    0: 1000,
    1: 2000,
    2: 5000,
    3: 10000,
    4: 20000,
    5: 50000,
    6: 100000
}

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Pengaturan")

CONF_THRESHOLD = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.5, 0.05
)

mode = st.sidebar.radio(
    "Metode Input",
    ["Upload Gambar", "Kamera"]
)

# =========================
# TITLE
# =========================
st.title("💰 Deteksi & Perhitungan Uang Rupiah")
st.caption("Menggunakan YOLOv8")

# =========================
# PROCESS IMAGE
# =========================
def process_image(image):
    results = model(image)
    boxes = results[0].boxes

    # gambar hasil deteksi
    img = results[0].plot()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    total = 0
    counter = Counter()
    details = []
    debug_data = []

    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # ✅ FILTER CONFIDENCE
            if conf < CONF_THRESHOLD:
                continue

            counter[cls] += 1

            debug_data.append({
                "class_id": cls,
                "nominal": nominal_map.get(cls, 0),
                "confidence": round(conf, 3)
            })

    # hitung total
    for kelas in sorted(counter.keys()):
        nominal = nominal_map.get(kelas, 0)
        jumlah = counter[kelas]
        subtotal = nominal * jumlah
        total += subtotal

        details.append({
            "nominal": nominal,
            "jumlah": jumlah,
            "subtotal": subtotal
        })

    return img, details, total, counter, debug_data

# =========================
# DISPLAY RESULT
# =========================
def show_result(image_np):
    col1, col2 = st.columns([2, 1])

    result_img, details, total, counter, debug_data = process_image(image_np)

    with col1:
        st.image(result_img, caption="Hasil Deteksi", use_column_width=True)

    with col2:
        st.subheader("📊 Ringkasan")

        if len(counter) == 0:
            st.warning("Tidak ada uang terdeteksi")
            return

        for d in details:
            st.write(
                f"💵 Rp {d['nominal']:,} × {d['jumlah']} = Rp {d['subtotal']:,}"
            )

        st.divider()
        st.success(f"💰 TOTAL: Rp {total:,.0f}")

    # DEBUG SECTION
    with st.expander("🔍 Debug Info"):
        st.write(debug_data)

# =========================
# INPUT
# =========================
if mode == "Upload Gambar":
    uploaded_file = st.file_uploader(
        "Upload gambar uang",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        show_result(image_np)

else:
    camera_image = st.camera_input("Ambil foto")

    if camera_image is not None:
        image = Image.open(camera_image)
        image_np = np.array(image)
        show_result(image_np)
