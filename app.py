import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Rupiah Vision AI",
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
# MAPPING (PASTIKAN SESUAI TRAINING!)
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
with st.sidebar:
    st.title("⚙️ Pengaturan")
    mode = st.radio("Input:", ["Upload", "Kamera"])
    conf_threshold = st.slider("Confidence", 0.0, 1.0, 0.4)

# =========================
# PROCESS IMAGE (SMOOTH)
# =========================
def process_image(image):
    results = model(image)  # ❗ TANPA CONF
    boxes = results[0].boxes

    img = image.copy()

    total = 0
    counter = Counter()
    details = []

    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # 🔥 FILTER MANUAL (SMOOTH)
            if conf < conf_threshold:
                continue

            counter[cls] += 1

            # DRAW BOX MANUAL
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{nominal_map[cls]} ({conf:.2f})"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # hitung total
    for k in sorted(counter.keys()):
        nominal = nominal_map[k]
        jumlah = counter[k]
        subtotal = nominal * jumlah
        total += subtotal

        details.append({
            "nominal": nominal,
            "jumlah": jumlah,
            "subtotal": subtotal
        })

    return img, details, total

# =========================
# UI
# =========================
st.title("💰 Rupiah Vision AI")
st.caption("Deteksi & Hitung Uang Otomatis")

source = None

if mode == "Upload":
    file = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])
    if file:
        source = Image.open(file)

else:
    cam = st.camera_input("Ambil foto")
    if cam:
        source = Image.open(cam)

# =========================
# RESULT
# =========================
if source is not None:
    col1, col2 = st.columns([3,2])

    img_np = np.array(source)

    with st.spinner("Processing..."):
        result_img, details, total = process_image(img_np)

    with col1:
        st.image(result_img, use_column_width=True)

    with col2:
        st.metric("💰 Total", f"Rp {total:,.0f}")

        if details:
            for d in details:
                st.write(f"Rp {d['nominal']:,} × {d['jumlah']} = Rp {d['subtotal']:,}")
        else:
            st.warning("Tidak ada deteksi")
