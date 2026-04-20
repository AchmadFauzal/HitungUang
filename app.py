import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(
    page_title="Rupiah Vision AI",
    page_icon="💰",
    layout="wide"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: gray;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model (Caching agar tidak reload setiap kali script dijalankan)
@st.cache_resource
def load_yolo_model():
    return YOLO("best_model.pt")

model = load_yolo_model()

nominal_map = {
    0: 1000, 1: 10000, 2: 100000, 
    3: 2000, 4: 20000, 5: 5000, 6: 50000
}

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.bi.go.id/id/fungsi-utama/pengedaran-uang/penerbitan-uang/Uang%20Tahun%20Emisi/Uang%20Kertas%20TE%202022.jpg", use_column_width=True)
    st.title("⚙️ Pengaturan")
    mode = st.radio("Metode Input:", ["📂 Upload Gambar", "📸 Kamera"])
    st.info("Aplikasi ini mendeteksi nominal uang Rupiah menggunakan teknologi AI (YOLOv8).")
    
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4)

# --- FUNGSI PROSES ---
def process_image(image):
    # Prediksi dengan threshold yang bisa diatur
    results = model(image, conf=conf_threshold)
    boxes = results[0].boxes

    # Plot hasil deteksi
    img_plotted = results[0].plot(labels=True, conf=True)
    img_rgb = cv2.cvtColor(img_plotted, cv2.COLOR_BGR2RGB)

    classes = boxes.cls.cpu().numpy().astype(int)
    counter = Counter(classes)

    total = 0
    details = []

    # Sortir kelas agar output rapi dari nominal terbesar/terkecil
    for kelas in sorted(counter.keys()):
        jumlah = counter[kelas]
        nominal = nominal_map[kelas]
        subtotal = nominal * jumlah
        total += subtotal
        details.append({
            "nominal": nominal,
            "jumlah": jumlah,
            "subtotal": subtotal
        })

    return img_rgb, details, total

# --- MAIN CONTENT ---
st.title("💰 Rupiah Vision AI")
st.markdown("### Solusi Cerdas Hitung Uang Rupiah Cepat & Akurat")
st.divider()

source_img = None
if mode == "📂 Upload Gambar":
    uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg","png","jpeg"])
    if uploaded_file:
        source_img = Image.open(uploaded_file)
else:
    camera_image = st.camera_input("Ambil Foto Uang")
    if camera_image:
        source_img = Image.open(camera_image)

if source_img is not None:
    col1, col2 = st.columns([3, 2])
    
    image_np = np.array(source_img)
    with st.spinner('Menganalisis gambar...'):
        result_img, details, total = process_image(image_np)

    with col1:
        st.subheader("🖼️ Hasil Deteksi")
        st.image(result_img, use_column_width=True, caption="Hasil Pemindaian AI")

    with col2:
        st.subheader("📊 Ringkasan")
        
        # Tampilan Metric yang Mencolok
        st.metric(label="Total Nilai Uang", value=f"Rp {total:,.0f}")
        
        if details:
            st.write("---")
            for item in details:
                with st.expander(f"💵 Rp {item['nominal']:,}"):
                    st.write(f"**Jumlah:** {item['jumlah']} lembar")
                    st.write(f"**Subtotal:** Rp {item['subtotal']:,}")
        else:
            st.warning("Uang tidak terdeteksi. Coba atur ulang threshold atau perbaiki pencahayaan.")

# Footer
st.markdown("""
    <div class="footer">
        Dibuat oleh Kelompok Ibu Kota Jawa Barat
    </div>
    """, unsafe_allow_html=True)