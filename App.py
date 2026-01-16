import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & TEMA BIRU-PUTIH
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartWeight AI",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS untuk Tema Biru Putih
st.markdown("""
    <style>
    /* Background utama putih */
    .stApp {
        background-color: #FFFFFF;
        color: #333333;
    }
    
    /* Sidebar warna biru sangat muda */
    section[data-testid="stSidebar"] {
        background-color: #F0F8FF;
    }
    
    /* Judul dan Header warna Biru Tua */
    h1, h2, h3 {
        color: #0056b3 !important;
    }
    
    /* Tombol Utama (Primary Button) Biru */
    div.stButton > button:first-child {
        background-color: #007BFF;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
    }
    div.stButton > button:first-child:hover {
        background-color: #0056b3;
        color: white;
    }
    
    /* Kotak Metrik */
    div[data-testid="stMetricValue"] {
        color: #007BFF;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOGIKA BACKEND (MODEL AI & CLASS)
# -----------------------------------------------------------------------------

@st.cache_resource
def train_model():
    """Melatih model dummy sekali saja saat aplikasi start"""
    np.random.seed(42)
    n_samples = 1000
    berat_awal = np.random.uniform(60, 120, n_samples)
    berat_target = berat_awal - np.random.uniform(5, 30, n_samples)
    defisit = np.random.uniform(300, 1000, n_samples)
    
    # Rumus simulasi
    kg_to_lose = berat_awal - berat_target
    days_needed = (kg_to_lose * 7700) / defisit
    days_needed_noise = days_needed * np.random.uniform(0.9, 1.1, n_samples)

    X = pd.DataFrame({
        'berat_awal': berat_awal,
        'berat_target': berat_target,
        'defisit': defisit
    })
    y = days_needed_noise

    model = LinearRegression()
    model.fit(X, y)
    return model

# Load Model
model_prediksi = train_model()

class SmartWeightAI:
    def __init__(self, nama, usia, gender, berat, tinggi, aktivitas, target_berat):
        self.nama = nama
        self.usia = usia
        self.gender = gender
        self.berat = berat
        self.tinggi = tinggi
        self.aktivitas = aktivitas
        self.target = target_berat
        self.bmi = 0
        self.bmr = 0
        self.tdee = 0
        self.daily_calories = 0

    def analisis_kesehatan(self):
        tinggi_m = self.tinggi / 100
        self.bmi = self.berat / (tinggi_m ** 2)
        if self.bmi < 18.5: return "Kekurangan Berat Badan", "⚠️"
        elif 18.5 <= self.bmi < 24.9: return "Normal (Sehat)", "✅"
        elif 25 <= self.bmi < 29.9: return "Kelebihan Berat Badan", "⚠️"
        else: return "Obesitas", "🚨"

    def hitung_kalori(self):
        if self.gender == 'Laki-laki':
            self.bmr = (10 * self.berat) + (6.25 * self.tinggi) - (5 * self.usia) + 5
        else:
            self.bmr = (10 * self.berat) + (6.25 * self.tinggi) - (5 * self.usia) - 161

        multipliers = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
        self.tdee = self.bmr * multipliers.get(self.aktivitas, 1.2)
        self.daily_calories = int(self.tdee - 500)
        if self.daily_calories < 1200: self.daily_calories = 1200

    def prediksi_waktu(self, model):
        input_data = pd.DataFrame({
            'berat_awal': [self.berat],
            'berat_target': [self.target],
            'defisit': [self.tdee - self.daily_calories]
        })
        prediksi_hari = model.predict(input_data)[0]
        return int(prediksi_hari)

    def generate_menu(self):
        # Logika sederhana penyesuaian menu
        if self.daily_calories > 1800:
            porsi = "Porsi Besar"
        else:
            porsi = "Porsi Sedang"
            
        return {
            "Sarapan 🍳": f"Oatmeal pisang & putih telur ({porsi})",
            "Makan Siang 🍱": "Nasi merah 100g, Dada ayam bakar, Tumis sayur",
            "Makan Malam 🥗": "Salad sayur dressing lemon & Ikan panggang",
            "Snack 🍎": "Apel atau Yogurt Low Fat"
        }

# -----------------------------------------------------------------------------
# 3. USER INTERFACE (STREAMLIT)
# -----------------------------------------------------------------------------

# --- SIDEBAR INPUT ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=80)
    st.title("Data Pengguna")
    st.write("Isi data diri Anda untuk memulai.")
    
    nama = st.text_input("Nama", "Ani")
    gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    usia = st.number_input("Usia (tahun)", 15, 90, 30)
    tinggi = st.number_input("Tinggi (cm)", 100, 250, 160)
    berat = st.number_input("Berat Awal (kg)", 30, 200, 75)
    target = st.number_input("Target Berat (kg)", 30, 200, 65)
    
    st.markdown("---")
    st.write("**Tingkat Aktivitas:**")
    akt_label = st.select_slider(
        "Seberapa sering Anda bergerak?",
        options=["Sangat Jarang", "Jarang (1-3x)", "Sedang (3-5x)", "Aktif (6-7x)", "Atlet"],
        value="Jarang (1-3x)"
    )
    
    # Mapping label ke angka
    akt_map = {"Sangat Jarang":1, "Jarang (1-3x)":2, "Sedang (3-5x)":3, "Aktif (6-7x)":4, "Atlet":5}
    aktivitas = akt_map[akt_label]
    
    tombol_analisis = st.button("🚀 Analisis Sekarang")

# --- MAIN PAGE ---
st.title("SmartWeight AI 🩺")
st.markdown(f"Selamat datang, **{nama}**! Mari wujudkan berat badan ideal Anda dengan bantuan AI.")

if tombol_analisis:
    # Inisialisasi Object
    user = SmartWeightAI(nama, usia, gender, berat, tinggi, aktivitas, target)
    
    # Proses Perhitungan
    status_text, icon = user.analisis_kesehatan()
    user.hitung_kalori()
    hari = user.prediksi_waktu(model_prediksi)
    minggu = round(hari/7, 1)
    menu = user.generate_menu()

    # --- TAMPILAN METRIK UTAMA ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("BMI Anda", f"{user.bmi:.1f}", f"{status_text} {icon}")
    
    with col2:
        st.metric("Target Kalori Harian", f"{user.daily_calories} kkal", "-500 kkal (Defisit)")
        
    with col3:
        st.metric("Estimasi Waktu", f"{minggu} Minggu", f"~{hari} Hari")

    st.markdown("---")

    # --- TABS: GRAFIK & MENU ---
    tab1, tab2, tab3 = st.tabs(["📊 Proyeksi Grafik", "🍽️ Rekomendasi Menu", "🤖 Konsultasi AI"])

    with tab1:
        st.subheader("Grafik Penurunan Berat Badan")
        
        # Membuat Data Grafik
        weeks_data = int(minggu) + 1
        x = list(range(weeks_data))
        y = np.linspace(berat, target, weeks_data)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y, marker='o', linestyle='-', color='#007BFF', linewidth=2, label='Prediksi AI')
        ax.axhline(y=target, color='red', linestyle='--', alpha=0.5, label='Target')
        ax.set_xlabel("Minggu ke-")
        ax.set_ylabel("Berat (kg)")
        ax.set_title(f"Perjalanan Menuju {target} kg")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ubah background plot agar menyatu dengan tema putih
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        st.pyplot(fig)
        st.info("💡 Grafik ini adalah estimasi linier berdasarkan defisit kalori yang konsisten.")

    with tab2:
        st.subheader("Rencana Makan Harian")
        col_menu1, col_menu2 = st.columns(2)
        
        items = list(menu.items())
        with col_menu1:
            for k, v in items[:2]:
                st.success(f"**{k}**\n\n{v}")
        with col_menu2:
            for k, v in items[2:]:
                st.info(f"**{k}**\n\n{v}")

    with tab3:
        st.subheader("Chatbot Kesehatan")
        user_query = st.text_input("Tanya sesuatu tentang diet Anda:", placeholder="Misal: Bolehkah saya makan gorengan?")
        
        if user_query:
            # Simple Rule-Based Response (Simulasi NLP)
            q = user_query.lower()
            if "gorengan" in q or "minyak" in q:
                ans = "Sebaiknya hindari gorengan. Jika sangat ingin, batasi 1 buah per minggu atau gunakan Air Fryer."
            elif "olahraga" in q:
                ans = "Untuk BMI Anda, jalan cepat 30 menit setiap pagi sangat disarankan."
            elif "lapar" in q:
                ans = "Jika lapar di luar jam makan, minumlah segelas air putih atau makan buah potong."
            else:
                ans = "Pertanyaan bagus! Pastikan Anda tetap konsisten dengan defisit kalori dan tidur yang cukup."
            
            st.write(f"🤖 **AI:** {ans}")

else:
    # Tampilan awal sebelum tombol ditekan
    st.info("👈 Silakan isi data di sidebar sebelah kiri dan tekan tombol 'Analisis Sekarang'.")
