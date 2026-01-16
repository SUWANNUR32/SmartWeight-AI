import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import google.generativeai as genai

# -----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & TEMA BIRU-PUTIH
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartWeight AI + Gemini",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #333333; }
    section[data-testid="stSidebar"] { background-color: #F0F8FF; }
    h1, h2, h3 { color: #0056b3 !important; }
    div.stButton > button:first-child { background-color: #007BFF; color: white; border-radius: 10px; }
    div[data-testid="stMetricValue"] { color: #007BFF; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOGIKA BACKEND (MODEL PREDIKSI & CLASS)
# -----------------------------------------------------------------------------

@st.cache_resource
def train_model():
    """Melatih model dummy sekali saja"""
    np.random.seed(42)
    n_samples = 1000
    berat_awal = np.random.uniform(60, 120, n_samples)
    berat_target = berat_awal - np.random.uniform(5, 30, n_samples)
    defisit = np.random.uniform(300, 1000, n_samples)
    
    kg_to_lose = berat_awal - berat_target
    days_needed = (kg_to_lose * 7700) / defisit
    days_needed_noise = days_needed * np.random.uniform(0.9, 1.1, n_samples)

    X = pd.DataFrame({'berat_awal': berat_awal, 'berat_target': berat_target, 'defisit': defisit})
    y = days_needed_noise

    model = LinearRegression()
    model.fit(X, y)
    return model

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
        self.status_text = ""

    def analisis_kesehatan(self):
        tinggi_m = self.tinggi / 100
        self.bmi = self.berat / (tinggi_m ** 2)
        if self.bmi < 18.5: 
            self.status_text = "Kekurangan Berat Badan"
            return self.status_text, "⚠️"
        elif 18.5 <= self.bmi < 24.9: 
            self.status_text = "Normal (Sehat)"
            return self.status_text, "✅"
        elif 25 <= self.bmi < 29.9: 
            self.status_text = "Kelebihan Berat Badan"
            return self.status_text, "⚠️"
        else: 
            self.status_text = "Obesitas"
            return self.status_text, "🚨"

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

# -----------------------------------------------------------------------------
# 3. FUNGSI GEMINI AI (MODIFIKASI UTAMA)
# -----------------------------------------------------------------------------
def tanya_gemini(api_key, user_obj, user_question, estimasi_hari):
    """
    Mengirim data user + pertanyaan ke Gemini untuk dianalisis
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('AIzaSyCz5BdfTRra0dWiS7nS666CUwfmyFlouhM')
        
        # PROMPT ENGINEERING: Memberi konteks data user ke AI
        prompt_system = f"""
        Kamu adalah Asisten Kesehatan Pribadi bernama 'SmartWeight AI'.
        Tugasmu adalah menjawab pertanyaan pengguna dan memberikan saran diet berdasarkan data berikut:
        
        DATA PENGGUNA:
        - Nama: {user_obj.nama}
        - Jenis Kelamin: {user_obj.gender}
        - Usia: {user_obj.usia} tahun
        - Berat Sekarang: {user_obj.berat} kg
        - Berat Target: {user_obj.target} kg
        - BMI: {user_obj.bmi:.2f} ({user_obj.status_text})
        - Kebutuhan Kalori (TDEE): {int(user_obj.tdee)} kkal
        - Target Kalori Diet Harian: {user_obj.daily_calories} kkal
        - Estimasi Waktu Mencapai Target: {estimasi_hari} hari

        INSTRUKSI:
        1. Jawablah dengan ramah, memotivasi, dan profesional.
        2. Gunakan Bahasa Indonesia yang baik.
        3. Selalu rujuk data di atas (misal: "Karena BMI kamu X...", "Dengan kalori Y...").
        4. Jika user minta menu, buatkan variasi menu yang sesuai dengan target kalori {user_obj.daily_calories} kkal.
        
        PERTANYAAN USER: {user_question}
        """
        
        response = model.generate_content(prompt_system)
        return response.text
    except Exception as e:
        return f"Maaf, terjadi kesalahan koneksi dengan Gemini AI: {e}"

# -----------------------------------------------------------------------------
# 4. USER INTERFACE
# -----------------------------------------------------------------------------

# --- SIDEBAR ---
with st.sidebar:
    st.title("Pengaturan")
    # Input API Key
    gemini_api_key = st.text_input("🔑 Masukkan Google Gemini API Key", type="password", help="Dapatkan di aistudio.google.com")
    
    st.markdown("---")
    st.header("Data Pengguna")
    nama = st.text_input("Nama", "Ani")
    gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    usia = st.number_input("Usia", 15, 90, 30)
    tinggi = st.number_input("Tinggi (cm)", 100, 250, 160)
    berat = st.number_input("Berat Awal (kg)", 30, 200, 75)
    target = st.number_input("Target Berat (kg)", 30, 200, 65)
    
    akt_map = {"Sangat Jarang":1, "Jarang (1-3x)":2, "Sedang (3-5x)":3, "Aktif (6-7x)":4, "Atlet":5}
    akt_label = st.select_slider("Aktivitas Fisik", options=list(akt_map.keys()), value="Jarang (1-3x)")
    aktivitas = akt_map[akt_label]
    
    tombol_analisis = st.button("🚀 Analisis Sekarang")

# --- MAIN PAGE ---
st.title("SmartWeight AI + Gemini 🧠")

if tombol_analisis:
    # Inisialisasi & Hitung
    user = SmartWeightAI(nama, usia, gender, berat, tinggi, aktivitas, target)
    status_text, icon = user.analisis_kesehatan()
    user.hitung_kalori()
    hari = user.prediksi_waktu(model_prediksi)
    minggu = round(hari/7, 1)

    # Simpan object user ke session state agar bisa diakses chatbot nanti
    st.session_state['user_data'] = user
    st.session_state['estimasi_hari'] = hari
    st.session_state['analisis_done'] = True
else:
    if 'analisis_done' not in st.session_state:
        st.info("👈 Masukkan data di sidebar dan klik 'Analisis Sekarang' untuk memulai.")
        st.stop()
    else:
        # Load data dari session state jika tombol tidak ditekan ulang (saat chat)
        user = st.session_state['user_data']
        hari = st.session_state['estimasi_hari']
        status_text, icon = user.analisis_kesehatan() # refresh status text variable
        minggu = round(hari/7, 1)

# TAMPILAN DASHBOARD
col1, col2, col3 = st.columns(3)
with col1: st.metric("BMI Anda", f"{user.bmi:.1f}", f"{status_text} {icon}")
with col2: st.metric("Target Kalori", f"{user.daily_calories} kkal", "-500 kkal")
with col3: st.metric("Estimasi Waktu", f"{minggu} Minggu", f"~{hari} Hari")

st.markdown("---")
tab1, tab2 = st.tabs(["📊 Grafik Progres", "🤖 Konsultasi AI (Gemini)"])

with tab1:
    st.subheader("Proyeksi Penurunan Berat Badan")
    weeks_data = int(minggu) + 1
    x = list(range(weeks_data))
    y = np.linspace(berat, target, weeks_data)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, marker='o', color='#007BFF', linewidth=2)
    ax.axhline(y=target, color='red', linestyle='--', label='Target')
    ax.set_title(f"Target {target} kg dalam {weeks_data} minggu")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab2:
    st.subheader("Konsultasi Personal dengan AI")
    st.write("Tanyakan apa saja tentang diet, analisis data Anda, atau minta resep makanan!")
    
    # Cek API Key
    if not gemini_api_key:
        st.warning("⚠️ Masukkan API Key Gemini di sidebar terlebih dahulu untuk menggunakan fitur Chatbot.")
    else:
        # Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Tampilkan chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input Chat User
        if prompt := st.chat_input("Contoh: Buatkan menu makan siang 500 kalori..."):
            # 1. Tampilkan pesan user
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Proses dengan Gemini
            with st.chat_message("assistant"):
                with st.spinner("AI sedang menganalisis data Anda..."):
                    response_text = tanya_gemini(gemini_api_key, user, prompt, hari)
                    st.markdown(response_text)
            
            # 3. Simpan respon AI
            st.session_state.messages.append({"role": "assistant", "content": response_text})
