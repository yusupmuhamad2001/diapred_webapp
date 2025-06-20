import streamlit as st #untuk membuat web
import numpy as np #untuk komputasi numerik dlm python #untuk memproses data
import pandas as pd #analisis data
import joblib #menyimpan model machine learning yang sudah di latih
import os #menyimpan dan mengelola file yang di unggah
import logging #untuk mencatat berbagai aktivitas dan monitoring aplikasi
import plotly.express as px #visualisasi data
from config import MODEL_PATH, SCALER_PATH, HISTORY_FILE, AGE_RANGE, BMI_RANGE, HBA1C_RANGE, GLUCOSE_RANGE

# Konfigurasi logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Function to load global model and scaler (for single predict)
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler not found! Please ensure the files are available in the 'models/' folder.")
        return None, None
    except Exception as e:
        st.error(f"Error while loading the model or scaler: {e}")
        return None, None

# Function for prediction
def predict_diabetes(input_data, model, scaler):
    try:
        # Validasi range nilai
        if not (AGE_RANGE[0] <= input_data[1] <= AGE_RANGE[1]):  # age
            raise ValueError("Usia harus antara 0-120 tahun")
        if not (BMI_RANGE[0] <= input_data[5] <= BMI_RANGE[1]):  # BMI
            raise ValueError("BMI harus antara 10-50")
        if not (HBA1C_RANGE[0] <= input_data[6] <= HBA1C_RANGE[1]):   # HbA1c
            raise ValueError("HbA1c harus antara 3-15")
        if not (GLUCOSE_RANGE[0] <= input_data[7] <= GLUCOSE_RANGE[1]):  # Blood glucose
            raise ValueError("Glukosa darah harus antara 50-300")
            
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)
        return 'Diabetes' if prediction[0] == 1 else 'Non-Diabetes'
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Function to calculate BMI
def calculate_bmi(weight, height):
    try:
        # BMI = weight (kg) / (height (m) ^ 2)
        bmi = weight / (height / 100) ** 2
        return round(bmi, 2)
    except Exception as e:
        st.error(f"Error during BMI calculation: {e}")
        return None
    
# Function to save prediction history
def save_to_history(data):
    try:
        history_file = HISTORY_FILE
        
        # Mengubah nama kolom dari bahasa Inggris ke Indonesia
        data.columns = ['Nama', 'Jenis Kelamin', 'Usia', 'Hipertensi', 'Penyakit Jantung', 
                       'Riwayat Merokok', 'BMI', 'Level HbA1c', 'Glukosa Darah', 'Hasil']
        
        # Menyimpan ke CSV
        if os.path.exists(history_file):
            # Jika file sudah ada, tambahkan tanpa header
            data.to_csv(history_file, mode='a', header=False, index=False)
        else:
            # Jika file belum ada, buat baru dengan header
            data.to_csv(history_file, mode='w', header=True, index=False)
            
        st.success("✅ Data berhasil disimpan!")
        
    except Exception as e:
        st.error("❌ Gagal menyimpan riwayat prediksi. Silakan coba lagi.")
        logging.error(f"Error saving history: {str(e)}")

# Function to load prediction history
def load_history():
    history_file = HISTORY_FILE
    if os.path.exists(history_file):
        return pd.read_csv(history_file)
    else:
        return pd.DataFrame(columns=['Name', 'Gender', 'Age', 'Hypertension', 'Heart Disease', 
                                     'Smoking History', 'BMI', 'HbA1c Level', 'Blood Glucose', 'Result'])

def validate_name(name):
    if not name.strip():
        return False, "Nama pasien tidak boleh kosong!"
    if len(name) < 2:
        return False, "Nama pasien terlalu pendek!"
    if not name.replace(" ", "").isalpha():
        return False, "Nama hanya boleh mengandung huruf!"
    return True, ""

def validate_input(data):
    errors = []
    
    # Validasi nama
    if len(data['name']) < 2:
        errors.append("Nama terlalu pendek")
    
    # Validasi usia
    if not (0 <= data['age'] <= 120):
        errors.append("Usia tidak valid")
    
    # Validasi BMI
    if not (10 <= data['bmi'] <= 50):
        errors.append("BMI tidak valid")
    
    return errors

def get_recommendations(result, bmi, glucose, hba1c, smoking_status, hypertension, heart_disease, age):
    recommendations = []
    
    # Rekomendasi berdasarkan status merokok
    if smoking_status == 'Perokok Aktif':
        recommendations.extend([
            "***Status: Perokok Aktif 🚬***",
            "- Sangat disarankan untuk berhenti merokok karena meningkatkan risiko komplikasi diabetes",   
        ])

    elif smoking_status == 'Mantan Perokok':
        recommendations.extend([
            "***Status: Mantan Perokok 🚬***",
            "- Pertahankan untuk tidak merokok kembali dan hindari paparan asap rokok pasif"
        ])
    
    elif smoking_status == 'Tidak Pernah':
        recommendations.extend([
            "***Status: Tidak Pernah Merokok ✨***",
            "- Pertahankan gaya hidup bebas rokok Anda!",
        ])
    
    # Rekomendasi untuk hipertensi
    if hypertension:
        recommendations.extend([
            "Status: Memiliki Hipertensi ⚠️",
            "- Batasi konsumsi garam (<2300mg/hari)",
            "- Hindari makanan tinggi sodium",
            "- Konsumsi makanan kaya potasium seperti pisang dan alpukat"
        ])
    
    # Rekomendasi untuk penyakit jantung
    if heart_disease:
        recommendations.extend([
            "***Status: Memiliki Penyakit Jantung ❤️***",
            "- Rutin kontrol ke dokter jantung",
            "- Batasi aktivitas fisik berat",
            "- Konsumsi makanan rendah lemak jenuh",
            "- Hindari stres berlebihan"
        ])
    
    if result == 'Diabetes':
        # Rekomendasi spesifik berdasarkan glukosa
        if glucose > 200:
            recommendations.extend([
                f"***Glukosa {glucose} mg/dL (>200) 🔴***",
                "- Kadar gula darah Anda sangat tinggi",
                "- Kontrol gula darah secara teratur"
                "- Periksa kadar gula darah setiap hari",
                "- Segera Konsultasikan dengan dokter"
            ])
        elif glucose > 150:
            recommendations.extend([
                f"Glukosa {glucose} mg/dL (>150) 🟡",
                "- Waspada! Kadar gula darah Anda mulai tinggi",
                "- Mulai batasi makanan manis dan karbohidrat tinggi",
                "- Tingkatkan aktivitas fisik minimal 30 menit per hari",
                "- Lakukan pemeriksaan gula darah rutin",
                "- Konsultasi dengan ahli gizi untuk penyesuaian pola makan"
            ])
        
        # Rekomendasi spesifik berdasarkan BMI
        if bmi > 30:
            recommendations.extend([
                f"***BMI {bmi:.1f} (Obesitas) ⚠️***",
                "- Program penurunan berat badan intensif",
                "- Segera konsultasi dengan ahli gizi"
            ])
        elif bmi > 25:
            recommendations.extend([
                f"***BMI {bmi:.1f} (Overweight) ⚠️***",
                "- Program penurunan berat badan moderat",
                "- Disarankan untuk konsultasi dengan ahli gizi"
            ])
        
        # Rekomendasi spesifik berdasarkan HbA1c
        if hba1c > 8:
            recommendations.extend([
                f"***HbA1c {hba1c}% (>8) 🔴***",
                "- Kadar HbA1c sangat tinggi",
                "- Segera konsultasikan dengan dokter",
            ])
        elif hba1c > 6.5:
            recommendations.extend([
                f"***HbA1c {hba1c}% (>6.5) 🟡***",
                "- Kadar HbA1c di atas normal",
                "- Disarankan untuk konsultasi dengan dokter"
            ])
        
        # Rekomendasi umum untuk penderita diabetes
        recommendations.extend([
            "Rekomendasi Umum Diabetes:",
            "- Kunjungi Dokter untuk pemeriksaan lebih lanjut",
            "- Olahraga minimal 30 menit/hari",
            "- Batasi konsumsi karbohidrat dan gula",
            "- Konsumsi makanan tinggi serat",
            "- Pantau gula darah secara rutin"
        ])
    else:
        # Rekomendasi untuk non-diabetes dengan faktor risiko
        if bmi > 30:
            recommendations.extend([
                f"***BMI {bmi:.1f} (Obesitas) ⚠️***",
                "- Risiko tinggi diabetes",
                "- Program penurunan berat badan diperlukan",
                "- Konsultasi dengan dokter atau ahli gizi"
            ])
        elif bmi > 25:
            recommendations.extend([
                f"***BMI {bmi:.1f} (Overweight) ⚠️***",
                "- Risiko diabetes meningkat",
                "- Pertimbangkan penurunan berat badan"
            ])
        
        if glucose > 140:
            recommendations.extend([
                f"***Glukosa {glucose} mg/dL (>140) ⚠️***",
                "- Waspadai pre-diabetes",
                "- Periksa gula darah secara berkala"
            ])
        
        recommendations.extend([
            "***Rekomendasi Pencegahan Diabetes:***", 
            "- Kunjungi Dokter untuk pemeriksaan lebih lanjut",
            "- Jaga Pola hidup sehat",
            "- Olahraga minimal 150 menit per minggu",
            "- Jaga Pola makan seimbang",
            "- Monitoring gula darah secara rutin"
        ])
    
    return recommendations

def show_history_analytics(history):
    st.write("### Visualisasi Data")
    
    # Cek apakah ada data riwayat
    if history.empty:
        st.info("⚠️ Belum ada data riwayat prediksi. Silakan lakukan prediksi terlebih dahulu.")
        return
    
    try:
        # Hitung jumlah untuk setiap hasil
        results_count = history['Hasil'].value_counts()
        total_predictions = len(history)
        
        # Buat dua kolom untuk informasi dan pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Tampilkan informasi dalam card/box dengan ukuran teks yang lebih kecil
            st.markdown("""
            <div style='background-color: #1e3d59; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 100%; display: flex; flex-direction: column; justify-content: center;'>
                <h4 style='text-align: center; color: #ffffff; margin-bottom: 15px; font-size: 16px;'>Ringkasan Prediksi</h4>
                <h2 style='text-align: center; color: #ffffff; margin-bottom: 15px; font-size: 24px;'>{} Orang</h2>
                <hr style='margin: 15px 0; border-color: rgba(255,255,255,0.2);'>
                <p style='font-size: 16px; color: #ffffff; text-align: center; margin: 10px 0;'>🔴 Diabetes: {} orang</p>
                <p style='font-size: 16px; color: #ffffff; text-align: center; margin: 10px 0;'>🟢 Non-Diabetes: {} orang</p>
            </div>
            """.format(total_predictions, 
                      results_count.get('Diabetes', 0),
                      results_count.get('Non-Diabetes', 0)), 
            unsafe_allow_html=True)
        
        with col2:
            # Pie chart hasil prediksi
            fig1 = px.pie(values=results_count.values, 
                         names=results_count.index,
                         title="Distribusi Hasil Prediksi",
                         color_discrete_sequence=['#118B50', '#FF2929'])
            fig1.update_layout(
                title_x=0.5,
                title_font_size=16,
                height=300,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig1, use_container_width=True)
            
    except KeyError:
        st.warning("⚠️ Format data riwayat tidak sesuai. Silakan pastikan data prediksi tersimpan dengan benar.")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan dalam menampilkan visualisasi: {str(e)}")
        logging.error(f"Error in show_history_analytics: {str(e)}")

def get_bmi_recommendations(bmi):
    recommendations = []
    
    if bmi < 18.5:
        recommendations.extend([
            f"***BMI {bmi:.1f} (Anda Kekurangan Berat Badan) ⚠️***",
            "- Tingkatkan asupan kalori dengan makanan bergizi",
            "- Konsumsi protein berkualitas tinggi",
            "- Lakukan olahraga secara teratur",
            "- Konsultasikan dengan ahli gizi untuk program penambahan berat badan yang sehat"
        ])
    elif 18.5 <= bmi < 24.9:
        recommendations.extend([
            f"***BMI {bmi:.1f} (Berat Badan Anda Normal) ✅***",
            "- Pertahankan pola makan seimbang",
            "- Lakukan olahraga rutin minimal 150 menit per minggu",
            "- Jaga kualitas tidur yang baik",
            "- Lanjutkan gaya hidup sehat yang sudah dijalani"
        ])
    elif 25 <= bmi < 29.9:
        recommendations.extend([
            f"***BMI {bmi:.1f} (Anda Kelebihan Berat Badan) ⚠️***",
            "- Kurangi porsi makan secara bertahap",
            "- Tingkatkan aktivitas fisik menjadi 45-60 menit per hari",
            "- Hindari makanan tinggi gula dan lemak jenuh",
            "- Pertimbangkan untuk berkonsultasi dengan ahli gizi",
        ])
    else:  # BMI >= 30
        recommendations.extend([
            f"***BMI {bmi:.1f} (Anda Obesitas) 🔴***",
            "- Segera konsultasi dengan dokter atau ahli gizi",
            "- Mulai program penurunan berat badan yang aman",
            "- Olahraga secara teratur selama minimal 30 menit setiap hari",
            "- Catat asupan makanan harian",
            "- Periksa kesehatan secara rutin",
            "- Hindari makanan dan minuman tinggi lemak"
        ])
            
    return recommendations

def show_about():
    st.write("## Tentang Aplikasi Prediksi Diabetes")
    
    st.write("### Deskripsi")
    st.write("""
    Aplikasi ini adalah alat bantu untuk melakukan prediksi awal risiko diabetes berdasarkan beberapa parameter kesehatan. 
    Hasil prediksi hanya bersifat petunjuk awal dan TIDAK menggantikan diagnosis medis profesional.
    """)
    
    st.write("### ⚠️ Penting!")
    st.write("""
    - Hasil prediksi ini BUKAN diagnosis final dan TIDAK dapat dijadikan sebagai keputusan medis.
    - Konsultasi dengan dokter atau tenaga medis profesional TETAP DIPERLUKAN untuk diagnosis yang akurat.
    - Aplikasi ini hanya bertujuan sebagai alat skrining awal.
    """)
    
    st.write("### 📝 Cara Penggunaan")
    st.write("""
    1. **Menu Prediksi**
       - Isi semua data yang diminta dengan lengkap dan akurat
       - Pastikan data yang dimasukkan sesuai 
       - Tekan tombol 'Prediksi' untuk melihat hasil
       - Baca rekomendasi yang diberikan sebagai panduan umum
    
    2. **Menu Hitung BMI**
       - Masukkan berat badan (kg) dan tinggi badan (cm)
       - Sistem akan menghitung BMI dan memberikan kategori serta rekomendasi

    """)
        
    #3. **Menu Riwayat**
       #- Lihat riwayat prediksi sebelumnya
       #- Unduh data riwayat dalam format CSV
       #- Analisis tren hasil prediksi

# Main function for Streamlit
def main():
    st.title("Prediksi Diabetes 🩺")
    st.sidebar.title("Menu")
    menu = st.sidebar.radio("Pilih Menu:", ["Prediksi", "Hitung BMI", "Tentang Aplikasi"])

    # Load model dan scaler
    model, scaler = load_model_and_scaler()

    if menu == "Prediksi":
        st.write("Masukkan data berikut untuk prediksi:")
        col1, col2, col3 = st.columns(3)

        with col1:
            name = st.text_input('Nama Pasien', value='')
            gender = st.selectbox('Jenis Kelamin', ['Perempuan', 'Laki-laki'])
            age = st.number_input('Usia', min_value=0, max_value=120, step=1, value=30)

        with col2:
            hypertension = st.selectbox('Riwayat Hipertensi', ['Tidak', 'Ya'])
            heart_disease = st.selectbox('Riwayat Penyakit Jantung', ['Tidak', 'Ya'])
            smoking_history = st.selectbox('Riwayat Merokok', ['Tidak Pernah', 'Mantan Perokok', 'Perokok Aktif'])

        with col3:
            bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, step=0.1, value=25.0)
            hba1c_level = st.number_input('Level HbA1c', min_value=3.0, max_value=15.0, step=0.1, value=5.0)
            blood_glucose_level = st.number_input('Level Glukosa Darah', min_value=50, max_value=300, step=1, value=100)

        # Konversi input
        gender = 1 if gender == 'Laki-laki' else 0
        hypertension = 1 if hypertension == 'Ya' else 0
        heart_disease = 1 if heart_disease == 'Ya' else 0
        smoking_history_map = {'Tidak Pernah': 0, 'Mantan Perokok': 2, 'Perokok Aktif': 1}
        smoking_history = smoking_history_map[smoking_history]

        input_data = [gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level]

        if model and scaler:
            if st.button("Prediksi"):
                is_valid, error_msg = validate_name(name)
                if not is_valid:
                    st.error(error_msg)
                else:
                    result = predict_diabetes(input_data, model, scaler)
                    if result:
                        if result == 'Diabetes':
                            st.markdown("<div style='background-color: #ff4d4d; color: white; padding: 10px; border-radius: 5px; text-align: center;'>HASIL SCREENING: TERINDIKASI DIABETES</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='background-color: #4caf50; color: white; padding: 10px; border-radius: 5px; text-align: center;'>HASIL SCREENING: TERINDIKASI NON-DIABETES</div>", unsafe_allow_html=True)
                        
                        # Tampilkan rekomendasi
                        st.write("### Rekomendasi Kesehatan:")
                        recommendations = get_recommendations(
                            result, 
                            bmi, 
                            blood_glucose_level, 
                            hba1c_level,
                            list(smoking_history_map.keys())[list(smoking_history_map.values()).index(smoking_history)],
                            hypertension == 1,
                            heart_disease == 1,
                            age
                        )
                        for rec in recommendations:
                            st.write(f"- {rec}")

                        # Buat DataFrame untuk history
                        data = pd.DataFrame({
                            'Nama': [name],
                            'Jenis Kelamin': ['Laki-laki' if gender == 1 else 'Perempuan'],
                            'Usia': [age],
                            'Hipertensi': ['Ya' if hypertension == 1 else 'Tidak'],
                            'Penyakit Jantung': ['Ya' if heart_disease == 1 else 'Tidak'],
                            'Riwayat Merokok': [list(smoking_history_map.keys())[list(smoking_history_map.values()).index(smoking_history)]],
                            'BMI': [bmi],
                            'Level HbA1c': [hba1c_level],
                            'Glukosa Darah': [blood_glucose_level],
                            'Hasil': [result]
                        })
                        
                        save_to_history(data)

    elif menu == "Hitung BMI":
        weight = st.number_input("Berat Badan (kg)", min_value=1.0, step=0.1, value=70.0)
        height = st.number_input("Tinggi Badan (cm)", min_value=1.0, step=0.1, value=170.0)

        if st.button("Hitung BMI"):
            bmi = calculate_bmi(weight, height)
            if bmi:
                st.success(f"BMI Anda adalah {bmi}")
                if bmi < 18.5:
                    st.info("Kategori: Kekurangan Berat Badan")
                elif 18.5 <= bmi < 24.9:
                    st.info("Kategori: Normal")
                elif 25 <= bmi < 29.9:
                    st.info("Kategori: Kelebihan Berat Badan")
                else:
                    st.info("Kategori: Obesitas")
                
                st.write("### Rekomendasi:")
                recommendations = get_bmi_recommendations(bmi)
                for rec in recommendations:
                    st.write(f"- {rec}")

    #elif menu == "Riwayat":
        #st.write("Riwayat Prediksi:")
        #history = load_history()
        #if history.empty:
            #st.info("Belum ada riwayat prediksi.")
        #else:
            # Terjemahkan nama kolom
            #history.columns = ['Nama', 'Jenis Kelamin', 'Usia', 'Hipertensi', 'Penyakit Jantung', 
                             #'Riwayat Merokok', 'BMI', 'Level HbA1c', 'Glukosa Darah', 'Hasil']
            
            # Tampilkan data dengan nomor index
            #st.write("### Data Riwayat Prediksi")
            #history.index = range(1, len(history) + 1)
            #st.dataframe(history)
            
            # Tombol unduh riwayat
            #csv = history.to_csv(index=False)
            #st.download_button(
                #label="Unduh Riwayat (CSV)",
                #data=csv,
                #file_name='riwayat_prediksi_diabetes.csv',
                #mime='text/csv'
            #)

            #if st.button("Visualisasi Riwayat"):
                #show_history_analytics(history)

    elif menu == "Tentang Aplikasi":
        show_about()

if __name__ == '__main__':
    main()
