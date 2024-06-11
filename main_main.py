import streamlit as st
import pickle
import streamlit as st
import numpy as np

# Membaca model
earthquake_model = pickle.load(open('gempa_bumi_model.sav', 'rb'))

# Membaca scaler
scaler = pickle.load(open('Robust.pkl', 'rb'))

# Streamlit app
st.title('Nepal Earthquake Severity Classifier')

# Membagi kolom
col1, col2 = st.columns(2)
with col1:
    Hazard = st.text_input('Input Tingkat Resiko Bahaya Gempa')
with col2:
    Exposure = st.text_input('Input Tingkat Akurasi Bahaya Gempa')
with col1:
    Housing = st.text_input('Input Tingkat Kerusakan Rumah Korban Gempa')
with col2:
    Poverty = st.text_input('Input Tingkat Kemiskinan Korban Gempa')
with col1:
    Vulnerability = st.text_input('Input Tingkat Bahaya Titik Gempa')
with col2:
    Severity = st.text_input('Input Tingkat Keparahan Korban Luka-luka')
with col1:
    Severity_Normalized = st.text_input('Input Tingkat Keparahan Korban Meninggal')

# Code untuk prediksi
diab_prediksi = ''

# Membuat tombol untuk prediksi
if st.button('Test Prediksi Gempa Bumi'):
    try:
        # Pastikan input dikonversi ke tipe data yang sesuai
        input_data = np.array([[int(Hazard), int(Exposure), int(Housing),
                                int(Poverty), int(Vulnerability), float(Severity),
                                int(Severity_Normalized)]])

        # Terapkan scaler hanya pada fitur yang sesuai
        features_to_scale = input_data[:, :7]
        scaled_features = scaler.transform(features_to_scale)

        # Gabungkan fitur yang sudah di-scaling dengan fitur yang tidak di-scaling
        combined_input_data = np.hstack((scaled_features, input_data[:, 7:]))

        # Lakukan prediksi
        diab_prediction = earthquake_model.predict(combined_input_data)

        # Asumsikan diab_prediction adalah hasil prediksi dari model
try:
    # Pastikan hasil prediksi diubah menjadi nilai tunggal sebelum dibandingkan
    if diab_prediction[0] in [1, 0]:  # Sesuaikan kondisi ini dengan kategori yang sesuai
        diab_prediksi = 'Kategori: Highest atau High'
    elif diab_prediction[0] == 5:
        diab_prediksi = 'Kategori: Medium-Low'
    elif diab_prediction[0] == 4:
        diab_prediksi = 'Kategori: Medium-High'
    elif diab_prediction[0] == 2:
        diab_prediksi = 'Kategori: Low'
    elif diab_prediction[0] == 3:
        diab_prediksi = 'Kategori: Lowest'
    else:
        diab_prediksi = 'Kategori tidak dikenali'

    st.success(diab_prediksi)
except ValueError as e:
    st.error(f'Error dalam konversi input: {e}')
