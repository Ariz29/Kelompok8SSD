import streamlit as st
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Memuat data
@st.cache
def load_data():
    data = pd.read_csv("nepal-earthquake-severity-index-latest (1).csv")
    return data

data = load_data()

# Fitur dan target
features = ['Hazard (Intensity)', 'Exposure', 'Housing', 'Poverty', 'Vulnerability', 'Severity', 'Severity Normalized']
X = data[features]
y = data['Severity category']

# Encoding target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.05, random_state=42)

# Normalisasi fitur
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# Inisialisasi dan melatih model SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Fungsi untuk melakukan prediksi
def predict_severity(input_data):
    # Normalisasi input data
    input_data_scaled = scaler.transform([input_data])
    
    # Melakukan prediksi
    prediction = svm_model.predict(input_data_scaled)
    
    # Mengembalikan kategori severity yang sesuai
    return label_encoder.inverse_transform(prediction)

# Main function
def main():
    st.title("Prediksi Kategori Severity Gempa Bumi di Nepal")
    
    st.write("Masukkan nilai untuk setiap fitur:")
    hazard = st.number_input("Hazard (Intensity)", min_value=0.0)
    exposure = st.number_input("Exposure", min_value=0.0)
    housing = st.number_input("Housing", min_value=0.0)
    poverty = st.number_input("Poverty", min_value=0.0)
    vulnerability = st.number_input("Vulnerability", min_value=0.0)
    severity = st.number_input("Severity", min_value=0.0)
    severity_normalized = st.number_input("Severity Normalized", min_value=0.0)
    
    input_data = [hazard, exposure, housing, poverty, vulnerability, severity, severity_normalized]
    
    if st.button("Prediksi"):
        prediction = predict_severity(input_data)
        st.write(f'Prediksi Kategori Severity: {prediction[0]}')

if __name__ == "__main__":
    main()