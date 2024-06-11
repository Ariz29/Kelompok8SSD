import streamlit as st
import pandas as pd
import pickle
import os

current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, 'svm_model.pkl')
    
with open(file_path, 'rb') as f:
# Proses pembacaan file
except FileNotFoundError as e:
print(f"File not found: {e}")
# Tindakan tambahan jika diperlukan, seperti log kesalahan atau fallback logic


svm_model = pickle.load(f)

# Streamlit app
st.title('Nepal Earthquake Severity Classifier')

# Sidebar untuk input data
st.sidebar.header('Input Data')

# Menerima input dari pengguna
input_data = {}
for feature in features:
    input_data[feature] = st.sidebar.number_input(f'{feature}', value=0.0)

# Predict button
if st.sidebar.button('Predict'):
    # Prepare data for prediction
    input_df = pd.DataFrame([input_data])

    # Perform prediction
    prediction = svm_model.predict(input_df)

    # Display prediction result
    st.write('## Prediction:')
    st.write(f'The predicted severity category is: {prediction[0]}')
