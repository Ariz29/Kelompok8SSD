import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open('svm_model.pkl', 'rb') as f:
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
