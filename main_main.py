from sklearn.externals import joblib

# Train and save your scikit-learn model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Save the model to disk
joblib.dump(svm_model, 'svm_model.pkl')

import streamlit as st
import pandas as pd
from sklearn.externals import joblib

# Load the saved model
svm_model = joblib.load('svm_model.pkl')

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
