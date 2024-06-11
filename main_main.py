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
