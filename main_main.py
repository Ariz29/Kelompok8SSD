import streamlit as st
import pandas as pd
import joblib

# Load data
@st.cache
def load_data():
    data = pd.read_csv("nepal-earthquake-severity-index-latest (1).csv")
    return data

# Load the trained SVM model
def load_model():
    with open('svm_model.pkl', 'rb') as file:
        svm_model = joblib.load(file)
    return svm_model

def main():
    st.title("Nepal Earthquake Severity Prediction")
    data = load_data()

    st.sidebar.title("Options")
    analysis_choice = st.sidebar.selectbox("Choose Analysis", ("Data Exploration", "Model Building"))

    if analysis_choice == "Data Exploration":
        st.title("Data Exploration")

        # Display the DataFrame
        st.write("## Raw Data")
        st.write(data)

        # Descriptive statistics
        st.write("## Descriptive Statistics")
        st.write(data.describe().T)

        # Missing values
        st.write("## Missing Values")
        st.write(data.isnull().sum())

        # Duplicated rows
        st.write("## Duplicated Rows")
        st.write(data.duplicated().sum())

    elif analysis_choice == "Model Building":
        st.title("Model Building")

        # Input fields for features
        st.write("## Input Features")
        hazard_intensity = st.number_input("Hazard (Intensity)", value=0.0)
        exposure = st.number_input("Exposure", value=0.0)
        housing = st.number_input("Housing", value=0.0)
        poverty = st.number_input("Poverty", value=0.0)
        vulnerability = st.number_input("Vulnerability", value=0.0)
        severity = st.number_input("Severity", value=0.0)
        severity_normalized = st.number_input("Severity Normalized", value=0.0)

        # Load the SVM model
        svm_model = load_model()

        # Check if input values exceed capacity
        if any(val > 1e6 for val in [hazard_intensity, exposure, housing, poverty, vulnerability, severity, severity_normalized]):
            st.error("Input value exceeds capacity. Please enter a value within the valid range.")

        else:
            # Prediction button
            if st.button("Predict"):
                # Predict severity category
                prediction = svm_model.predict([[hazard_intensity, exposure, housing, poverty, vulnerability, severity, severity_normalized]])

                # Display prediction
                st.write("## Prediction")
                st.write(f"Predicted Severity Category: {prediction[0]}")

if __name__ == "__main__":
    main()
