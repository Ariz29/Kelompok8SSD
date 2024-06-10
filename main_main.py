import streamlit as st
import pandas as pd
import pickle

# Load data
@st.cache
def load_data():
    data = pd.read_csv("nepal-earthquake-severity-index-latest (1).csv")
    return data

# Save the trained SVM model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Load the trained SVM model
def load_model(filename):
    with open(filename, 'rb') as file:
        svm_model = pickle.load(file)
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

        # Train and save the SVM model
        features = ['Hazard (Intensity)', 'Exposure', 'Housing', 'Poverty', 'Vulnerability', 'Severity', 'Severity Normalized']
        X = data[features]
        y = data['Severity category']
        svm_model = SVC(kernel='linear')
        svm_model.fit(X, y)
        save_model(svm_model, 'svm_model.pkl')

        # Prediction button
        if st.button("Predict"):
            # Load the SVM model
            loaded_model = load_model('svm_model.pkl')

            # Predict severity category
            prediction = loaded_model.predict([[hazard_intensity, exposure, housing, poverty, vulnerability, severity, severity_normalized]])

            # Display prediction
            st.write("## Prediction")
            st.write(f"Predicted Severity Category: {prediction[0]}")

if __name__ == "__main__":
    main()
