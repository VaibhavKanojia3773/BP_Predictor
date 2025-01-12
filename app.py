import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load saved models
def load_models():
    
    systolic_model = joblib.load('models/final_pipeline_systolic_model.pkl')
    diastolic_model = joblib.load('models/final_pipeline_diastolic_model.pkl')
    return systolic_model, diastolic_model

    
final_pipeline_systolic, final_pipeline_diastolic = load_models()

# Function to predict systolic and diastolic blood pressure
def predict_blood_pressure(age, height, weight, heart_rate, bmi, mean, std, max_pressure, min_pressure, median,
                           hypertension, diabetes, cerebral_infarction, cerebrovascular_disease):
    
    # Convert categorical inputs to binary (0 or 1)
    hypertension = 1 if hypertension == 'Yes' else 0
    diabetes = 1 if diabetes == 'Yes' else 0
    cerebral_infarction = 1 if cerebral_infarction == 'Yes' else 0
    cerebrovascular_disease = 1 if cerebrovascular_disease == 'Yes' else 0
    
    # Create input data as DataFrame (same structure as training data)
    input_data = pd.DataFrame({
        'Age(year)': [age],
        'Height(cm)': [height],
        'Weight(kg)': [weight],
        'Heart Rate(b/m)': [heart_rate],
        'BMI(kg/m^2)': [bmi],
        'mean': [mean],
        'std': [std],
        'max': [max_pressure],
        'min': [min_pressure],
        'median': [median],
        'Hypertension': [hypertension],
        'Diabetes': [diabetes],
        'cerebral infarction': [cerebral_infarction],
        'cerebrovascular disease': [cerebrovascular_disease]
    })

    # Predict systolic and diastolic blood pressure
    systolic_prediction = final_pipeline_systolic.predict(input_data)
    diastolic_prediction = final_pipeline_diastolic.predict(input_data)

    # Prepare the plot
    fig, ax = plt.subplots()
    ax.bar(['Systolic BP', 'Diastolic BP'], [systolic_prediction[0], diastolic_prediction[0]], color=['blue', 'green'])
    ax.set_ylabel('Blood Pressure (mmHg)')
    ax.set_ylim(0, 200)  # Set y-axis limits for better visualization
    
    # Save the plot to a file and return the file path
    plt.savefig('bp_prediction.png')
    
    return systolic_prediction[0], diastolic_prediction[0], 'bp_prediction.png'

# Streamlit UI
def main():
    st.title('Blood Pressure Prediction System')
    st.write('Enter the following details to predict systolic and diastolic blood pressure.')

    # User input fields
    age = st.number_input('Age (years)', min_value=0, max_value=120, value=30)
    height = st.number_input('Height (cm)', min_value=0, max_value=250, value=170)
    weight = st.number_input('Weight (kg)', min_value=0, max_value=200, value=70)
    heart_rate = st.number_input('Heart Rate (b/m)', min_value=0, max_value=200, value=75)
    bmi = st.number_input('BMI (kg/m^2)', min_value=0.0, max_value=50.0, value=24.0)
    mean = st.number_input('Mean Blood Pressure', min_value=50, max_value=200, value=120)
    std = st.number_input('Blood Pressure Standard Deviation', min_value=0.0, max_value=50.0, value=10.0)
    max_pressure = st.number_input('Max Blood Pressure', min_value=100, max_value=200, value=140)
    min_pressure = st.number_input('Min Blood Pressure', min_value=50, max_value=100, value=80)
    median = st.number_input('Median Blood Pressure', min_value=50, max_value=150, value=110)

    # Categorical Inputs
    hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
    diabetes = st.selectbox('Diabetes', ['Yes', 'No'])
    cerebral_infarction = st.selectbox('Cerebral Infarction', ['Yes', 'No'])
    cerebrovascular_disease = st.selectbox('Cerebrovascular Disease', ['Yes', 'No'])

    # Button to make predictions
    if st.button('Predict Blood Pressure'):
        # Call the prediction function
        systolic, diastolic, plot_path = predict_blood_pressure(
            age,
            height,
            weight,
            heart_rate,
            bmi,
            mean,
            std,
            max_pressure,
            min_pressure,
            median,
            hypertension,
            diabetes,
            cerebral_infarction,
            cerebrovascular_disease
        )

        # Display the results
        st.write(f'Predicted Systolic Blood Pressure: {systolic:.2f} mmHg')
        st.write(f'Predicted Diastolic Blood Pressure: {diastolic:.2f} mmHg')

        # Display the plot
        st.image(plot_path)

# Run the Streamlit app
if __name__ == '__main__':
    main()
