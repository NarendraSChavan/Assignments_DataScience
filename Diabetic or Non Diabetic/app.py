import streamlit as st
import pickle 
import pandas as pd


#Load the trained logisitic regresssion model
try:
  with open('logistic_regression_model.pkl','rb') as file:
    model = pickle.load(file)
except FileNotFoundError:
  st.error("Error: 'logistic_regression_model.pkl' not found. Please ensure the model is saved in the same directory.")
  st.stop()

#Set the title of the Streamlit application
st.title('Diabetes Prediction App')
st.write('Enter the patient information below to predict the likelihood of diabetes.')


pregnancies = st.slider("Number of months Pregnancies",0,17,3)
glucose = st.slider("Glucose Level (mg/dL)",44,199,117)
blood_pressure = st.slider("Blood Pressure (mmHg)",24,122,72)
skin_thickness = st.slider("Skin Thickness (mm)",7,99,29)
insulin = st.slider("Insulin Level (muU/ml)",14,846,125)
bmi = st.number_input("BMI",min_value=18.2,max_value=67.1,value=32.3,step=0.1)
dpf = st.number_input("Diabetes Pedigree Function",min_value=0.078,max_value=2.42,value=0.472,step=0.001,format="%.3f")
age = st.slider("Age",21,81,33)

user_input = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf,
    'Age': age
}

features_df = pd.DataFrame([user_input])

if st.button('Predict'):
  prediction = model.predict(features_df)
  prediction_proba = model.predict_proba(features_df)[:,1]

  st.subheader('Prediction Result:')
  if prediction[0] == 1:
    st.error(f'The model predicts that the patient is likely to have diabetes with a probability of {prediction_proba[0]:.2f}.')
  else:
    st.success(f'The model predicts that the patient is unlikely to have diabetes with a probability of {1 - prediction_proba[0]:.2f}.')
    st.write('---')
  st.subheader('Feature Importance (from training data):')
  st.write(features_df)