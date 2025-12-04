import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.title('Diabetes Prediction App')

pregrancicies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=0)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=0)
skin_thickness=st.number_input('Skin Thickness', min_value=0, max_value=100, value=0)
insulin = st.number_input('Insulin', min_value=0, max_value=500, value=0)
bmi=st.number_input('BMI', min_value=0.0, max_value=50.0, value=0.0)
pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.0)
age = st.number_input('Age', min_value=0, max_value=120, value=0)


model = joblib.load(f'Diabetes Prediction\diabetes.pkl')
if st.button('Predict',type='secondary'):
    X = pd.DataFrame([pregrancicies,glucose,blood_pressure,skin_thickness,insulin,bmi,pedigree,age]).T
    pred =model.predict(X)
    st.success('Diabetes : '+('Yes' if pred[0] == 1 else 'No'))


    