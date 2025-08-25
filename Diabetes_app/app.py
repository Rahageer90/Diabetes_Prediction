import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin

class XGBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb_model, imputer=None, scaler=None):
        self.xgb_model = xgb_model
        self.imputer = imputer
        self.scaler = scaler

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X_proc = X
        if self.imputer:
            X_proc = self.imputer.transform(X_proc)
        if self.scaler:
            X_proc = self.scaler.transform(X_proc)
        dmatrix = xgb.DMatrix(X_proc)
        return (self.xgb_model.predict(dmatrix) > 0.5).astype(int)

    def predict_proba(self, X):
        X_proc = X
        if self.imputer:
            X_proc = self.imputer.transform(X_proc)
        if self.scaler:
            X_proc = self.scaler.transform(X_proc)
        dmatrix = xgb.DMatrix(X_proc)
        probs = self.xgb_model.predict(dmatrix)
        return np.vstack([1 - probs, probs]).T

pipeline = joblib.load("best_pipeline.pkl")

st.set_page_config(page_title="Diabetes Detector", page_icon="🩺", layout="wide")
st.title("Diabetes Detector App")
st.markdown("Enter details to predict the risk of diabetes:")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 0, 1)
    glucose = st.number_input("Glucose", 0, 300, 95, 1)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70, 1)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 23, 1)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 85, 1)
    bmi = st.number_input("BMI", 0.0, 70.0, 28.0, 0.1)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.25, 0.01)
    age = st.number_input("Age", 1, 120, 28, 1)

input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]],
                          columns=["Pregnancies","Glucose","BloodPressure","SkinThickness",
                                   "Insulin","BMI","DiabetesPedigreeFunction","Age"])

if st.button("Predict Diabetes"):
    pred = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][1]

    col1, col2 = st.columns(2)
    if pred == 1:
        col1.error("High risk of Diabetes !!!")
        col2.progress(int(prob * 100))
        st.write(f"Confidence: {prob:.2f}")
    else:
        col1.success("Low risk of Diabetes :)")
        col2.progress(int((1 - prob) * 100))
        st.write(f"Confidence of the probability: {1 - prob:.2f}")

with st.expander("About this prediction"):
    st.write("""
   This application works by machine learning model (XGBoost) and can make mistakes
    """)
