import streamlit as st
import pickle
import pandas as pd
import os
import zipfile

import urllib.request
url = 'https://drive.google.com/file/d/1SUWrKamPl2fEj_cLyy9XCKt_vkOKdnFM/view?usp=drive_link'
urllib.request.urlretrieve(url, 'diabetes_model.pkl')

with open("diabetes_model.pkl", "rb") as f:
    model, feature_names = pickle.load(f)

# LabelEncoderek betöltése
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Ez a része kódolja a megfelelő LabelEncoder-eket
le_gender = encoders[0]
le_smoking = encoders[1]
le_hypertension = encoders[2] if len(encoders) > 2 else None
le_heart_disease = encoders[3] if len(encoders) > 2 else None

st.title("Diabétesz Előrejelző Webalkalmazás")

# Felhasználói bemenetek
gender = st.selectbox("Nem:", le_gender.classes_)
smoking = st.selectbox("Dohányzás történet:", le_smoking.classes_)

if le_hypertension is not None:
    hypertension_input = st.selectbox("Magas vérnyomás:", le_hypertension.classes_)
else:
    hypertension_input = st.selectbox("Magas vérnyomás:", ["nem", "igen"])  # vagy default bináris

if le_heart_disease is not None:
    heart_disease_input = st.selectbox("Szívbetegség:", le_heart_disease.classes_)
else:
    heart_disease_input = st.selectbox("Szívbetegség:", ["nem", "igen"])  # vagy default bináris

age = st.slider("Életkor:", 0, 100, 25)
bmi = st.number_input("BMI:", min_value=10.0, max_value=60.0, value=22.5)
hba1c = st.number_input("HbA1c szint:", min_value=3.0, max_value=15.0, value=5.5)
blood_glucose = st.number_input("Vércukorszint:", min_value=50.0, max_value=300.0, value=100.0)

if st.button("Előrejelzés"):
    # Kategóriák kódolása
    gender_enc = le_gender.transform([gender])[0]
    smoking_enc = le_smoking.transform([smoking])[0]
    if le_hypertension is not None:
        hypertension_enc = le_hypertension.transform([hypertension_input])[0]
    else:
        hypertension_enc = 1 if hypertension_input == "igen" else 0
    
    if le_heart_disease is not None:
        heart_disease_enc = le_heart_disease.transform([heart_disease_input])[0]
    else:
        heart_disease_enc = 1 if heart_disease_input == "igen" else 0

    # Bemeneti dictionary
    input_dict = {
        "gender": gender_enc,
        "age": age,
        "smoking_history": smoking_enc,
        "hypertension": hypertension_enc,
        "heart_disease": heart_disease_enc,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": blood_glucose
    }

    # DataFrame létrehozása a feature_names sorrendjében
    input_data = pd.DataFrame([[input_dict[col] for col in feature_names]], columns=feature_names)


    # Előrejelzés
    prediction = model.predict(input_data)[0]

    # Eredmény megjelenítése
    if prediction == 1:
        st.error("Cukorbetegség lehetséges")
    else:
        st.success("Nincs valószínú cukorbetegség")
