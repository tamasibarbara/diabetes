import streamlit as st
import pickle
import pandas as pd
import os
import zipfile
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Itt add meg a Google Drive fájl azonosítóját (file ID)
file_id = '1p1I98h3k36hGyZtRR_Sim0psqYstE74g'  # ezt a linkből kell kivenni

# Töltsd le, ha még nincs meg a fájl
import os
if not os.path.exists('diabetes_model.pkl'):
    download_file_from_google_drive(file_id, 'diabetes_model.pkl')

# Modell betöltése
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

st.title("Cukorbetegság előrejelzés")

# Felhasználói bemenetek
gender = st.selectbox("Nem:", le_gender.classes_)
smoking = st.selectbox("Dohányzás előfordulása:", le_smoking.classes_)

if le_hypertension is not None:
    hypertension_input = st.selectbox("Magas vérnyomás van-e:", le_hypertension.classes_)
else:
    hypertension_input = st.selectbox("Magas vérnyomás van-e:", ["nem", "igen"])  # vagy default bináris

if le_heart_disease is not None:
    heart_disease_input = st.selectbox("Szívbetegség van-e:", le_heart_disease.classes_)
else:
    heart_disease_input = st.selectbox("Szívbetegség van-e:", ["nem", "igen"])  # vagy default bináris

age = st.slider("Életkor:", 0, 100, 25)
bmi = st.number_input("BMI (min:10,max:60):", min_value=10.0, max_value=60.0, value=22.5)
hba1c = st.number_input("HbA1c szint (min:3,max:15):", min_value=3.0, max_value=15.0, value=5.5)
blood_glucose = st.number_input("Vércukorszint (min:50,max:300):", min_value=50.0, max_value=300.0, value=100.0)

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
        st.success("Nem valószínű cukorbetegség")
