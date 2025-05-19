import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Betöltés: itt a saját csv-dre cseréld le!
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Másoljuk az adatot
data = df.copy()

# Kategóriák kódolása
le_gender = LabelEncoder()
le_smoking = LabelEncoder()

data["gender"] = le_gender.fit_transform(data["gender"])
data["smoking_history"] = le_smoking.fit_transform(data["smoking_history"])

if data["hypertension"].dtype == object:
    le_hypertension = LabelEncoder()
    data["hypertension"] = le_hypertension.fit_transform(data["hypertension"])
else:
    le_hypertension = None  # nincs kódolva, mert már numerikus

if data["heart_disease"].dtype == object:
    le_hypertension = LabelEncoder()
    data["heart_disease"] = le_hypertension.fit_transform(data["heart_disease"])
else:
    le_hypertension = None  # nincs kódolva, mert már numerikus

# Bemeneti jellemzők és célváltozó
X = data.drop("diabetes", axis=1)
y = data["diabetes"]

# Tanító és teszt halmaz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell tanítása
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Modell mentése
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

# Encoderek mentése
if le_hypertension is not None:
    with open("encoders.pkl", "wb") as f:
        pickle.dump((le_gender, le_smoking, le_hypertension), f)
else:
    with open("encoders.pkl", "wb") as f:
        pickle.dump((le_gender, le_smoking), f)