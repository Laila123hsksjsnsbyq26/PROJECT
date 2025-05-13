
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
import os
print("Les fichiers dans le dossier:", os.listdir())

st.set_page_config(page_title="Détection Maladies Cardiaques", layout="wide")

st.title(" Application de Détection des Maladies Cardiaques")

# Charger les données
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# Charger le modèle
@st.cache_resource

def load_model():
    return pickle.load(open("Random Forest1", "rb"))

model = load_model()

# Initialiser l'état
if "show_prediction_form" not in st.session_state:
    st.session_state.show_prediction_form = False

# Layout des boutons
analyse, prediction = st.columns(2)

with analyse:
    if st.button("Afficher Analyse & Visualisation"):
        st.session_state.show_prediction_form = False

        st.subheader("Aperçu des données :")
        st.dataframe(df.head())

        st.subheader("Statistiques globales :")
        st.write(df.describe())

        st.subheader("Distribution de HeartDisease :")
        st.bar_chart(df["HeartDisease"].value_counts())

        fig1, ax1 = plt.subplots()
        ax1.hist(df["Age"], bins=20, color='skyblue', edgecolor='black')
        ax1.set_title("Distribution des âges")
        ax1.set_xlabel("Âge")
        ax1.set_ylabel("Nombre de patients")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x="HeartDisease", y="Cholesterol", ax=ax2)
        ax2.set_title("Cholestérol selon la maladie cardiaque")
        st.pyplot(fig2)

        fig4, ax4 = plt.subplots()
        sns.countplot(data=df, x="Sex", hue="HeartDisease", ax=ax4)
        ax4.set_title("Répartition du sexe selon HeartDisease")
        st.pyplot(fig4)

with prediction:
    if st.button("Lancer la Prédiction"):
        st.session_state.show_prediction_form = True

# Formulaire de prédiction
if st.session_state.show_prediction_form:
    st.subheader(" Remplissez les informations du patient :")

    Age = st.slider("Âge", 18, 100, 50)
    Sex = st.selectbox("Sexe", ["M", "F"])
    ChestPainType = st.selectbox("Type de douleur thoracique", ["ATA", "NAP", "ASY", "TA"])
    RestingBP = st.slider("Pression au repos", 80, 200, 120)
    Cholesterol = st.slider("Cholestérol", 100, 600, 200)
    FastingBS = st.selectbox("Glycémie à jeun > 120 mg/dl", [0, 1])
    RestingECG = st.selectbox("ECG au repos", ["Normal", "ST", "LVH"])
    MaxHR = st.slider("Fréquence cardiaque maximale", 60, 220, 150)
    ExerciseAngina = st.selectbox("Angine à l'effort", ["Y", "N"])
    Oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0, step=0.1)
    ST_Slope = st.selectbox("Pente ST", ["Up", "Flat", "Down"])

    # Encodage
    sex_bin = 1 if Sex == "M" else 0
    chest_dict = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
    restecg_dict = {"Normal": 0, "ST": 1, "LVH": 2}
    exang_bin = 1 if ExerciseAngina == "Y" else 0
    slope_dict = {"Up": 0, "Flat": 1, "Down": 2}

    input_data = np.array([[
        Age, sex_bin, chest_dict[ChestPainType], RestingBP, Cholesterol,
        FastingBS, restecg_dict[RestingECG], MaxHR,
        exang_bin, Oldpeak, slope_dict[ST_Slope]
    ]])

    if st.button(" Prédire"):
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error(" Ce patient risque d’avoir une maladie cardiaque.")
        else:
            st.success(" Ce patient ne présente probablement pas de risque.")
