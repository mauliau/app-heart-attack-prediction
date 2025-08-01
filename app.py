import streamlit as st
import pickle
import numpy as np

# Load model
with open('heart_attack_prediction.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Prediksi Serangan Jantung")

# Input dari user
age = st.number_input("Umur", min_value=0, max_value=120, value=30)
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
region = st.selectbox("Wilayah", ["Rural", "Urban"])
income_level = st.selectbox("Tingkat Pendapatan", ["Low", "Middle", "High"])
hypertension = st.selectbox("Hipertensi", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])
cholesterol_level = st.slider("Tingkat Kolesterol", 100, 400, 200)
obesity = st.selectbox("Obesitas", [0, 1])
waist_circumference = st.slider("Lingkar Pinggang (cm)", 20, 200, 90)
family_history = st.selectbox("Riwayat Keluarga", [0, 1])
smoking_status = st.selectbox("Status Merokok", ["Never", "Former", "Current"])
alcohol_consumption = st.selectbox("Konsumsi Alkohol", ["None", "Moderate", "High"])
physical_activity = st.selectbox("Aktivitas Fisik", ["Low", "Moderate", "High"])
dietary_habits = st.selectbox("Kebiasaan Diet", ["Unhealthy", "Healthy"])
air_pollution_exposure = st.selectbox("Paparan Polusi", ["Low", "Moderate", "High"])
stress_level = st.selectbox("Tingkat Stres", ["Low", "Moderate", "High"])
sleep_hours = st.slider("Jam Tidur per Hari", 0.0, 24.0, 7.0)
blood_pressure_systolic = st.slider("Tekanan Darah Sistolik", 60, 200, 120)
blood_pressure_diastolic = st.slider("Tekanan Darah Diastolik", 30, 130, 80)
fasting_blood_sugar = st.slider("Gula Darah Puasa", 70, 250, 100)
cholesterol_hdl = st.slider("HDL Kolesterol", 5, 100, 50)
cholesterol_ldl = st.slider("LDL Kolesterol", -20, 300, 130)
triglycerides = st.slider("Trigliserida", 50, 400, 150)
EKG_results = st.selectbox("Hasil EKG", ["Normal", "Abnormal"])
previous_heart_disease = st.selectbox("Riwayat Penyakit Jantung Sebelumnya", [0, 1])
medication_usage = st.selectbox("Penggunaan Obat", [0, 1])
participated_in_free_screening = st.selectbox("Ikut Skrining Gratis", [0, 1])

# Encoding
def encode_inputs():
    return np.array([[
        age,
        1 if gender == "Male" else 0,
        {"Rural":0, "Urban":1,}[region],
        {"Low":0, "Middle":1, "High":2}[income_level],
        hypertension,
        diabetes,
        cholesterol_level,
        obesity,
        waist_circumference,
        family_history,
        {"Never":0, "Former":1, "Current":2}[smoking_status],
        {"None":0, "Moderate":1, "High":2}[alcohol_consumption],
        {"Low":0, "Moderate":1, "High":2}[physical_activity],
        {"Unhealthy":0, "Healthy":1}[dietary_habits],
        {"Low":0, "Moderate":1, "High":2}[air_pollution_exposure],
        {"Low":0, "Moderate":1, "High":2}[stress_level],
        sleep_hours,
        blood_pressure_systolic,
        blood_pressure_diastolic,
        fasting_blood_sugar,
        cholesterol_hdl,
        cholesterol_ldl,
        triglycerides,
        {"Normal":0, "Abnormal":1}[EKG_results],
        previous_heart_disease,
        medication_usage,
        participated_in_free_screening
    ]])

# Prediksi
if st.button("Prediksi Serangan Jantung"):
    input_data = encode_inputs()
    prob = model.predict_proba(input_data)[0][1]
    pred = int(prob >= 0.3)  # threshold

    st.markdown("---")
    st.write("**Probabilitas Serangan Jantung:**", round(prob * 100, 2), "%")

    if pred == 1:
        st.error("⚠️ Individu ini **berisiko mengalami serangan jantung**")
    else:
        st.success("✅ Individu ini **tidak terindikasi serangan jantung**")