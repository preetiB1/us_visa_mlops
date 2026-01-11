import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="US Visa Predictor", layout="centered")
st.title("🇺🇸 US Visa Approval Predictor")

st.markdown("Enter applicant details:")

continent = st.selectbox("Continent", ["Asia", "Europe", "Africa", "North America", "South America", "Oceania"])
education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "Doctorate"])
has_exp = st.selectbox("Has Job Experience?", ["Y", "N"])
job_training = st.selectbox("Requires Job Training?", ["Y", "N"])
employees = st.number_input("Number of Employees", min_value=1)
yr_estab = st.number_input("Year of Establishment", min_value=1900, max_value=2026)
region = st.selectbox("Region of Employment", ["Northeast", "South", "West", "Midwest"])
wage = st.number_input("Prevailing Wage", min_value=0.0)
unit = st.selectbox("Wage Unit", ["Hour", "Week", "Month", "Year"])
full_time = st.selectbox("Full Time Position?", ["Y", "N"])

if st.button("Predict Visa Status"):
    payload = {
        "continent": continent,
        "education_of_employee": education,
        "has_job_experience": has_exp,
        "requires_job_training": job_training,
        "no_of_employees": employees,
        "yr_of_estab": yr_estab,
        "region_of_employment": region,
        "prevailing_wage": wage,
        "unit_of_wage": unit,
        "full_time_position": full_time
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        st.success(result["result"])
    else:
        st.error("API Error")
