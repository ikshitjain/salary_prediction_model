import streamlit as st
import pandas as pd
import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction")

st.markdown("Enter the following details to predict your salary:")

# User Inputs
age = st.slider("Age", 18, 65, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education", ["Bachelors", "Masters", "PhD"])
job_role = st.selectbox("Job Role", ["Data Analyst", "Software Engineer", "Manager", "HR Specialist", "Developer"])
experience = st.slider("Years of Experience", 0, 40, 2)
workclass = st.selectbox("Workclass", ["Private", "Government"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
relationship = st.selectbox("Relationship", ["Not-in-family", "Spouse", "Own-child", "Unmarried"])
race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander"])

# Prepare user input
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Education": education,
    "JobRole": job_role,
    "Experience": experience,
    "Workclass": workclass,
    "MaritalStatus": marital_status,
    "Relationship": relationship,
    "Race": race
}])

# Load and prepare training data
df = pd.read_csv("dataset.csv")

label_encoders = {}
le_cols = ['Gender', 'Education', 'JobRole', 'Workclass', 'MaritalStatus', 'Relationship', 'Race']

# Encode the dataset and store encoders
for col in le_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train the model
X = df.drop("Salary", axis=1)
y = df["Salary"]
model = LinearRegression()
model.fit(X, y)

# Encode input with stored encoders
for col in le_cols:
    le = label_encoders[col]
    input_df[col] = le.transform([input_df[col][0]])

# Predict
salary_pred = model.predict(input_df)[0]
st.success(f"💰 Predicted Salary: ₹{int(salary_pred):,}")

# Plot
st.markdown("### 📈 Salary Distribution (Sample Data)")
fig, ax = plt.subplots()
df['Salary'].hist(bins=10, ax=ax, color='skyblue')
ax.set_xlabel("Salary")
ax.set_ylabel("Count")
st.pyplot(fig)
