import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
# Configure page layout
st.set_page_config(page_title='Personal Fitness Tracker', layout='wide')

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Calorie Prediction", "Statistics & Insights", "Workout Suggestions"])

# Load Data
@st.cache_data
def load_data():
    calories = pd.read_csv(r"C:\Users\bharg\Documents\calories.csv")
    exercise = pd.read_csv(r"C:\Users\bharg\Documents\exercise.csv")
    df = exercise.merge(calories, on="User_ID")
    df.drop(columns="User_ID", inplace=True)
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    return df

df = load_data()

# Encode gender properly
df["Gender"] = df["Gender"].map({"male": 1, "female": 0})

# Ensure all categorical columns are properly encoded
df = pd.get_dummies(df, drop_first=True)

# 🏠 **Home Page**
if page == "Home":
    st.title("🏋️ Personal Fitness Tracker")
    st.write("Monitor and predict your fitness metrics dynamically!")
    st.image("https://source.unsplash.com/800x400/?fitness,gym", use_container_width=True)

    # Overview of Dataset
    st.subheader("Dataset Overview")
    st.write(df.head())

    # Correlation Heatmap
    st.subheader("Correlation Between Features")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# 🔥 **Calorie Prediction**
elif page == "Calorie Prediction":
    st.title("🔥 Calorie Burn Prediction")
    
    # User Input
    age = st.slider("Age", 10, 100, 30)
    bmi = st.slider("BMI", 15, 40, 22)
    duration = st.slider("Duration (min)", 0, 60, 20)
    heart_rate = st.slider("Heart Rate", 60, 160, 80)
    body_temp = st.slider("Body Temperature (°C)", 36, 42, 37)
    gender = st.radio("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0

    # Prepare input dataframe
    input_data = pd.DataFrame({
        "Age": [age], 
        "BMI": [bmi], 
        "Duration": [duration], 
        "Heart_Rate": [heart_rate], 
        "Body_Temp": [body_temp], 
        "Gender_male": [gender]
    })

    # Model Training
    X = df.drop(columns=["Calories"])
    y = df["Calories"]
    model = RandomForestRegressor(n_estimators=500, max_depth=6)
    model.fit(X, y)

    # Ensure input data matches training feature names
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_data)
    
    st.subheader("Predicted Calories Burned")
    st.write(f"**{round(prediction[0], 2)} kilocalories**")

# 📊 **Statistics & Insights**
elif page == "Statistics & Insights":
    st.title("📊 Statistics & Insights")
    
    # User Stats vs Dataset
    st.subheader("Distribution of BMI and Heart Rate")
    fig, ax = plt.subplots(ncols=2, figsize=(12,5))
    sns.histplot(df["BMI"], bins=20, kde=True, ax=ax[0])
    ax[0].set_title("BMI Distribution")
    sns.histplot(df["Heart_Rate"], bins=20, kde=True, ax=ax[1])
    ax[1].set_title("Heart Rate Distribution")
    st.pyplot(fig)
    
    st.subheader("Top Workout Durations")
    st.bar_chart(df.groupby("Duration")["Calories"].mean())

# 🏃 **Workout Suggestions**
elif page == "Workout Suggestions":
    st.title("🏃 Workout Suggestions")
    
    st.write("Suggested activities based on your fitness level:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Low Intensity")
        st.write("✔️ Yoga")
        st.write("✔️ Walking")
        st.write("✔️ Light Stretching")
    
    with col2:
        st.subheader("High Intensity")
        st.write("✔️ Running")
        st.write("✔️ HIIT")
        st.write("✔️ Strength Training")
