import joblib
import streamlit as st
import pandas as pd
import numpy as np
import os

# BASE_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Paths

model_path = os.path.join(BASE_DIR, "models/model.pkl")
scaler_path = os.path.join(BASE_DIR, "models/scaler.pkl")
feature_path = os.path.join(BASE_DIR, "models/feature.pkl")

# Load Trained Model Files

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature = joblib.load(feature_path)


# Premium UI

st.markdown("""
<style>

/* App Background */
html, body, .stApp{
    background-color:#f5f7fa !important;
}

[data-testid="stAppViewcontainer"]
{
    background-color:#f5f7fa !important;
}

/* Main Title */
h1{
    text-align:center;
    color:#1f2937;
    font-weight:bold
}

/* Sub Heading */
h2{
    color:#374151;
    font-weight:bold;
}

/* Predict Button */
.stButton > button{
    background-color:#2563eb !important;
    color:white !important;
    border:none !important;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:18px;
    font-weight:bold;
    
}

/* Button Hover */

.stButton > button:hover{
    background-color:#1d4ed8 !important;
    color:white !important;
}

/* Input Boxes */
.stNumberInput input,
.stSelectbox
div[data-baseweb = "select"]
{
    border-radius:8px;
}

/* Prediction Box */
.stSuccess{
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html = True)

# Title

st.title("💻 LAPTOP PRICE PREDICTOR")
st.write("Predict Laptop Price Using Machine Learning")

# Collecting User Input

st.subheader("💻 Laptop Information")

col1, col2, = st.columns(2) # >= 2 Columns Layout

with col1:

    Company = st.selectbox("Company", ["Dell","Lenovo","HP","Asus","Acer","MSI","Toshiba","Apple","Samsung","Razer","Mediacom",
"Microsoft","Xiaomi","Vero","Chuwi","Google","Fujitsu","LG","Huawei"])

    TypeName = st.selectbox("TypeName", ["Notebook","Gaming","Ultrabook","2 in 1 Convertables","Workstation","Netbook"])

    RAM = st.number_input("Ram")

    Weight = st.number_input("Weight")

    TouchScreen = st.selectbox("TouchScreen", [0,1])

    IPS =  st.selectbox("Ips", [0,1])

with col2:

    PPI = st.number_input("Ppi")

    CPU_Brand = st.selectbox("Cpu_brand", ["Intel Core i7","Intel Core i5","Intel Core i3","AMD Processor","Other Intel Processor"])

    HDD = st.number_input("HDD")

    SDD = st.number_input("SDD")

    GPU_Brand = st.selectbox("Gpu_brand", ["Intel","Nvidia","AMD"])

    OS = st.selectbox("Os", ["Windows","Mac","Others"])

    # User Information Collected #



predict_btn = st.button("🚀 Predict Price", use_container_width = True)

if predict_btn:

    # Processing User Input

    with st.spinner("🤖 AI is Analyzing Laptop Specifications..."):

        # Collecting User Inputs
        input_data = [Company,TypeName,RAM,Weight,TouchScreen,IPS,PPI,CPU_Brand,HDD,SDD,GPU_Brand,OS]

        # convert Input To DataFrame

        input_df = pd.DataFrame([input_data],columns = ["Company","Typename","Ram","Weight","TouchScreen","Ips","Ppi","CPU_brand","HDD","SDD","GPU_brand","Os"])

        # Encoding Categorical Features

        input_df = pd.get_dummies(input_df) 

         # Align Features With Training Data

        input_df = input_df.reindex(columns = feature, fill_value = 0)

        # Scaling 

        input_df = scaler.transform(input_df)

        # Final Prediction

        prediction = model.predict(input_df)
        prediction = np.exp(prediction)

        st.subheader(("📊 Prediction Result"))
        st.success(f"💻 Predict Laptop Price: {prediction[0]:.2f}")

# Footer

st.markdown(""" <div style = "text-align:center;
color:gray;">
        
Built with ❤️ using <b>Python</b>
,<b>Streamlit</b>
,<b>Numpy</b>
,<b>Pandas</b> and 
<b> Scikit-learn</b>
        
<br><br>

Developed by <b> HABEEB KHAN </b>

</div>
""" , unsafe_allow_html = True)
      
