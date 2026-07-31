# 💻 Laptop Price Predictor

A Machine Learning web application built with Python and Streamlit that predicts the estimated price of a laptop based on its specifications such as company, processor, RAM, storage, display, GPU, operating system, and other hardware features.

## 🚀 Features

- Laptop Price Prediction using Linear Regression
- Interactive Streamlit Web Application
- Real-time Price Prediction
- Automatic Feature Encoding
- Feature Scaling using StandardScaler
- Clean and Professional User Interface

## 📂 Project Structure

```
Laptop_Price_Predictor/

├── Dataset/
│   └── laptop_data_cleaned.csv
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature.pkl
│
├── App.py
├── Training.py
├── requirements.txt
└── README.md
```

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## 📊 Machine Learning Workflow

1. Load Laptop Dataset
2. Remove Duplicate Records
3. Data Preprocessing
4. Feature Encoding
5. Feature Scaling
6. Train Linear Regression Model
7. Save Trained Model Files
8. Predict Laptop Price using Streamlit

## ▶️ Run The Project

Install all required libraries

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run App.py
```

If the above command doesn't work, use:

```bash
python -m streamlit run App.py
```

## 👨‍💻 Developed By

**Habeeb Khan**