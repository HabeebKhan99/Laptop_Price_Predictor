import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR,"Dataset","laptop_data_cleaned.csv")
df = pd.read_csv(csv_path)

df = df.drop_duplicates()

x = df.drop("Price",axis = 1)
y = df["Price"]

x = pd.get_dummies(x, dtype = int)
print(x)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

x_train,x_test,y_train,y_test = train_test_split (x,y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lr = LinearRegression()
lr.fit(x_train,y_train)

prediction = lr.predict(x_test)

print("Accuracy:", r2_score(y_test,prediction))


import joblib
import os

model_path = os.path.join(BASE_DIR,"models/model.pkl")
scaler_path = os.path.join(BASE_DIR,"models/scaler.pkl")
feature_path = os.path.join(BASE_DIR,"models/feature.pkl")

joblib.dump(lr, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(x.columns, feature_path)
