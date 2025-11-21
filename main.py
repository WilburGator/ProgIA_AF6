from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import pandas as pd
import os

from modules import normal as no



if os.path.exists('./data/data_scaled.csv'):
    data_scaled = pd.read_csv('./data/data_scaled.csv')
    print("Ya existe")
else:
    data_frame = pd.read_csv('./data/data.csv')
    print("No existe")
    data_frame = data_frame.iloc[:, 1:]
    data_scaled = no.normalizar(data_frame)
    data_scaled = pd.concat([data_scaled, no.OHE(data_frame,'occupation_status')], axis = 1)
    data_scaled = pd.concat([data_scaled, no.OHE(data_frame,'defaults_on_file')], axis = 1)
    data_scaled = pd.concat([data_scaled, no.OHE(data_frame,'product_type')], axis = 1)
    data_scaled = pd.concat([data_scaled, no.OHE(data_frame,'loan_intent')], axis = 1)
    data_scaled = pd.concat([data_scaled, no.le_encode(data_frame,'loan_status')], axis = 1)
    data_scaled.to_csv("./data/data_scaled.csv", index=False)

print(data_scaled.head())
X = data_scaled.drop("loan_status", axis=1)
y = data_scaled["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

