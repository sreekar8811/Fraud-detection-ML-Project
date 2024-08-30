# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:21:26 2024

@author: sreekar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
data = pd.read_csv('synthetic_online_payment_fraud11.csv')
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Display transaction type counts
print(data['type'].value_counts())

# Plot distribution of transaction types
type_counts = data['type'].value_counts()
transactions = type_counts.index
quantity = type_counts.values

plt.figure(figsize=(8, 8))
plt.pie(quantity, labels=transactions, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})
plt.title('Distribution of Transaction Type')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Calculate correlation matrix
correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending=False))

# Encode categorical features
label_encoder = LabelEncoder()
data["type"] = label_encoder.fit_transform(data["type"])
data['isFraud'] = data["isFraud"].map({0: "no Fraud", 1: "Fraud"})

print(data.head())

# Prepare the data for training
X = data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]]
y = data["isFraud"].map({"no Fraud": 0, "Fraud": 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


# Initialize models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier(n_neighbors=3),  # Reduced k to avoid error
    "Neural Network": MLPClassifier(max_iter=1000)
}

# Train and evaluate models
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    
    print(f"{model_name} Model Accuracy: {accuracy}")
    print(f"{model_name} Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"{model_name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

# Plotting the model accuracies
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.xticks(rotation=45)
plt.show()

# Making a prediction with the Decision Tree model for demonstration
features = np.array([[4, 9000.60, 9000.60, 0.0]])
for model_name, model in models.items():
    predicted_label = model.predict(features)
    predicted_label_text = "Fraud" if predicted_label == 1 else "no Fraud"
    print(f"Prediction with {model_name} for features {features}: {predicted_label_text}")
