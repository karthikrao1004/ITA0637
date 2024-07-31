import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the Wine dataset
wine = load_wine()
X = wine.data  # features
y = wine.target  # target

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Fit the Naive Bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)

# Step 4: Predict with the test data
y_pred = nb.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))
