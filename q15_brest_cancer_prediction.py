import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Create a sample dataset
data = {
    'mean_radius': [17.99, 20.57, 19.69, 11.42, 20.29,17.99, 20.57, 19.69, 11.42, 20.29,17.99, 20.57, 19.69, 11.42, 20.29,17.99, 20.57, 19.69, 11.42, 20.29],
    'mean_texture': [10.38, 17.77, 21.25, 20.38, 14.34,10.38, 17.77, 21.25, 20.38, 14.34,10.38, 17.77, 21.25, 20.38, 14.34,10.38, 17.77, 21.25, 20.38, 14.34],
    'mean_perimeter': [122.8, 132.9, 130.0, 77.58, 135.1,122.8, 132.9, 130.0, 77.58, 135.1,122.8, 132.9, 130.0, 77.58, 135.1,122.8, 132.9, 130.0, 77.58, 135.1],
    'mean_area': [1001, 1326, 1203, 386.1, 1297,1001, 1326, 1203, 386.1, 1297,1001, 1326, 1203, 386.1, 1297,1001, 1326, 1203, 386.1, 1297],
    'mean_smoothness': [0.1184, 0.08474, 0.1096, 0.1425, 0.1003,0.1184, 0.08474, 0.1096, 0.1425, 0.1003,0.1184, 0.08474, 0.1096, 0.1425, 0.1003,0.1184, 0.08474, 0.1096, 0.1425, 0.1003],
    'diagnosis': ['M', 'M', 'M', 'B', 'M','M', 'M', 'M', 'B', 'M','M', 'M', 'M', 'B', 'M','M', 'M', 'M', 'B', 'M']  # M for Malignant, B for Benign
}

# Step 2: Convert to pandas DataFrame
df = pd.DataFrame(data)

# Part a: Print the first five rows
print("First five rows of the dataset:")
print(df.head())

# Part b: Basic statistical computations
print("\nBasic statistical summary:")
print(df.describe())

# Part c: Columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# Part d: Detect and replace null values
print("\nChecking for null values:")
print(df.isnull().sum())

# Replace null values with the mode
for column in df.columns:
    if df[column].isnull().sum() > 0:
        df[column].fillna(df[column].mode()[0], inplace=True)

print("\nNull values after replacement (if any):")
print(df.isnull().sum())

# Part e: Split the data into training and testing sets
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Part f: Train a Naive Bayes classifier and evaluate its performance
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the performance
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nAccuracy Score:")
print(accuracy)
