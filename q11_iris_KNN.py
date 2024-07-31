import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data  # features
y = iris.target  # target

# Step 2: Create a DataFrame for plotting
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

# Map the target numbers to species names
species_names = iris.target_names
df['species'] = df['species'].apply(lambda x: species_names[x])

# Plot the data
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['sepal width (cm)'], df['sepal length (cm)'], c=df['species'].astype('category').cat.codes, cmap='viridis')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Sepal Length (cm)')
plt.title('Sepal Width vs Sepal Length')
plt.colorbar(scatter, label='Species')
plt.show()

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Fit the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 5: Predict with the test data
y_pred = knn.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=species_names))
