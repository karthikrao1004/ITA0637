import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Create the dataset
data = {
    'Feature1': [2, 4, 4, 4, 6, 6, 6, 8, 8, 8,2, 4, 4, 4, 6, 6, 6, 8, 8, 8,2, 4, 4, 4, 6, 6, 6, 8, 8, 8,2, 4, 4, 4, 6, 6, 6, 8, 8, 8],
    'Feature2': [4, 2, 4, 6, 4, 6, 8, 4, 6, 8,4, 2, 4, 6, 4, 6, 8, 4, 6, 8,4, 2, 4, 6, 4, 6, 8, 4, 6, 8,4, 2, 4, 6, 4, 6, 8, 4, 6, 8],
    'Label':    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,0, 0, 0, 0, 0, 1, 1, 1, 1, 1,0, 0, 0, 0, 0, 1, 1, 1, 1, 1,0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

# Step 2: Convert it to a pandas DataFrame
df = pd.DataFrame(data)

# Step 3: Apply the KNN algorithm
# Split the data into training and testing sets
X = df[['Feature1', 'Feature2']]
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Step 4: Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
