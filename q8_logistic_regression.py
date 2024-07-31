import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Create the dataset
data = {
    'Age': [25, 45, 35, 50, 23, 40, 60, 48, 33, 36],
    'Income': [50000, 64000, 58000, 80000, 22000, 52000, 85000, 75000, 42000, 48000],
    'Loan Amount': [2000, 3000, 2400, 3500, 1200, 2500, 4000, 3000, 2200, 2600],
    'Credit Score': [720, 690, 710, 680, 730, 700, 650, 640, 710, 680],
    'Class': ['Good', 'Good', 'Good', 'Bad', 'Good', 'Good', 'Bad', 'Bad', 'Good', 'Bad']
}

# Step 2: Convert it to a pandas DataFrame
df = pd.DataFrame(data)

# Convert categorical labels to numerical labels
df['Class'] = df['Class'].map({'Good': 1, 'Bad': 0})

# Step 3: Apply the Logistic Regression classifier
# Split the data into training and testing sets
X = df[['Age', 'Income', 'Loan Amount', 'Credit Score']]
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Step 4: Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
