import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to train a model and evaluate its performance
def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    execution_time = end_time - start_time
    return accuracy, execution_time

# Initialize the classifiers
decision_tree = DecisionTreeClassifier()
logistic_regression = LogisticRegression(max_iter=200)
knn = KNeighborsClassifier()

# Evaluate each classifier
dt_accuracy, dt_time = evaluate_model(decision_tree, X_train, X_test, y_train, y_test)
lr_accuracy, lr_time = evaluate_model(logistic_regression, X_train, X_test, y_train, y_test)
knn_accuracy, knn_time = evaluate_model(knn, X_train, X_test, y_train, y_test)

# Print the results
print(f"Decision Tree Classifier: Accuracy = {dt_accuracy:.2f}, Execution Time = {dt_time:.4f} seconds")
print(f"Logistic Regression: Accuracy = {lr_accuracy:.2f}, Execution Time = {lr_time:.4f} seconds")
print(f"KNN Classifier: Accuracy = {knn_accuracy:.2f}, Execution Time = {knn_time:.4f} seconds")
