import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create the dataset
data = {
    'Sales': [250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440],
    'Advertising': [20, 22, 23, 25, 26, 27, 28, 29, 30, 32, 34, 36, 38, 40, 42, 45, 48, 50, 52, 55]
}

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Step 2a: Print the first five rows
print("First five rows of the dataset:")
print(df.head())

# Step 2b: Basic statistical computations
print("\nBasic statistical computations:")
print(df.describe())

# Step 2c: The columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)

# Step 2d: Explore the data using scatterplot
plt.scatter(df['Advertising'], df['Sales'])
plt.xlabel('Advertising Expenditure')
plt.ylabel('Sales')
plt.title('Sales vs Advertising Expenditure')
plt.show()

# Step 2e: Detect null values in the dataset
print("\nDetecting null values:")
print(df.isnull().sum())

# Assuming there are null values, we replace them with the mode value
# Here, we just showcase the code to replace null values if they existed
# mode_sales = df['Sales'].mode()[0]
# mode_advertising = df['Advertising'].mode()[0]
# df['Sales'].fillna(mode_sales, inplace=True)
# df['Advertising'].fillna(mode_advertising, inplace=True)

# Step 2f: Split the data into training and test sets
X = df[['Advertising']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build and evaluate the model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print model parameters
print("\nModel parameters:")
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Evaluate the model
print("\nModel evaluation:")
print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")

# Predict the sales for the next quarter (assuming next quarter's advertising expenditure is 60)
next_quarter_advertising = np.array([[60]])
predicted_sales = model.predict(next_quarter_advertising)
print(f"\nPredicted sales for the next quarter with advertising expenditure of 60: {predicted_sales[0]}")
