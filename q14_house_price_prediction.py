import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data
data = {
    'size': [1400, 1600, 1700, 1800, 1900, 2000],
    'bedrooms': [3, 3, 4, 4, 4, 5],
    'bathrooms': [2, 3, 2, 3, 3, 4],
    'location': [1, 1, 2, 2, 3, 3],
    'price': [300000, 350000, 400000, 420000, 450000, 500000]
}

df = pd.DataFrame(data)
print("First five rows of the dataset:")
print(df.head())
print("\nBasic statistical computations:")
print(df.describe())
print("\nColumns and their data types:")
print(df.dtypes)
print("\nDetecting null values:")
print(df.isnull().sum())
print("\nNull values after replacement:")
print(df.isnull().sum())

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Feature Correlations')
plt.show()
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("\nModel Performance:")
print(f"Root Mean Squared Error: {rmse}")
new_house = pd.DataFrame({
    'size': [1500],
    'bedrooms': [3],
    'bathrooms': [2],
    'location': [1]
})

predicted_price = model.predict(new_house)
print("\nPredicted price for the new house:", predicted_price[0])
