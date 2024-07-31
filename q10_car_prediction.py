import pandas as pd

# Sample car dataset
data = {
    'make': ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi'],
    'model': ['Corolla', 'Civic', 'Focus', '3 Series', 'A4'],
    'year': [2015, 2016, 2017, 2018, 2019],
    'engine_size': [1.8, 2.0, 2.5, 3.0, 2.0],
    'number_of_doors': [4, 4, 4, 4, 4],
    'sale_price': [15000, 16000, 17000, 30000, 35000]
}

df = pd.DataFrame(data)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode categorical features
le_make = LabelEncoder()
le_model = LabelEncoder()

df['make'] = le_make.fit_transform(df['make'])
df['model'] = le_model.fit_transform(df['model'])

# Define features and target variable
X = df[['make', 'model', 'year', 'engine_size', 'number_of_doors']]
y = df['sale_price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
