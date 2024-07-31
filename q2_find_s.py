import pandas as pd

# Create the dataset
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'Air Temp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'Enjoy Sport': ['Yes', 'Yes', 'No', 'Yes']
}

# Create DataFrame
df = pd.DataFrame(data)

# Define the Find-S algorithm
def find_s_algorithm(df):
    # Initialize the most specific hypothesis
    hypothesis = ['0'] * (len(df.columns) - 1)

    # Iterate through each example in the dataset
    for i, row in df.iterrows():
        if row['Enjoy Sport'] == 'Yes':  # Consider only positive examples
            for j in range(len(hypothesis)):
                if hypothesis[j] == '0':  # Replace initial '0' with the first positive example attribute
                    hypothesis[j] = row[j]
                elif hypothesis[j] != row[j]:  # Generalize hypothesis
                    hypothesis[j] = '?'
    return hypothesis

# Apply the Find-S algorithm
hypothesis = find_s_algorithm(df)

# Show the resulting hypothesis
print(f'The most specific hypothesis is: {hypothesis}')
