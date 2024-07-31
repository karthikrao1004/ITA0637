import pandas as pd

# Create the dataset
data = {
    'Example': [1, 2, 3, 4],
    'Citations': ['Some', 'Many', 'Many', 'Many'],
    'Size': ['Small', 'Big', 'Medium', 'Small'],
    'In Library': ['No', 'No', 'No', 'No'],
    'Price': ['Affordable', 'Expensive', 'Expensive', 'Affordable'],
    'Editions': ['Few', 'Many', 'Few', 'Many'],
    'Buy': ['No', 'Yes', 'Yes', 'Yes']
}

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Define the Find-S algorithm
def find_s_algorithm(df):
    # Initialize the most specific hypothesis
    hypothesis = ['0'] * (len(df.columns) - 2)  # Exclude 'Example' and 'Buy' columns

    # Iterate through each example in the dataset
    for i, row in df.iterrows():
        if row['Buy'] == 'Yes':  # Consider only positive examples
            for j in range(len(hypothesis)):
                if hypothesis[j] == '0':  # Replace initial '0' with the first positive example attribute
                    hypothesis[j] = row[j + 1]
                elif hypothesis[j] != row[j + 1]:  # Generalize hypothesis
                    hypothesis[j] = '?'
    return hypothesis

# Apply the Find-S algorithm
hypothesis = find_s_algorithm(df)

# Show the resulting hypothesis
print(f'The most specific hypothesis is: {hypothesis}')
