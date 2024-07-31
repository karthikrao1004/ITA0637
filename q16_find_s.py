import pandas as pd

# Create the dataset
data = {
    'Size': ['Big', 'Small', 'Small', 'Big', 'Small'],
    'Color': ['Red', 'Red', 'Red', 'Blue', 'Blue'],
    'Shape': ['Circle', 'Triangle', 'Circle', 'Circle', 'Circle'],
    'Class': ['No', 'No', 'Yes', 'No', 'Yes']
}

# Create DataFrame
df = pd.DataFrame(data)

# Define the Find-S algorithm
def find_s_algorithm(df):
    # Initialize the most specific hypothesis
    hypothesis = ['0'] * (len(df.columns) - 1)

    # Iterate through each example in the dataset
    for i, row in df.iterrows():
        if row['Class'] == 'Yes':  # Consider only positive examples
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
