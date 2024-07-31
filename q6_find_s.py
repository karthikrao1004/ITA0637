import pandas as pd

# Step 1: Create the dataset
data = {
    'Origin': ['Japan', 'Japan', 'Japan', 'USA', 'Japan'],
    'Manufacturer': ['Honda', 'Toyota', 'Toyota', 'Chrysler', 'Honda'],
    'Color': ['Blue', 'Green', 'Blue', 'Red', 'White'],
    'Decade': ['1980', '1970', '1990', '1980', '1980'],
    'Type': ['Economy', 'Sports', 'Economy', 'Economy', 'Economy'],
    'Example Type': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']
}

# Step 2: Convert it to a pandas DataFrame
df = pd.DataFrame(data)

# Step 3: Define the Find-S algorithm
def find_s_algorithm(df):
    # Initialize the most specific hypothesis
    hypothesis = ['0'] * (len(df.columns) - 1)

    # Iterate through each example in the dataset
    for i, row in df.iterrows():
        if row['Example Type'] == 'Positive':  # Consider only positive examples
            for j in range(len(hypothesis)):
                if hypothesis[j] == '0':  # Replace initial '0' with the first positive example attribute
                    hypothesis[j] = row[j]
                elif hypothesis[j] != row[j]:  # Generalize hypothesis
                    hypothesis[j] = '?'
    return hypothesis

# Step 4: Apply the Find-S algorithm
hypothesis = find_s_algorithm(df)

# Step 5: Show the resulting hypothesis
print(f'The most specific hypothesis is: {hypothesis}')
