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


# Extract the features and target variable
attributes = df.columns[1:-1]
target = df.columns[-1]

# Initialize the hypothesis space
specific_h = ['0'] * len(attributes)
general_h = [['?' for _ in range(len(attributes))] for _ in range(len(attributes))]
print(specific_h)
print(general_h)
# Candidate-Elimination algorithm
for i, row in df.iterrows():
    if row[target] == 'Yes':
        # Update the specific hypothesis
        for j in range(len(attributes)):
            if specific_h[j] == '0':
                specific_h[j] = row[j + 1]
            elif specific_h[j] != row[j + 1]:
                specific_h[j] = '?'

        # Update the general hypothesis
        for j in range(len(general_h)):
            if general_h[j] != ['?' for _ in range(len(attributes))]:
                if row[j + 1] != specific_h[j]:
                    general_h[j][j] = '?'
    elif row[target] == 'No':
        for j in range(len(attributes)):
            if row[j + 1] != specific_h[j] and specific_h[j] != '?':
                general_h[j][j] = specific_h[j]
            else:
                general_h[j][j] = '?'

# Remove redundant general hypotheses
general_h = [hypothesis for hypothesis in general_h if hypothesis != ['?' for _ in range(len(attributes))]]

print("Specific Hypothesis:", specific_h)
print("General Hypotheses:", general_h)
