import numpy as np

def candidate_elimination(examples):
    num_attributes = len(examples[0]) - 1  # Number of attributes in the examples
    specific_hypothesis = ['0'] * num_attributes  # Most specific hypothesis
    general_hypothesis = [['?'] * num_attributes]  # Most general hypothesis

    for example in examples:
        if example[-1] == 'Yes':  # Positive example
            for i in range(num_attributes):
                if specific_hypothesis[i] == '0':
                    specific_hypothesis[i] = example[i]
                elif specific_hypothesis[i] != example[i]:
                    specific_hypothesis[i] = '?'
            
            for hypothesis in general_hypothesis:
                for i in range(num_attributes):
                    if hypothesis[i] != '?' and hypothesis[i] != specific_hypothesis[i]:
                        general_hypothesis.remove(hypothesis)
                        break
        else:  # Negative example
            temp_hypothesis = general_hypothesis.copy()
            for hypothesis in temp_hypothesis:
                for i in range(num_attributes):
                    if hypothesis[i] == '?':
                        if example[i] != specific_hypothesis[i]:
                            new_hypothesis = hypothesis.copy()
                            new_hypothesis[i] = specific_hypothesis[i]
                            if new_hypothesis not in general_hypothesis:
                                general_hypothesis.append(new_hypothesis)
                general_hypothesis.remove(hypothesis)

    return specific_hypothesis, general_hypothesis

# Example dataset
dataset = [
    ['Some', 'Small', 'No', 'Affordable', 'Few', 'No'],
    ['Many', 'Big', 'No', 'Expensive', 'Many', 'Yes'],
    ['Many', 'Medium', 'No', 'Expensive', 'Few', 'Yes'],
    ['Many', 'Small', 'No', 'Affordable', 'Many', 'Yes']
]

# Applying the Candidate-Elimination algorithm
specific, general = candidate_elimination(dataset)
print("Specific hypothesis:", specific)
print("General hypotheses:", general)
