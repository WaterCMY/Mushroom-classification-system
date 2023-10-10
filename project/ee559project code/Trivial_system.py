import pandas as pd

# Load the dataset
train_df = pd.read_csv('project\Mushroom_datasets\mushroom_train.csv')

# Get the label column
train_labels = train_df.iloc[:, -1]

# Replace 'p' with 1 and 'e' with 2
re_labels = train_labels.replace({'p': 0, 'e': 1})

# Update the label column in the dataset
train_df.iloc[:, -1] = re_labels

# Print the updated dataset
# print(train_df)

# Get the unique values and their frequency count of the labels
class_counts = train_df['class'].value_counts()

N0 = class_counts[0]
N1 = class_counts[1]

import random

def random_assignment(N0, N1, N):
    # Calculate the probabilities
    p0 = N0 / N
    p1 = N1 / N

    # Generate a random number
    random_num = random.uniform(0, 1)

    # Assign the class based on the random number
    if random_num <= p0:
        return 0
    else:
        return 1


N = N0 + N1   # Total number of training data points

# Generate the Trivial_system predictions
Trivial_system_predictions = []
for _ in range(N):
    assignment = random_assignment(N0, N1, N)
    Trivial_system_predictions.append([assignment])

# Convert the dataset to a DataFrame
Trivial_system_predictions_data = pd.DataFrame(Trivial_system_predictions, columns=['label'])

predictions = Trivial_system_predictions_data['label']  

# Get the actual labels from the train_labels
labels = train_df['class']

# Calculate the number of correctly classified examples
correct_predictions = (predictions == labels).sum()
print(correct_predictions)

# Calculate accuracy
accuracy = correct_predictions / len(Trivial_system_predictions_data)

# Print the accuracy
print("Trivial_system Accuracy:", accuracy)

from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision recall and f1 score
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)

# Print the scores
print("Trivial_system F1 Score:", f1)



