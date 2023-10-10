import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

train_df = pd.read_csv("project\Mushroom_datasets/mushroom_train.csv")
test_df = pd.read_csv("project\Mushroom_datasets/mushroom_test.csv")
# Get the label column
train_labels = train_df.iloc[:, -1]
test_labels = test_df.iloc[:, -1]
# Replace 'p' with 1 and 'e' with 2
re_labels = train_labels.replace({'p': 0, 'e': 1})
# Update the label column in the dataset
train_df.iloc[:, -1] = re_labels
# Replace 'p' with 1 and 'e' with 2
re_labels = test_labels.replace({'p': 0, 'e': 1})
# Update the label column in the dataset
test_df.iloc[:, -1] = re_labels

train_one_hot_encoding = pd.get_dummies(train_df.iloc[:, :-1])
train_one_hot_encoding["class"] = train_df.iloc[:, -1]

test_one_hot_encoding = pd.get_dummies(test_df.iloc[:, :-1])
test_one_hot_encoding["class"] = test_df.iloc[:, -1]

x_train = train_one_hot_encoding.drop(["class"], axis=1).values 
y_train = train_one_hot_encoding["class"].values
x_test = test_one_hot_encoding.drop(["class"], axis=1).values 
y_test = test_one_hot_encoding["class"].values

"""Standardize"""
scaler = StandardScaler()
scaler.fit(x_train[:, 0:3])
scaler.fit(x_test[:, 0:3])
x_train[:, 0:3] = scaler.transform(x_train[:, 0:3])
x_test[:, 0:3] = scaler.transform(x_test[:, 0:3])

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Create SVM classifier
svm_model = SVC()

# Perform cross-validation
cv_scores = cross_val_score(svm_model, x_train, y_train, cv=5)

# Calculate mean and standard deviation of cross-validation scores
mean_score = np.mean(cv_scores)
std_score = np.std(cv_scores)

# Print the mean and standard deviation
print("cross-validation Mean score:", mean_score)
print("cross-validation Standard deviation:", std_score)

# Fit the SVM model on the entire training data
svm_model.fit(x_train, y_train)

# Make predictions on the test data
predictions = svm_model.predict(x_test)

# Calculate accuracy and F1 score on the test data
accuracy_test = accuracy_score(y_test, predictions)
f1_test = f1_score(y_test, predictions, average='macro')

# Print the results
print("Test Accuracy:", accuracy_test)
print("Test F1 Score:", f1_test)

