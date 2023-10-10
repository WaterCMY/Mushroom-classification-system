import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.neighbors import NearestCentroid

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

# Initialize the classifier
nmc = NearestCentroid()

# Fit the classifier to the training data
nmc.fit(x_train, y_train)

# Predict the labels for the test data
y_pred = nmc.predict(x_test)

# Calculate the F1-score
f1 = f1_score(y_test, y_pred)

print("Baseline_system F1-score:", f1)
# Calculate the accuracy
accuracy = nmc.score(x_test, y_test)

print("Baseline_system Accuracy:", accuracy)
