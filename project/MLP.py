import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split

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

# Define a list of model configurations to try
models = [
    ("MLP (3 layers)", MLPClassifier(hidden_layer_sizes=(100, 100, 100))),
    ("MLP (2 layers)", MLPClassifier(hidden_layer_sizes=(100, 100))),
    ("MLP (1 layer)", MLPClassifier(hidden_layer_sizes=(100,))),
]

# Perform cross-validation and select the best model configuration
best_model = None
best_accuracy = 0.0
best_model_name = ""
for name, model in models:
    cv_scores = cross_val_score(model, x_train, y_train, cv=5)
    # Calculate mean and standard deviation of cross-validation scores
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)

    # Print the mean and standard deviation
    print(f"{name} - cross-validation Mean score:", mean_score)
    print(f"{name} - cross-validation Standard deviation:", std_score)
    accuracy = cv_scores.mean()
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy
        best_model_name = name

print("Best Model Configuration:")
print(best_model_name)

#train the best model
best_model.fit(x_train, y_train)

# Predict on the test set
y_pred = best_model.predict(x_test)

# Calculate test accuracy
accuracy = accuracy_score(y_test, y_pred)
print("test Accuracy:", accuracy)

# Calculate test F1 score
f1 = f1_score(y_test, y_pred)
print("test F1 Score:", f1)

num_hidden_layers = len(best_model.hidden_layer_sizes)  # 隐藏层的数量
num_neurons_per_layer = best_model.hidden_layer_sizes[0]  # 每个隐藏层的神经元数量
input_dim = x_train.shape[1]  # 输入特征的维度

# 计算权重的数量
weight_count = (input_dim * num_neurons_per_layer)  # 输入层到第一个隐藏层之间的权重数量
for _ in range(num_hidden_layers - 1):
    weight_count += (num_neurons_per_layer * num_neurons_per_layer)  # 每个隐藏层到下一个隐藏层之间的权重数量
weight_count += (num_neurons_per_layer * 1)  # 最后一个隐藏层到输出层之间的权重数量

# 计算偏置的数量
bias_count = num_neurons_per_layer * num_hidden_layers + 1  # 每个隐藏层和输出层都有一个偏置

# 计算自由度
degrees_of_freedom = weight_count + bias_count

print("自由度:", degrees_of_freedom)
print(x_train.shape[0])

print(x_test.shape[0])



