from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA

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
print(x_train.shape[1])
"""Apply PCA"""
pca = PCA(n_components=10)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
x_train = x_train_pca
x_test = x_test_pca

# from sklearn.feature_selection import VarianceThreshold

# # 创建VarianceThreshold对象，设置阈值
# selector = VarianceThreshold(threshold=0.1)

# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression

# # 创建逻辑回归模型
# model = LogisticRegression()

# # 创建RFE对象，设置模型和要选择的特征数量
# selector = RFE(estimator=model, n_features_to_select=10)

# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import RandomForestClassifier

# # 创建随机森林分类器
# model = RandomForestClassifier()

# # 创建SelectFromModel对象，设置模型和阈值
# selector = SelectFromModel(estimator=model, threshold=0.1)


# # 对特征进行选择
# x_train = selector.fit_transform(x_train, y_train)
# x_test = selector.transform(x_test)


# 定义逻辑回归模型
logistic_regression = LogisticRegression()

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000, 2000, 3000]
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(logistic_regression, param_grid, cv=5)

# 在训练数据上执行网格搜索
grid_search.fit(x_train, y_train)

# 打印最佳参数组合
print("Best Parameters:", grid_search.best_params_)

# 使用最佳参数组合的模型进行预测
best_model = grid_search.best_estimator_
predictions = best_model.predict(x_test)

# 计算准确率和F1得分
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", accuracy)
print("F1 Score:", f1)