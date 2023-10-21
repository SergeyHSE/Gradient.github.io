# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 20:18:07 2023

@author: SergeyHSE
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin

dataset = load_breast_cancer()
dataset['DESCR'].split('\n')[11:68]
"""
 '    :Attribute Information:',
 '        - radius (mean of distances from center to points on the perimeter)',
 '        - texture (standard deviation of gray-scale values)',
 '        - perimeter',
 '        - area',
 '        - smoothness (local variation in radius lengths)',
 '        - compactness (perimeter^2 / area - 1.0)',
 '        - concavity (severity of concave portions of the contour)',
 '        - concave points (number of concave portions of the contour)',
 '        - symmetry',
 '        - fractal dimension ("coastline approximation" - 1)',
 '',
 '        The mean, standard error, and "worst" or largest (mean of the three',
 '        worst/largest values) of these features were computed for each image,',
 '        resulting in 30 features.  For instance, field 0 is Mean Radius, field',
 '        10 is Radius SE, field 20 is Worst Radius.',
 '',
 '        - class:',
 '                - WDBC-Malignant',
 '                - WDBC-Benign',
 '',
 '    :Summary Statistics:',
 '',
 '    ===================================== ====== ======',
 '                                           Min    Max',
 '    ===================================== ====== ======',
 '    radius (mean):                        6.981  28.11',
 '    texture (mean):                       9.71   39.28',
 '    perimeter (mean):                     43.79  188.5',
 '    area (mean):                          143.5  2501.0',
 '    smoothness (mean):                    0.053  0.163',
 '    compactness (mean):                   0.019  0.345',
 '    concavity (mean):                     0.0    0.427',
 '    concave points (mean):                0.0    0.201',
 '    symmetry (mean):                      0.106  0.304',
 '    fractal dimension (mean):             0.05   0.097',
 '    radius (standard error):              0.112  2.873',
 '    texture (standard error):             0.36   4.885',
 '    perimeter (standard error):           0.757  21.98',
 '    area (standard error):                6.802  542.2',
 '    smoothness (standard error):          0.002  0.031',
 '    compactness (standard error):         0.002  0.135',
 '    concavity (standard error):           0.0    0.396',
 '    concave points (standard error):      0.0    0.053',
 '    symmetry (standard error):            0.008  0.079',
 '    fractal dimension (standard error):   0.001  0.03',
 '    radius (worst):                       7.93   36.04',
 '    texture (worst):                      12.02  49.54',
 '    perimeter (worst):                    50.41  251.2',
 '    area (worst):                         185.2  4254.0',
 '    smoothness (worst):                   0.071  0.223',
 '    compactness (worst):                  0.027  1.058',
 '    concavity (worst):                    0.0    1.252',
 '    concave points (worst):               0.0    0.291',
 '    symmetry (worst):                     0.156  0.664',
 '    fractal dimension (worst):            0.055  0.208',
 '    ===================================== ====== ======',
"""

X, Y = dataset['data'], dataset['target']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

####################################################
#                 SGD LOG REG                      #
####################################################

class LogisticRegressionCustom(ClassifierMixin):
    def __init__(self, alpha=0, lr=0.5, max_iter=1e5, fit_intercept=True, batch_size = 32):
        self.alpha = alpha
        self.lr = lr
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.batch_size = batch_size
        self.classes_ = np.array([0, 1])

    @staticmethod
    def _sigmoid(x):
        # use scipy.special.expit for calculating sigmoid function
        return expit(x)
     
    def _add_intercept(self, X):
        '''
        Add intersept coef 
        :param X: initial matrix
        '''

        X_copy = np.full((X.shape[0], X.shape[1] + 1), fill_value=1.)
        X_copy[:, :-1] = X

        return X_copy

    def fit(self, X, Y):

        assert X.shape[0] == Y.shape[0]

        if self.fit_intercept:  # добавляем свободный коэфициент
            X_copy = self._add_intercept(X)
        else:
            X_copy = X.copy()
            
        self.classes_ = np.array([0, 1])
        
        # Initialize weights and intercept
        self.weights = np.zeros(X_copy.shape[1])
        #self.weights = np.random.uniform(-1, 1, size=X_copy.shape[1])
        self.intercept_ = 0.0

        for iteration in range(int(self.max_iter)):
            # Choise mini_batch 
            indices = np.random.choice(X_copy.shape[0], self.batch_size, replace=False)
            X_batch = X_copy[indices]
            Y_batch = Y[indices]

            predictions = self._sigmoid(X_batch.dot(self.weights) + self.intercept_)
            grad = -X_batch.T.dot(Y_batch - predictions) / X_batch.shape[0]
            grad += 2 * self.alpha * self.weights
            self.weights -= self.lr * grad
            self.intercept_ -= self.lr * np.mean(Y_batch - predictions)

        return self

    def predict(self, X):

        if self.fit_intercept:
            X_copy = self._add_intercept(X)
        else:
            X_copy = X.copy()

        # check that number of features in X_copy is iqual number of coef
        assert X_copy.shape[1] == self.weights.shape[0]

        # Calculate predictions by sigmoid function
        predictions = self._sigmoid(X_copy.dot(self.weights) + self.intercept_)

        # Convert predictuons to binary values (0 or 1)
        binary_predictions = np.round(predictions).astype(int)

        return binary_predictions

    def predict_proba(self, X):

        if self.fit_intercept:
            X_copy = self._add_intercept(X)
        else:
            X_copy = X.copy()
         
        assert X_copy.shape[1] == self.weights.shape[0]

        predictions = self._sigmoid(X_copy.dot(self.weights) + self.intercept_)

        prob_predictions = np.column_stack((1 - predictions, predictions))

        return prob_predictions
###########################################################################

model = LogisticRegressionCustom(alpha=1, fit_intercept=True)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# check metrics
from sklearn.metrics import accuracy_score, classification_report

Y_prob = model.predict_proba(X_test)

report = classification_report(Y_test, Y_pred)
print(report)
"""
              precision    recall  f1-score   support

           0       0.38      1.00      0.55        43
           1       0.00      0.00      0.00        71

    accuracy                           0.38       114
   macro avg       0.19      0.50      0.27       114
weighted avg       0.14      0.38      0.21       114
"""
Y_prob

weight_sorted = sorted(zip(model.weights.ravel(), dataset.feature_names), reverse=True)
weights_scaler = [x[0] for x in weight_sorted]
features_scaler = [x[1] for x in weight_sorted]
df_scaler = pd.DataFrame({'features_scaler':features_scaler, 'weights_scaler':weights_scaler})

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6)) 
df_scaler.plot.barh(x='features_scaler', y='weights_scaler', color='skyblue', legend=False, ax=ax)
plt.title('Feature Weights (Scaled)')
plt.xlabel('Weight')
plt.ylabel('Features')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
for i, v in enumerate(df_scaler['weights_scaler']):
    ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('output.png', dpi=300)
plt.show()

# Find more appropriate parametr for learning rate

lrs = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 5, 10] # Learning rates
f1_list = []

for lr in lrs:
    model = LogisticRegressionCustom(alpha=1.0, lr=lr, max_iter=10000, fit_intercept=True)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    f1 = f1_score(Y_test, Y_pred, average='macro')
    f1_list.append(f1)

plt.figure(figsize=(12, 7), dpi=100)
plt.semilogx(lrs, f1_list, marker='o', linestyle='-')
plt.title('The dependence of the metric on the learning rate')
plt.xlabel('Learning rate')
plt.ylabel('F1-score')
plt.grid()
plt.show()

# Find more appropriate parametr for regularization coef
alpha_values = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0] 
results = []

for alpha in alpha_values:
    model = LogisticRegressionCustom(alpha=alpha, lr=1e-4, max_iter=10000, fit_intercept=True)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    f1 = f1_score(Y_test, Y_pred, average='macro')
    results.append((alpha, f1))

for alpha, f1 in results:
    if alpha == 0.0:
        print(f"Model without regularization: F1-мера = {f1}")
    else:
        print(f"Model with alpha={alpha}: F1-мера = {f1}")

# Choise the best regularization coef by KFold, ShuffleSplit

from sklearn.model_selection import KFold, ShuffleSplit

# Create objects for KFold and ShuffleSplit
kf = KFold(n_splits=5, shuffle=True, random_state=42)
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

alpha_values = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]

kf_scores = []
ss_scores = []

for alpha in alpha_values:
    model = LogisticRegressionCustom(alpha=alpha, lr=1e-4, max_iter=10000, fit_intercept=True, batch_size=32)

    kf_fold_scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_val)
        kf_fold_scores.append(f1_score(Y_val, Y_pred, average='macro'))

    kf_score = np.mean(kf_fold_scores)
    kf_scores.append(kf_score)
 
    ss_fold_scores = []
    for train_idx, val_idx in ss.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_val)
        ss_fold_scores.append(f1_score(Y_val, Y_pred, average='macro'))

    ss_score = np.mean(ss_fold_scores)
    ss_scores.append(ss_score)

best_alpha_kf = alpha_values[np.argmax(kf_scores)]
best_alpha_ss = alpha_values[np.argmax(ss_scores)]

print("The best alpha for KFold:", best_alpha_kf)
print("The best alpha for ShuffleSplit:", best_alpha_ss)

%%time
alpha = 0
num_repeats = 100
kf = KFold(n_splits=5, shuffle=True, random_state=42)

f1_scores_kf = []

for _ in range(num_repeats):
    f1_scores_fold_kf = []
    
    # KFold
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model = LogisticRegressionCustom(alpha=alpha, lr=1e-4, max_iter=10000, fit_intercept=True)
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)
        f1 = f1_score(Y_test, Y_pred, average='macro') 
        f1_scores_fold_kf.append(f1)

    f1_scores_kf.append(np.mean(f1_scores_fold_kf)) 
