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
import matplotlib.pyplot as plt
import seaborn as sns

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
X_df = pd.DataFrame(X, columns=dataset.feature_names)
y_df = pd.DataFrame(Y)

plt.figure(figsize=(18, 18), dpi=100)
corr_mat = X_df.corr()
sns.heatmap(corr_mat, annot=True, cmap='coolwarm')
plt.show()

plt.figure(figsize=(8, 6), dpi=100)
count_data = y_df['target'].value_counts().reset_index()
count_data.columns = ['target', 'count']
ax = sns.barplot(x='target', y='count', data=count_data)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Count of 0 and 1 in the Target')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}',
                (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                va='center', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.show()

X_df.shape
name_features = dataset.feature_names
type(name_features)
name_features = list(name_features)
num_rows = 15
num_col = 2
num_plots = len(name_features)

custom_palette = ["#0099CC", "#FF9900", "#99CC00", "#FF6666", "#6600CC"]
sns.set_palette(custom_palette)

fig, axes = plt.subplots(nrows=num_rows, ncols=num_col, figsize=(10, 40), dpi=100)
axes = axes.flatten()
for i in range(num_plots):
    if i < num_plots:
        sns.histplot(x=name_features[i], data=X_df, ax=axes[i], kde=True)
        axes[i].set_title(f'Histplot for {name_features[i]}')
    else:
        fig.delaxes(axes[i])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=num_rows, ncols=num_col, figsize=(10, 40), dpi=100)
axes = axes.flatten()
for i in range(num_plots):
    if i < num_plots:
        sns.boxplot(x=name_features[i], data=X_df, ax=axes[i], orient='h',
                    notch=True, medianprops={"color": "r", "linewidth": 2})
        axes[i].set_title(f'Boxplot fot {name_features[i]}')
        median_val = X_df[name_features[i]].median()
        median_x = median_val
        median_y = 0.45
        axes[i].text(median_x, median_y, f'Median: {median_val:.2f}',
        va='center', color='r', fontsize=10, ha='left')
        mean_val = X_df[name_features[i]].mean()
        mean_x = mean_val
        mean_y = -0.45
        axes[i].text(mean_x, mean_y, f'Mean: {mean_val:.2f}',
                     va='center', color='darkblue', fontsize=10, ha='left')
    else:
        fig.delaxes(axes[i])
plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler

scaler_alldata = StandardScaler()
scaler_alldata.fit(X_df)  
X_df = scaler_alldata.transform(X_df)
X_df = pd.DataFrame(data=X_df, columns=dataset.feature_names)

# Because we have a higher correlation between some variables we need remove one of them to
# check main parametrs

import statsmodels.api as sm

log_reg = sm.Logit(endog=y_df, exog=X_df.drop('mean radius', axis=1)).fit()
print(log_reg.summary())

"""
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                 target   No. Observations:                  569
Model:                          Logit   Df Residuals:                      540
Method:                           MLE   Df Model:                           28
Date:                Mon, 23 Oct 2023   Pseudo R-squ.:                  0.9564
Time:                        13:12:05   Log-Likelihood:                -16.364
converged:                       True   LL-Null:                       -375.72
Covariance Type:            nonrobust   LLR p-value:                2.384e-133
===========================================================================================
                              coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------------------
mean texture                0.0769      2.193      0.035      0.972      -4.222       4.376
mean perimeter             16.8270     29.898      0.563      0.574     -41.772      75.426
mean area                  -4.0364     34.429     -0.117      0.907     -71.516      63.443
mean smoothness            -5.0169      3.373     -1.487      0.137     -11.628       1.594
mean compactness           15.2518      8.255      1.848      0.065      -0.927      31.431
mean concavity            -11.5815      8.405     -1.378      0.168     -28.055       4.892
mean concave points        -4.5521      6.391     -0.712      0.476     -17.078       7.974
mean symmetry               1.3828      1.608      0.860      0.390      -1.769       4.534
mean fractal dimension     -1.9900      2.833     -0.702      0.482      -7.543       3.563
radius error               -2.5763     10.687     -0.241      0.809     -23.522      18.369
texture error               3.2859      2.217      1.482      0.138      -1.060       7.632
perimeter error             7.4840      8.501      0.880      0.379      -9.177      24.145
area error                -22.6468     20.846     -1.086      0.277     -63.504      18.211
smoothness error           -1.2344      1.630     -0.757      0.449      -4.429       1.960
compactness error          -8.0736      5.041     -1.601      0.109     -17.954       1.807
concavity error             9.2044      5.417      1.699      0.089      -1.412      19.821
concave points error      -11.0397      5.387     -2.049      0.040     -21.598      -0.481
symmetry error              2.0896      2.403      0.870      0.384      -2.620       6.799
fractal dimension error    16.5983      8.128      2.042      0.041       0.668      32.529
worst radius              -24.0862     24.636     -0.978      0.328     -72.372      24.199
worst texture              -6.8861      3.508     -1.963      0.050     -13.762      -0.010
worst perimeter           -17.6062     21.865     -0.805      0.421     -60.461      25.248
worst area                 21.4682     37.028      0.580      0.562     -51.105      94.041
worst smoothness            2.5105      3.073      0.817      0.414      -3.512       8.533
worst compactness           5.7097      6.665      0.857      0.392      -7.354      18.774
worst concavity            -6.7691      6.213     -1.090      0.276     -18.946       5.408
worst concave points        2.4271      4.478      0.542      0.588      -6.349      11.203
worst symmetry             -3.5569      2.880     -1.235      0.217      -9.202       2.088
worst fractal dimension   -10.9496      5.548     -1.974      0.048     -21.824      -0.075
===========================================================================================
"""
"""
Logistic regression model has successfully converged, and it appears to be a good fit for the data,
as indicated by the high pseudo R-squared value and the low p-value for the likelihood ratio test.
However examining the coefficient estimates and their associated standard errors
show not good results.
"""

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

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

from sklearn.metrics import roc_curve, roc_auc_score

Y_prob_d = np.delete(Y_prob, 0, 1)
Y_prob_d

fpr, tpr, threshold = roc_curve(Y_test, Y_prob_d)
roc_auc = roc_auc_score(Y_test, Y_prob_d)

plt.figure(figsize=(9, 7), dpi=100)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', color='red', label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

weight_sorted = sorted(zip(model.weights.ravel(), dataset.feature_names), reverse=True)
weights_scaler = [x[0] for x in weight_sorted]
features_scaler = [x[1] for x in weight_sorted]
df_scaler = pd.DataFrame({'features_scaler':features_scaler, 'weights_scaler':weights_scaler})

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

"""
For the selected value of the regularization coef, estimate the variance of
the average value of the quality metric on the test batches.
To do this, perform cross-validation sufficiently many times (at least 100)
and calculate the sample variance.
"""

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

variance_kf = np.var(f1_scores_kf)

print(f"Variance for KFold with shuffle: {variance_kf}")

alpha = 0
num_repeats = 100

ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42) 

f1_scores_ss = [] 

for _ in range(num_repeats):
    f1_scores_fold_ss = []
    
    # ShuffleSplit
    for train_index, test_index in ss.split(X): 
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model = LogisticRegressionCustom(alpha=alpha, lr=1e-4, max_iter=10000, fit_intercept=True)
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)
        f1 = f1_score(Y_test, Y_pred, average='macro') 
        f1_scores_fold_ss.append(f1)

    f1_scores_ss.append(np.mean(f1_scores_fold_ss))
    
variance_ss = np.var(f1_scores_ss)
print(f"Variance for ShuffleSplit: {variance_ss}")

model = LogisticRegressionCustom(alpha=0, lr=1e-4, max_iter=10000, fit_intercept=True)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

Y_prob = model.predict_proba(X_test)

report = classification_report(Y_test, Y_pred)
print(report)

"""
              precision    recall  f1-score   support

           0       0.95      0.98      0.97        43
           1       0.99      0.97      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
"""

threshold = 0.5 
predicted_labels = (Y_prob[:, 1] > threshold).astype(int)
predicted_labels

Y_prob_d = np.delete(Y_prob, 0, 1)
Y_prob_d
fpr, tpr, threshold = roc_curve(Y_test, Y_prob_d)
roc_auc = roc_auc_score(Y_test, Y_prob_d)

plt.figure(figsize=(8, 6), dpi=100)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', color='red', label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
