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
