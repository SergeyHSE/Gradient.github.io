"""
Each record in the database describes a Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. The attributes are deﬁned as follows (taken from the UCI Machine Learning Repository1): CRIM: per capita crime rate by town

ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS: proportion of non-retail business acres per town
CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX: nitric oxides concentration (parts per 10 million)
1https://archive.ics.uci.edu/ml/datasets/Housing
123
20.2. Load the Dataset 124
RM: average number of rooms per dwelling
AGE: proportion of owner-occupied units built prior to 1940
DIS: weighted distances to ﬁve Boston employment centers
RAD: index of accessibility to radial highways
TAX: full-value property-tax rate per $10,000
PTRATIO: pupil-teacher ratio by town 12. B: 1000(Bk−0.63)2 where Bk is the proportion of blacks by town 13. LSTAT: % lower status of the population
MEDV: Median value of owner-occupied homes in $1000s
We can see that the input attributes have a mixture of units.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.base import BaseEstimator
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)

path = r"YourPath"
path = path.replace('\\', '/')
data = pd.read_csv(path)
data.head()
X = data.drop(['medv'], axis=1)
y = data['madv']

#House prise distribution

plt.figure(figsize(10, 8), dpi=100)
plt.title('House price distribution')
plt.xlabel('Price')
plt.ylabel('Samples')
plt.hist(y, bins=20)
plt.show()

# House price outliers
plt.figure(figsize=(10, 8), dpi=100)
plt.title('House price outliers')
plt.boxplot(y)
plt.show()

# Find the median, quartile, interquartile regions (IQR)

q1 = np.quantile(y, 0.25)
q3 = np.quantile(y, 0.75)
med = np.median(y)
iqr = q3-q1
upper_bound = q3+(1.5*iqr) 
lower_bound = q1-(1.5*iqr)
print(iqr, upper_bound, lower_bound)

outliers = y[(y <= lower_bound) | (y >= upper_bound)]
print('Outliers:{}'.format(outliers))

y_without_outliers = y[(y <= lower_bound) | (y >= upper_bound)]

plt.figure(figsize=(10, 8), dpi=100)
plt.boxplot(y_without_outliers)
plt.title('House price without outliers')
plt.show()

y.shape
y_without_outliers.shape

# Apply z-score method

from scipy import stats

z = np.abs(stats.zscore(y))
data_z_score = y[(z < 3)]
data_z_score.shape

outlier_result_table = pd.DataFrame([[y.shape[0], y_without_outliers.shape[0], data_z_score.shape[0]]],
                                    columns=['Initial data', 'IQR', 'Z-score'])
print(outlier_result_table)

# Build hists for features

feature_names = list(X.columns)
num_rows = 7
num_cols = 2
num_plots = len(feature_names)

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 20), dpi=150)
axes = axes.flatten()

for i in range(num_plots):
    if i < num_plots:
        ax = axes[i]
        feature_data = X[feature_names[i]]
        ax.hist(x=feature_data, bins=30, color='tab:green')
        ax.set_title(f'Hist plot for {feature_names[i]}')
    else:
        fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

corr = X.corr()
corr.to_csv('Correlation.csv')

# Find the file in our directory

import os

current_directory = os.getcwd()
filename = 'Correlation.csv'
file_path = os.path.join(current_directory, filename)
if os.path.exists(file_path):
    print(f"The file '{filename}' is located at: {file_path}")
else:
    print(f"The file '{filename}' was not found in the current directory.")

# Create correlation heatmap

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
figure, ax = plt.subplots(figsize=(11, 9), dpi=150)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .7})
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

# Buld OLS model to look at significance of features
import statsmodels.api as sm

X_reg = sm.add_constant(X)
regression = sm.OLS(y, X_reg).fit()
print(regression.summary())

"""
OLS Regression Results                            
==============================================================================
Dep. Variable:                   medv   R-squared:                       0.741
Model:                            OLS   Adj. R-squared:                  0.734
Method:                 Least Squares   F-statistic:                     108.1
Date:                Sun, 17 Sep 2023   Prob (F-statistic):          6.72e-135
Time:                        21:45:24   Log-Likelihood:                -1498.8
No. Observations:                 506   AIC:                             3026.
Df Residuals:                     492   BIC:                             3085.
Df Model:                          13                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         36.4595      5.103      7.144      0.000      26.432      46.487
crim          -0.1080      0.033     -3.287      0.001      -0.173      -0.043
zn             0.0464      0.014      3.382      0.001       0.019       0.073
indus          0.0206      0.061      0.334      0.738      -0.100       0.141
chas           2.6867      0.862      3.118      0.002       0.994       4.380
nox          -17.7666      3.820     -4.651      0.000     -25.272     -10.262
rm             3.8099      0.418      9.116      0.000       2.989       4.631
age            0.0007      0.013      0.052      0.958      -0.025       0.027
dis           -1.4756      0.199     -7.398      0.000      -1.867      -1.084
rad            0.3060      0.066      4.613      0.000       0.176       0.436
tax           -0.0123      0.004     -3.280      0.001      -0.020      -0.005
ptratio       -0.9527      0.131     -7.283      0.000      -1.210      -0.696
b              0.0093      0.003      3.467      0.001       0.004       0.015
lstat         -0.5248      0.051    -10.347      0.000      -0.624      -0.425
==============================================================================
Omnibus:                      178.041   Durbin-Watson:                   1.078
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              783.126
Skew:                           1.521   Prob(JB):                    8.84e-171
Kurtosis:                       8.281   Cond. No.                     1.51e+04
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.51e+04. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

"""
Let's write MAPE function and calculate it for y_test_mean
"""

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate y_meen as an array of mean values
type(y_test)
y_meen = np.full_like(y_test, np.mean(y_test))
len(y_test)

# Calculate MAPE between y_test and y_meen
mape = MAPE(y_test, y_meen)
print("MAPE between y_test and y_mean:", mape)





