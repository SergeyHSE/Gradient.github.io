import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Find the median, quartile, interquartile regions

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

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


