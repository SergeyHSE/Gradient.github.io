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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


