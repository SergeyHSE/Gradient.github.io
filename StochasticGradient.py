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

plt.figure(figsize(10, 8), dpi=100)
plt.title('House price distribution')
plt.xlabel('Price')
plt.ylabel('Samples')
plt.hist(y, bins=20)
plt.show()



