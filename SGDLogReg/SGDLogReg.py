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
