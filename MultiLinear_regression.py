

##  Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

##  Importing the dataset

dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

## printing the dependent variables and independent variables

print(X)

print(y)

## using one hot encoder to convert the categorical data into vectors

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder="passthrough")

# print(X) for checking the categorical data before transform

X=np.array(ct.fit_transform(X))

print(X)

##
