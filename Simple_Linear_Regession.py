
#Importing librarries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset Salary_Data.csv

dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

# Split data into test and training 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train)

print(X_test)

print(y_train)

print(y_test)

# Traing our Simple regression model using training sets

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the test set result

y_pred=regressor.predict(X_train)

# Visualing the trainging set
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Train data salary vs experience")
plt.xlabel('year of experience')
plt.ylabel('Salary')
plt.show

# Visualing the test set

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
##plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Test data Salary vs experience")
plt.xlabel('year of experience')
plt.ylabel('Salary')
plt.show