# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 10:45:17 2020

@author: Vrinda
"""

#data collection
import pandas as pd
dataset=pd.read_csv('USA_Housing.csv')


#data interpretation
dataset.info()
dataset.describe()

#data cleaning
#dropping the address column
dataset.drop('Address',inplace=True,axis=1)

from sklearn import preprocessing
pp=preprocessing.StandardScaler()

#separating the independent and target variables
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#noramalizing the independent variables to get in the range
x=pd.DataFrame(pp.fit_transform(x))

#splitting the dataset for training and testing
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2,random_state=92)

from sklearn.linear_model import LinearRegression as lnreg
ln=lnreg()

#training the model
ln.fit(x_train,y_train)

#testing the model
y_pred=ln.predict(x_test)

#calculating the accuracy
acc=ln.score(x_test,y_test)

#predicting the price of the house given all other parameters
from numpy import asarray as arr
y_pred2=ln.predict(arr([50000,6.0,5,3,10000]).reshape(1,-1))
print(y_pred2)
