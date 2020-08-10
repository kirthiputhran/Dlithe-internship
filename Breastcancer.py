# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:03:02 2020

@author: Kirthi
"""
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt 
#Data collection
dataset=pd.read_csv('Breast_Cancer.csv')
dataset.info()

#drop unnecessary colums
dataset.drop(['id','Unnamed: 32'],inplace=True,axis=1)

#plot heatmap to see the corelation between features and remove the features with highest correlation with any other feature
plt.figure(figsize=(20,20))  
sb.heatmap(dataset.corr(), annot=True, fmt='.0%')
dataset.drop(['perimeter_mean','area_mean','perimeter_se','area_se','radius_worst','area_worst','perimeter_worst'],inplace=True,axis=1)

#replacing the target variable with 0 and 1
from sklearn import preprocessing as pp
labelencoder=pp.LabelEncoder()
dataset['diagnosis']=labelencoder.fit_transform(dataset['diagnosis'])

#separate dependent and independent variables
x=dataset.iloc[:,1:23].values
y=dataset.iloc[:,0].values

#normalize the values to give equal priority to all the columns
from sklearn import preprocessing
pp=preprocessing.StandardScaler()
x=pd.DataFrame(pp.fit_transform(x))

#split the dataset for training and testing
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2,random_state=7)

#apply logistic regression algorithm to train the model
from sklearn.linear_model import LogisticRegression as lr
logreg=lr()
logreg.fit(x_train,y_train)

#test the model
y_pred=logreg.predict(x_test)

#compute the accuracy of prediction
accuracy=logreg.score(x_test,y_test)
print('Logistic Regression accuracy:',accuracy)
