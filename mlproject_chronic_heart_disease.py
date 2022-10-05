# -*- coding: utf-8 -*-


#importing the Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#loading .csv file data to a Pandas Dataframe
heart_data =pd.read_csv('/content/heart.csv')
heart_data.head()
#heart_data.tail()

heart_data['target'].value_counts()

X = heart_data.drop(columns='target',axis=1)
Y = heart_data['target']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

#Model Training - Logistic Regression
model=LogisticRegression()

#Training the model with the Training data
model.fit(X_train,Y_train)

#Accuracy  prediction on the Training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print("Accuracy of Model on the Training Data: ",training_data_accuracy)

#Accuracy prediction on the Test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print("Accuracy of Model on the Test Data: ",test_data_accuracy)

#Predictive System
input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)

inputData_as_numpyArray = np.asarray(input_data)

inputData_reshaped = inputData_as_numpyArray.reshape(1,-1)

predict = model.predict(inputData_reshaped)
#print(predict)

if(predict[0] == 0):
    print("the person suffers from heart disease")
else:
    print("the person does not suffer from heart disease")



#Predictive System
input_data = (554,1,0,124,266,0,0,109,1,2.2,1,1,3)

inputData_as_numpyArray = np.asarray(input_data)

inputData_reshaped = inputData_as_numpyArray.reshape(1,-1)

predict = model.predict(inputData_reshaped)
#print(predict)

if(predict[0] == 0):
    print("the person suffers or might have some heart disease")
else:
    print("the person does not suffer from heart disease")



