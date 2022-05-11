import pandas as pd
import cv2 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import time
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score

df = pd.read_csv('../data/csvfile/assign3/fspacenonoverlap.csv', header = None)[1:].sample(frac=1).reset_index(drop=True)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
k = 10
kf = KFold(n_splits=k, random_state=None)
model = RandomForestClassifier(random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1)
acc_score = [] 
for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
    acc = accuracy_score(pred_values , y_test)
    acc_score.append(acc)   
avg_acc_score = sum(acc_score)/k
print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))

#########################################################################################
#print('next model')

df = pd.read_csv('../data/csvfile/assign3/fspacenonoverlap.csv', header = None)[1:].sample(frac=1).reset_index(drop=True)
X = df.iloc[:,:-1]
NN = 81
y = df.iloc[:,-1]
k = 10
kf = KFold(n_splits=k, random_state=None)

ANNmodel = Sequential()
ANNmodel.add(Dense(units=800, kernel_initializer='uniform', 
                       activation='relu', input_dim=NN))
ANNmodel.add(Dense(units=600, kernel_initializer='uniform', 
                       activation='relu'))
ANNmodel.add(Dense(units=400, kernel_initializer='uniform', 
                       activation='relu'))
ANNmodel.add(Dense(units=2, kernel_initializer='uniform', 
                       activation='sigmoid'))
ANNmodel.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])
    

acc_score = []

for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    #y = pd.get_dummies(y_train)
    ANNmodel.fit(X_train, y_train, batch_size=20, epochs=20)
    pred_values = model.predict(X_test)
     
    acc = accuracy_score(pred_values , y_test)
    acc_score.append(acc)
     
avg_acc_score = sum(acc_score)/k
 
print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))


#ref: https://www.askpython.com/python/examples/k-fold-cross-validation