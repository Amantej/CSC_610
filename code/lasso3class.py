import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score


x_train = pd.read_csv('../data/csvfile/assign2/overlap_Merged_image0image1image2_train.csv',header=None)[1:]
x_test = pd.read_csv('../data/csvfile/assign2/overlap_Merged_image0image1image2_test.csv',header=None)[1:]
x_train[81][x_train[81]==2]=1
x_test[81][x_test[81]==2]=1
y = x_train[81]
Y = np.array(y)
x_train.drop(81,axis=1,inplace=True)
X = x_train
X1 = np.array(X)
lamda = 0.1
X2 = X1.transpose()
XXdash = np.matmul(X2, X1)
IX = inv(XXdash)
ymulX = np.matmul(X2, Y)
temp = np.matmul(ymulX,IX)
S= np.sign(temp)
temp_one = (S*(lamda/2))
temp_two = ymulX-temp_one
temp_three = np.matmul(temp_two,IX)
ZZ1 = np.matmul(X1, temp_three)
ZZ2 = ZZ1 >ZZ1.mean()
yhatTrain = ZZ2.astype(int)
CC = confusion_matrix(Y, yhatTrain)
pd.DataFrame(CC).to_csv('../data/csvfile/assign2/lassocc.csv', index = False)
TN = CC[1,1]
FP = CC[1,0]
FN = CC[0,1]
TP = CC[0,0]
FPFN = FP+FN
TPTN = TP+TN
Accuracy = 1/(1+(FPFN/TPTN))
print("Train_Accuracy_Score:",Accuracy)
Precision = 1/(1+(FP/TP))
print("Train_Precision_Score:",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Train_Sensitivity_Score:",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Train_Specificity_Score:",Specificity)
print('------------------------------------')

test_y = x_test[81].astype(int)
Y_test = np.array(test_y)
x_test.drop(81,axis=1,inplace=True)
x_test_np = np.array(x_test)

Z1_test = np.matmul(x_test_np, temp_three)
Z2_test = Z1_test > Z1_test.mean()

yhat_test = Z2_test.astype(int)

CC_test = confusion_matrix(test_y, yhat_test)
TN = CC_test[0,0]
FP = CC_test[0,1]
FN = CC_test[1,0]
TP = CC_test[1,1]
FPFN = FP+FN
TPTN = TP+TN
Accuracy = 1/(1+(FPFN/TPTN))
print("Test_Accuracy_Score:",Accuracy)
Precision = 1/(1+(FP/TP))
print("Test_Precision_Score:",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Test_Sensitivity_Score:",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Test_Specificity_Score:",Specificity)
print('------------------------------------')
#Inbuilt performance metrics
print('sklearn.metrics Accuracy',accuracy_score(test_y, yhat_test))
print('sklearn.metrics precision',precision_score(test_y, yhat_test))
print('sklearn.metrics sensitivity',recall_score(test_y, yhat_test))
