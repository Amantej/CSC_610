import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import time
x_train = pd.read_csv('../data/csvfile/assign2/non-overlap_Merged_image0image1image2_train.csv',header=None)
x_test = pd.read_csv('../data/csvfile/assign2/non-overlap_Merged_image0image1image2_test.csv',header=None)
    
y = x_train[81]
Y = np.array(y)
x_train.drop(81,axis=1,inplace=True)
X = x_train
X1 = np.array(X) 
t0= time.time()
randomFClassifier = RandomForestClassifier(random_state=0,n_estimators=500,oob_score=True, n_jobs=-1)
model = randomFClassifier.fit(X1, Y)
t1 = time.time() - t0
print("Time elapsed: ", t1)
importance = model.feature_importances_
indices = importance.argsort()[::-1]

oob_error = 1- randomFClassifier.oob_score_

test_y = x_test[81]
Y_test = np.array(test_y)
x_test.drop(81,axis=1,inplace=True)
x_test_np = np.array(x_test)
y_pred = randomFClassifier.predict(x_test_np)
CC_test = confusion_matrix(test_y, y_pred)
pd.DataFrame(CC_test).to_csv('../data/csvfile/assign2/rf3class.csv', index = False)

TP = CC_test[0,0]
TN = CC_test[1,1]+CC_test[1,2]+CC_test[2,1]+CC_test[2,2]
FP = CC_test[0,1]+CC_test[0,2]
FN = CC_test[1,0] + CC_test[2,0]

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

print(classification_report(test_y, y_pred, labels=[0, 1, 2]))

