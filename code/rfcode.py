import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
input_data = pd.read_csv('../data/csvfile/Merged_image0image1.csv',header=None)
y = input_data[81]
input_data.drop(81,axis=1,inplace=True)
X = input_data

tmp = np.array(X)
X1 = tmp[:,0:81] 
Y1 = np.array(y)

row, col = X.shape
TR = round(row*0.72)
TT = row-TR

X1_train = X1[0:TR-1,:]
Y1_train = Y1[0:TR-1]
rF = RandomForestClassifier(random_state=0, n_estimators=500, 
oob_score=True, n_jobs=-1)
model = rF.fit(X1_train,Y1_train)
importance = model.feature_importances_
indices = importance.argsort()[::-1]
##########################################################################
################
#https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
std = np.std([model.feature_importances_ for model in rF.estimators_], 
axis=0)
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], 
importance[indices[f]]))
plt.bar(range(X.shape[1]), importance[indices], color="r", 
yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices+1, rotation=90)
plt.show()
##########################################################################

X1_test = X1[TR:row,:]
y_test = Y1[TR:row]
yhat_test = rF.predict(X1_test)
# Confusion matrix analytics
CC_test = confusion_matrix(y_test, yhat_test)
pd.DataFrame(CC_test).to_csv('../data/csvfile/assign2/rf2class.csv', index = False)
TN = CC_test[1,1]
FP = CC_test[1,0]
FN = CC_test[0,1]
TP = CC_test[0,0]
FPFN = FP+FN
TPTN = TP+TN
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)
Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)
# Built-in accuracy measure
#from sklearn.metrics import classification_report
from sklearn import metrics
#print(classification_report(y_test, yhat_test, labels=[0, 1, 2]))
print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, yhat_test))
print("BuiltIn_Precision:",metrics.precision_score(y_test, yhat_test))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, yhat_test,))
