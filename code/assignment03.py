import pandas as pd
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os 
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import time
def imresize(height,width):
    ratio=261/height
    width=width*ratio
    temp=int(width)/9
    width=int(temp)*9
    height=261
    return(height,int(width))
path = '../data/input_images/FIDS30/'
feature_name = 'bananas'
count = 5
df = pd.DataFrame()
for i in range(0,count):
    #ig = str(input('Enter File name: '))
    imagepath = path+feature_name+'/'+str(i+1)+'.jpg'
    print(imagepath)
    image= cv2.imread(imagepath)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    print('original image shape:',image.shape)
    raw_height, raw_width = image.shape
    height, width = imresize(raw_height,raw_width)
    image=cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    #plt.imshow(image, cmap=plt.get_cmap('gray'))
    #plt.axis('off')
    ri_tmp = np.zeros((height, width), np.uint8)
    ri_th1 = image.mean()
    for k in range(height):
        for l in range(width):
            if(image[k][l]<ri_th1):
                ri_tmp[k][l] = 0
            else:
                ri_tmp[k][l] = 261 
    #plt.imshow(ri_tmp, cmap=plt.get_cmap('gray'))
    ri_oo = round(((height-9)*(width-9))/81)
    ri_flato = np.zeros((ri_oo, 82), np.uint8)
    p = 0
    for k in range(0,height-9,9):
        for l in range(0,width-9,9):
            ri_crop_tmp1 = image[k:k+9,l:l+9]
            ri_flato[p,0:81] = ri_crop_tmp1.flatten()
            p = p + 1
    print('feature space rows, columsn:',pd.DataFrame(ri_flato).shape)
    df = df.append(pd.DataFrame(ri_flato)) 
print('Next class inputs')
feature_name = 'strawberries'
for i in range(0,count):
    imagepath = path+feature_name+'/'+str(i+1)+'.jpg'
    print(imagepath)
    image= cv2.imread(imagepath)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    print('original image shape:',image.shape)
    raw_height, raw_width = image.shape
    height, width = imresize(raw_height,raw_width)
    image=cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    #plt.imshow(image, cmap=plt.get_cmap('gray'))
    #plt.axis('off')
    ri_tmp = np.zeros((height, width), np.uint8)
    ri_th1 = image.mean()
    for k in range(height):
        for l in range(width):
            if(image[k][l]<ri_th1):
                ri_tmp[k][l] = 0
            else:
                ri_tmp[k][l] = 261 
    #plt.imshow(ri_tmp, cmap=plt.get_cmap('gray'))
    ri_oo = round(((height-9)*(width-9))/81)
    ri_flato = np.ones((ri_oo, 82), np.uint8)
    p = 0
    for k in range(0,height-9,9):
        for l in range(0,width-9,9):
            ri_crop_tmp2 = image[k:k+9,l:l+9]
            ri_flato[p,0:81] = ri_crop_tmp2.flatten()
            p = p + 1
    print('feature space rows, columsn:',pd.DataFrame(ri_flato).shape)
    df = df.append(pd.DataFrame(ri_flato)) 
    
df.to_csv('../data/csvfile/assign3/fspacenonoverlap.csv', index=False)

########################################################################################

input_data = pd.read_csv('../data/csvfile/assign3/fspacenonoverlap.csv', header = None)[1:]
input_data = input_data.sample(frac=1).reset_index(drop=True)
def randomcl(df):
    input_data1 = df.copy()
    y = input_data1[81]
    input_data1.drop(81,axis=1,inplace=True)
    X = input_data1
    tmp = np.array(X)
    X1 = tmp[:,0:81] 
    Y1 = np.array(y)

    row, col = X.shape
    TR = round(row*0.72)
    TT = row-TR

    X1_train = X1[0:TR-1,:]
    Y1_train = Y1[0:TR-1]
    t0= time.time()
    rF = RandomForestClassifier(random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1)
    model = rF.fit(X1_train,Y1_train)
    t1 = time.time() - t0
    print("Time elapsed: ", t1)

    X1_test = X1[TR:row,:]
    y_test = Y1[TR:row]
    yhat_test = rF.predict(X1_test)
    # Confusion matrix analytics
    CC_test = confusion_matrix(y_test, yhat_test)
    TN = CC_test[1,1]
    FP = CC_test[1,0]
    FN = CC_test[0,1]
    TP = CC_test[0,0]
    FPFN = FP+FN
    TPTN = TP+TN
    Accuracy = 1/(1+(FPFN/TPTN))
    print('From confusion matrix:')
    print("Our_Accuracy_Score:",Accuracy)
    Precision = 1/(1+(FP/TP))
    print("Our_Precision_Score:",Precision)
    Sensitivity = 1/(1+(FN/TP))
    print("Our_Sensitivity_Score:",Sensitivity)
    Specificity = 1/(1+(FP/TN))
    print("Our_Specificity_Score:",Specificity)
    # Built-in accuracy measure
    #from sklearn import metrics
    print('************************************************************')
    print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, yhat_test))
    print("BuiltIn_Precision:",metrics.precision_score(y_test, yhat_test))
    print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, yhat_test))
    print('************************************************************')

randomcl(input_data)
##################################################################################

def anncl(df):
    input_data1 = df.copy()
    NN = 81
    y = input_data1[81]
    input_data1.drop(81,axis=1,inplace=True)
    X = input_data1
    tmp = np.array(X)
    X1 = tmp[:,0:81] 
    Y1 = np.array(y)

    row, col = X.shape
    TR = round(row*0.72)
    TT = row-TR

    X1_train = X1[0:TR-1,:]
    Y1_train = Y1[0:TR-1]
    y = pd.get_dummies(Y1_train)
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
    t0= time.time()
    ANNmodel.fit(X1_train, y, batch_size=20, epochs=20)
    t1 = time.time() - t0
    print("Time elapsed: ", t1)
    #ANNmodel.save_weights("C:/Users/s_suthah/Desktop/Book-Data/model.h5")
    #print("Saved model to disk")
    # Testing with 20% data
    X1_test = X1[TR:row,:]
    y_test = Y1[TR:row]
    #y_test = pd.get_dummies(Y1_test)
    yhat_test = ANNmodel.predict_classes(X1_test)
    # Confusion matrix analytics
    CC_test = confusion_matrix(y_test, yhat_test)
    #TN = CC_test[0,0]
    #FP = CC_test[0,1]
    #FN = CC_test[1,0]
    #TP = CC_test[1,1]
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
    print('************************************************************')
    print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, yhat_test))
    print("BuiltIn_Precision:",metrics.precision_score(y_test, yhat_test))
    print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, yhat_test))
    print('************************************************************')

anncl(input_data)

########################################################################################

#model = Sequential()
#model.add(Dense(512, input_dim=81, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
## compile the keras model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## fit the keras model on the dataset
#t01= time.time()
#model.fit(X1_train, Y1_train, epochs = 75, batch_size=10)
#t11 = time.time() - t01
#print("Time elapsed: ", t11)
## evaluate the keras model
#_, accuracy = model.evaluate(X1_train,Y1_train)
#print('training Accuracy: %.2f' % (accuracy*100))
#_, accuracy = model.evaluate(X1_test,y_test)
#print('testing Accuracy: %.2f' % (accuracy*100))

########################################################################################

#temp1 = input_data.iloc[:,:80].sum(axis=1)
#temp_df = pd.DataFrame(temp1)
#temp_df['bar'] = pd.qcut(temp_df[0], q=13, precision=0, duplicates='drop')
##print(temp_df['bar'].value_counts())
#cond0 =  (temp_df[0] > 20311.0) & (temp_df[0] <= 20400.0)
#cond1 = (temp_df[0] > 11158.0) & (temp_df[0] <= 12378.0)
#cond2 = (temp_df[0] > 19063.0) & (temp_df[0] <= 20311.0)
#cond3 = (temp_df[0] > 17324.0) & (temp_df[0] <= 19063.0)
#cond4 = (temp_df[0] > 14842.0) & (temp_df[0] <= 15949.0)
#cond5 = (temp_df[0] > 8814.0) & (temp_df[0] <= 11158.0)
#cond6 = (temp_df[0] > 162.0) & (temp_df[0] <= 8814.0)
#cond7 = (temp_df[0] > 15949.0) & (temp_df[0] <= 17324.0)
#cond8 = (temp_df[0] > 13609.0) & (temp_df[0] <= 14842.0)
#cond9 = (temp_df[0] > 12378.0) & (temp_df[0] <= 13609.0)
#conditions = [cond0, cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8, cond9]
#index = temp_df.index
#for i in range(0,10):
#    indi = index[conditions[i]]
#    temp = input_data.filter(items = indi, axis=0)
#    print(temp.shape)
#    temp.to_csv('../data/csvfile/assign3/subdataset'+str(i)+'.csv', index=False)

########################################################################################

temp1 = input_data.iloc[:,:80].sum(axis=1)
temp_df = pd.DataFrame(temp1)
temp_df['bar'] = pd.qcut(temp_df[0], q=13, precision=0, duplicates='drop')
#print(temp_df['bar'].value_counts())
u = []
for i in temp_df['bar']: 
    u.append(str(i))
temp_df['bar'] = u
cond = list(temp_df['bar'].unique())
fil = []
cond = list(cond)
for i in range(0,10):
    fil.append(temp_df['bar'] == cond[i])
index = temp_df.index
for i in range(0,10):
    indi = index[fil[i]]
    temp = input_data.filter(items = indi, axis=0)
    print(temp.shape)
    temp.to_csv('../data/csvfile/assign3/subdataset'+str(i)+'.csv', index=False)


########################################################################################
print('subdatasets to random forest classifier')
for i in range(0,10):
    df = pd.read_csv('../data/csvfile/assign3/subdataset'+str(i)+'.csv',header = None)[1:]
    randomcl(df)
    print('************************************************************')

print('subdatasets to Artificial neural networks')
print('################################################################################')
for i in range(0,10):
    df = pd.read_csv('../data/csvfile/assign3/subdataset'+str(i)+'.csv',header = None)[1:]
    anncl(df)
    print('************************************************************')