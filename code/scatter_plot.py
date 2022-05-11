import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
x=['non-overlap_Merged_image0image1_Train.csv','non-overlap_Merged_image0image1_Test.csv','non-overlap_Merged_image0image1image2_Train.csv','non-overlap_Merged_image0image1image2_Test.csv','overlap_Merged_image0image1_Train.csv','overlap_Merged_image0image1_Test.csv','overlap_Merged_image0image1image2_Train.csv','overlap_Merged_image0image1image2_Test.csv']
for i in x:
    f = os.path.join('../data/csvfile/assign2/', i)
    X_test = pd.read_csv(f,header=None)[1:]
#sns.scatterplot(data=X_test,x=X_test[10],y=X_test[80],hue=X_test[81])
#sns.scatterplot(data=X_test,x=X_test[10],y=X_test[80],hue=X_test[81])
    Y_test = X_test[:][81]
    X_test_np = np.array(X_test)
    Y_test_np = np.array(Y_test)    
    feature_29 = X_test_np[1:,29]
    feature_69 = X_test_np[1:,69]
    Y_test_val = Y_test_np[1:]
    colors = {0: 'gold', 1: 'black', 2:'lime'}
    fig, ax = plt.subplots()
    for g in np.unique(Y_test_val):
        ix = np.where(Y_test_val == g)
        ax.scatter(feature_29[ix], feature_69[ix], c = colors[g], label = g, s = 1)
    ax.legend()
    plt.xlabel('Feature 29')
    plt.ylabel('Feature 69')
    plt.title('Feature 29 and 69 of data %s' %i)
    plt.show()

