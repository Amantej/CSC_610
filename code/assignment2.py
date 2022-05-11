import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#import seaborn as sns

image0 = pd.read_csv("../data/csvfile/sw_fspaceimage0.csv")
image1 = pd.read_csv("../data/csvfile/sw_fspaceimage1.csv")
image2 = pd.read_csv("../data/csvfile/sw_fspaceimage2.csv")

frames = [image0, image1]
Non_overlap_merge = pd.concat(frames)
Merge_Index = np.arange(len(Non_overlap_merge))
random_non_overlap_merge = np.random.permutation(Merge_Index)
Non_overlap_img01=Non_overlap_merge.sample(frac=1).reset_index(drop=True)

Non_overlap_img01.to_csv('../data/csvfile/assign2/overlap_Merged_image0image1.csv', index=False)

frames1 = [Non_overlap_img01, image2]
Non_overlap_merge1 = pd.concat(frames1)
Merge_Index1 = np.arange(len(Non_overlap_merge1))
random_non_overlap_merge1 = np.random.permutation(Merge_Index1)
Non_overlap_img012=Non_overlap_merge1.sample(frac=1).reset_index(drop=True)

Non_overlap_img012.to_csv('../data/csvfile/assign2/overlap_Merged_image0image1image2.csv', index=False)
Overlap_img01 = pd.read_csv('../data/csvfile/assign2/overlap_Merged_image0image1.csv',header=None)
Overlap_img012 = pd.read_csv('../data/csvfile/assign2/overlap_Merged_image0image1image2.csv',header=None)

Non_overlap_img01 = pd.read_csv('../data/csvfile/Merged_image0image1.csv', header=None)
Non_overlap_img012 = pd.read_csv('../data/csvfile/Merged_image0image1image2.csv', header=None)

NO_img01_Length, NO_img01_Width = Non_overlap_img01.shape
Threshold = round(NO_img01_Length*0.72)
NO_img01_Train = Non_overlap_img01[1:Threshold]
NO_img01_Test = Non_overlap_img01[Threshold:]
NO_img01_Train.to_csv('../data/csvfile/assign2/non-overlap_Merged_image0image1_Train.csv', index = False)
NO_img01_Test.to_csv('../data/csvfile/assign2/non-overlap_Merged_image0image1_Test.csv', index = False)
NO_img01_Train[[29,69]].hist(bins=81)
NO_img01_Test[[29,69]].hist(bins=81)
NO_img01_Train_Mean = NO_img01_Train[[29,69]].mean()
print('Mean values of Non_overlap img01 Training Data is: \n', NO_img01_Train_Mean)
NO_img01_Test_Mean = NO_img01_Test[[29,69]].mean()
print('Mean values of Non_overlap img01 Testing Data is: \n', NO_img01_Test_Mean)
NO_img01_Train_Var = np.var(NO_img01_Train[[29,69]])
print('Variance of Non_overlap img01 Training Data is: \n', NO_img01_Train_Var)
NO_img01_Test_Var = np.var(NO_img01_Test[[29,69]])
print('Variance of Non_overlap img01 Testing Data is: \n', NO_img01_Test_Var)
NO_img012_Length, NO_img012_Width = Non_overlap_img012.shape
Threshold = round(NO_img012_Length*0.72)
NO_img012_Train = Non_overlap_img012[1:Threshold]
NO_img012_Test = Non_overlap_img012[Threshold:]
NO_img012_Train.to_csv('../data/csvfile/assign2/non-overlap_Merged_image0image1image2_Train.csv', index = False)
NO_img012_Test.to_csv('../data/csvfile/assign2/non-overlap_Merged_image0image1image2_Test.csv', index = False)
NO_img012_Train[[29,69]].hist(bins=81)
NO_img012_Test[[29,69]].hist(bins=81)
NO_img012_Train_Mean = NO_img012_Train[[29,69]].mean()
print('Mean values of Non_overlap img012 Training Data is: \n', NO_img012_Train_Mean)
NO_img012_Test_Mean = NO_img012_Test[[29,69]].mean()
print('Mean values of Non_overlap img012 Testing Data is: \n', NO_img012_Test_Mean)
NO_img012_Train_Var = np.var(NO_img012_Train[[29,69]])
print('Variance of Non_overlap img012 Training Data is: \n', NO_img012_Train_Var)
NO_img012_Test_Var = np.var(NO_img012_Test[[29,69]])
print('Variance of Non_overlap img012 Testing Data is: \n', NO_img012_Test_Var)
OL_img01_Length, OL_img01_Width = Overlap_img01.shape
Threshold = round(OL_img01_Length*0.72)
OL_img01_Train = Overlap_img01[1:Threshold]
OL_img01_Test = Overlap_img01[Threshold:]
OL_img01_Train.to_csv('../data/csvfile/assign2/overlap_Merged_image0image1_Train.csv', index = False)
OL_img01_Test.to_csv('../data/csvfile/assign2/overlap_Merged_image0image1_Test.csv', index = False)

OL_img01_Train[[29,69]].hist(bins=81)
OL_img01_Test[[29,69]].hist(bins=81)
OL_img01_Train_Mean = OL_img01_Train[[29,69]].mean()
print('Mean values of Overlapping img01 Training Data is: \n', OL_img01_Train_Mean)
OL_img01_Test_Mean = OL_img01_Test[[29,69]].mean()
print('Mean values of Overlapping img01 Testing Data is: \n', OL_img01_Test_Mean)
OL_img01_Train_Var = np.var(OL_img01_Train[[29,69]])
print('Variance of Overlapping img01 Training Data is: \n', OL_img01_Train_Var)
OL_img01_Test_Var = np.var(OL_img01_Test[[29,69]])
print('Variance of Overlapping img01 Testing Data is: \n', OL_img01_Test_Var)

OL_img012_Length, OL_img012_Width = Overlap_img012.shape
Threshold = round(OL_img012_Length*0.72)
OL_img012_Train = Overlap_img012[1:Threshold]
OL_img012_Test = Overlap_img012[Threshold:]
OL_img012_Train.to_csv('../data/csvfile/assign2/overlap_Merged_image0image1image2_Train.csv', index = False)
OL_img012_Test.to_csv('../data/csvfile/assign2/overlap_Merged_image0image1image2_Test.csv', index = False)
OL_img012_Train[[29,69]].hist(bins=81)
OL_img012_Test[[29,69]].hist(bins=81)
OL_img012_Train_Mean = OL_img012_Train[[29,69]].mean()
print('Mean values of Overlapping img012 Training Data is: \n', OL_img012_Train_Mean)
OL_img012_Test_Mean = OL_img012_Test[[29,69]].mean()
print('Mean values of Overlapping img012 Testing Data is: \n', OL_img012_Test_Mean)
OL_img012_Train_Var = np.var(OL_img012_Train[[29,69]])
print('Variance of Overlapping img012 Training Data is: \n', OL_img012_Train_Var)
OL_img012_Test_Var = np.var(OL_img012_Test[[29,69]])
print('Variance of Overlapping img012 Testing Data is: \n', OL_img012_Test_Var)
