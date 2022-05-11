import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

image0 =  cv2.imread("../data/input_images/FIDS30/oranges/9.jpg")
image1 =  cv2.imread("../data/input_images/FIDS30/bananas/43.jpg")
image2 =  cv2.imread("../data/input_images/FIDS30/strawberries/37.jpg")

plt.imshow(image0[:,:,0])
plt.imshow(image0[:,:,1])
plt.imshow(image0[:,:,2])

plt.imshow(image1[:,:,0])
plt.imshow(image1[:,:,1])
plt.imshow(image1[:,:,2])

plt.imshow(image2[:,:,0])
plt.imshow(image2[:,:,1])
plt.imshow(image2[:,:,2])


image0Gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY) 
height_image0Gray, width_image0gray = image0Gray.shape

image1Gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
height_image1Gray, width_image1gray = image1Gray.shape 

image2Gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) 
height_image2Gray, width_image2gray = image2Gray.shape 

def imresize(height,width):
    ratio=261/height
    width=width*ratio
    temp=int(width)/9
    width=int(temp)*9
    height=261
    return(height,int(width))

height_image0Gray, width_image0gray = imresize(height_image0Gray,width_image0gray) #for cantaloupes
height_image1Gray, width_image1gray = imresize(height_image1Gray,width_image1gray) #for resimage1nas
height_image2Gray, width_image2gray = imresize(height_image2Gray,width_image2gray) #for resimage2wberries

resimage0 = cv2.resize(image0Gray, dsize=(width_image0gray, height_image0Gray), interpolation=cv2.INTER_CUBIC)
resimage1 = cv2.resize(image1Gray, dsize=(width_image1gray, height_image1Gray), interpolation=cv2.INTER_CUBIC)
resimage2 = cv2.resize(image2Gray, dsize=(width_image2gray, height_image2Gray), interpolation=cv2.INTER_CUBIC)


plt.imshow(resimage0, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.imshow(resimage1, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.imshow(resimage2, cmap=plt.get_cmap('gray'))
plt.axis('off')

cv2.imwrite('../data/output_images/image0.jpg', resimage0)
cv2.imwrite('../data/output_images/image1.jpg', resimage1)
cv2.imwrite('../data/output_images/image2.jpg', resimage2)

tmpo = np.zeros((height_image0Gray, width_image0gray), np.uint8)
th1 = resimage0.mean()
for i in range(height_image0Gray):
    for j in range(width_image0gray):
        if(resimage0[i][j]<th1):
            tmpo[i][j] = 0
        else:
            tmpo[i][j] = 261  

plt.imshow(tmpo, cmap=plt.get_cmap('gray'))

tmpb = np.zeros((height_image1Gray, width_image1gray), np.uint8)
th2 = resimage1.mean()
for i in range(height_image1Gray):
    for j in range(width_image1gray):
        if(resimage1[i][j]<th2):
            tmpb[i][j] = 0
        else:
            tmpb[i][j] = 261  

plt.imshow(tmpb, cmap=plt.get_cmap('gray'))

tmps = np.zeros((height_image2Gray, width_image2gray), np.uint8)
th3 = resimage2.mean()
for i in range(height_image2Gray):
    for j in range(width_image2gray):
        if(resimage2[i][j]<th3):
            tmps[i][j] = 0
        else:
            tmps[i][j] = 261  

plt.imshow(tmps, cmap=plt.get_cmap('gray'))

oo = round(((height_image0Gray-9)*(width_image0gray-9))/81)
flato = np.zeros((oo, 82), np.uint8)
k = 0
for i in range(0,height_image0Gray-9,9):
    for j in range(0,width_image0gray-9,9):
        crop_tmp1 = resimage0[i:i+9,j:j+9]
        flato[k,0:81] = crop_tmp1.flatten()
        k = k + 1

fspaceimage0 = pd.DataFrame(flato)  
fspaceimage0.to_csv('../data/csvfile/fspaceimage0.csv', index=False)

bb = round(((height_image1Gray-9)*(width_image1gray-9))/81)
flatb = np.ones((bb, 82), np.uint8)
k = 0
for i in range(0,height_image1Gray-9,9):
    for j in range(0,width_image1gray-9,9):
        crop_tmp2 = resimage1[i:i+9,j:j+9]
        flatb[k,0:81] = crop_tmp2.flatten()
        k = k + 1

fspaceimage1 = pd.DataFrame(flatb)  
fspaceimage1.to_csv('../data/csvfile/fspaceimage1.csv', index=False)

ss = round(((height_image2Gray-9)*(width_image2gray-9))/81)
flats = np.full((ss, 82), 2)
k = 0
for i in range(0,height_image2Gray-9,9):
    for j in range(0,width_image2gray-9,9):
        crop_tmp3 = resimage2[i:i+9,j:j+9]
        flats[k,0:81] = crop_tmp3.flatten()
        k = k + 1

fspaceimage2 = pd.DataFrame(flats)
fspaceimage2.to_csv('../data/csvfile/fspaceimage2.csv', index=False)

otable_size = round((height_image0Gray-9)*(width_image0gray-9))
sw_flato = np.zeros((otable_size, 82), np.uint8)
k = 0
for i in range(0,height_image0Gray-9):
    for j in range(0,width_image0gray-9):
        sw_crop_tmp1 = resimage0[i:i+9,j:j+9]
        sw_flato[k,0:81] = sw_crop_tmp1.flatten()
        k = k + 1
sw_fspaceimage0 = pd.DataFrame(sw_flato)  
sw_fspaceimage0.to_csv('../data/csvfile/sw_fspaceimage0.csv', index=False)


btable_size = round((height_image1Gray-9)*(width_image1gray-9))
sw_flatb = np.ones((btable_size, 82), np.uint8)
k = 0
for i in range(0,height_image1Gray-9):
    for j in range(0,width_image1gray-9):
        sw_crop_tmp2 = resimage1[i:i+9,j:j+9]
        sw_flatb[k,0:81] = sw_crop_tmp2.flatten()
        k = k + 1
sw_fspaceimage1 = pd.DataFrame(sw_flatb) 
sw_fspaceimage1.to_csv('../data/csvfile/sw_fspaceimage1.csv', index=False)

stable_size = round((height_image2Gray-9)*(width_image2gray-9))
sw_flats = np.full((stable_size, 82), 2)
k = 0
for i in range(0,height_image2Gray-9):
    for j in range(0,width_image2gray-9):
        sw_crop_tmp3 = resimage2[i:i+9,j:j+9]
        sw_flats[k,0:81] = sw_crop_tmp3.flatten()
        k = k + 1
sw_fspaceimage2 = pd.DataFrame(sw_flats)  #panda object
sw_fspaceimage2.to_csv('../data/csvfile/sw_fspaceimage2.csv', index=False)


number_observations_fspaceimage0=fspaceimage0.shape[0]
number_observations_fspaceimage1=fspaceimage1.shape[0]
number_observations_fspaceimage2=fspaceimage2.shape[0]

dimension_fspaceimage0=fspaceimage0.shape[1]-1
dimension_fspaceimage1=fspaceimage1.shape[1]-1
dimension_fspaceimage2=fspaceimage2.shape[1]-1

mean_fspaceimage0=fspaceimage0[fspaceimage0.columns[0:81]].mean()
sd_fspaceimage0=fspaceimage0[fspaceimage0.columns[0:81]].std()

mean_fspaceimage1=fspaceimage1[fspaceimage1.columns[0:81]].mean()
sd_fspaceimage1=fspaceimage1[fspaceimage1.columns[0:81]].std()

mean_fspaceimage2=fspaceimage2[fspaceimage2.columns[0:81]].mean()
sd_fspaceimage2=fspaceimage2[fspaceimage2.columns[0:81]].std()

print("Number of observations in image0G 9X9 block feature is: ", number_observations_fspaceimage0)
print("Number of observations in image1 9X9 block feature is: ", number_observations_fspaceimage1)
print("Number of observations in image2 9X9 block feature is: ", number_observations_fspaceimage2)

print("Dimension of Data in image0 9X9 block feature is: ", dimension_fspaceimage0)
print("Dimension of Data in image1 9X9 block feature is: ", dimension_fspaceimage1)
print("Dimension of Data in image2 9X9 block feature is: ", dimension_fspaceimage2)

print("Mean of image0 9X9 block feature is: \n", mean_fspaceimage0)
print("Mean of image1 9X9 block feature is: \n", mean_fspaceimage1)
print("Mean of image2 9X9 block feature is: ", mean_fspaceimage2)

print("Standard Deviation of image0 9X9 block feature is: \n", sd_fspaceimage0)
print("Standard Deviation of image1 9X9 block feature is: \n", sd_fspaceimage1)
print("Standard Deviation of image2 9X9 block feature is: ", sd_fspaceimage2)

mean_fspaceimage0.plot()
mean_fspaceimage1.plot()
mean_fspaceimage2.plot()

sd_fspaceimage0.plot()
sd_fspaceimage1.plot()
sd_fspaceimage2.plot()

random_value=np.random.randint(81)
fspaceimage0[random_value].hist(bins=81)
fspaceimage1[random_value].hist(bins=81)
fspaceimage2[random_value].hist(bins=81)


number_observations_sw_fspaceimage0=sw_fspaceimage0.shape[0]
number_observations_sw_fspaceimage1=sw_fspaceimage1.shape[0]
number_observations_sw_fspaceimage2=sw_fspaceimage2.shape[0]

dimension_sw_fspaceimage0=sw_fspaceimage0.shape[1]-1
dimension_sw_fspaceimage1=sw_fspaceimage1.shape[1]-1
dimension_sw_fspaceimage2=sw_fspaceimage2.shape[1]-1

mean_sw_fspaceimage0=sw_fspaceimage0[sw_fspaceimage0.columns[0:81]].mean()
sd_sw_fspaceimage0=sw_fspaceimage0[sw_fspaceimage0.columns[0:81]].std()

mean_sw_fspaceimage1=sw_fspaceimage1[sw_fspaceimage1.columns[0:81]].mean()
sd_sw_fspaceimage1=sw_fspaceimage1[sw_fspaceimage1.columns[0:81]].std()

mean_sw_fspaceimage2=sw_fspaceimage2[sw_fspaceimage2.columns[0:81]].mean()
sd_sw_fspaceimage2=sw_fspaceimage2[sw_fspaceimage2.columns[0:81]].std()

print("Number of observations in image0 sliding block feature is: ", number_observations_sw_fspaceimage0)
print("Number of observations in image1 sliding block feature is: ", number_observations_sw_fspaceimage1)
print("Number of observations in image2 sliding block feature is: ", number_observations_sw_fspaceimage2)

print("Dimension of Data in image0 sliding block feature is: ", dimension_sw_fspaceimage0)
print("Dimension of Data in image1 sliding block feature is: ", dimension_sw_fspaceimage1)
print("Dimension of Data in image2 sliding block feature is: ", dimension_sw_fspaceimage2)

print("Mean of image0 sliding block feature is: \n", mean_sw_fspaceimage0)
print("Mean of image1 sliding block feature is: \n", mean_sw_fspaceimage1)
print("Mean of image2 sliding block feature is: ", mean_sw_fspaceimage2)

print("Standard Deviation of image0 sliding block feature is: \n", sd_sw_fspaceimage0)
print("Standard Deviation of image1 sliding block feature is: \n", sd_sw_fspaceimage1)
print("Standard Deviation of image2 sliding block feature is: ", sd_sw_fspaceimage2)

mean_sw_fspaceimage0.plot()
mean_sw_fspaceimage1.plot()
mean_sw_fspaceimage2.plot()

sd_sw_fspaceimage0.plot()
sd_sw_fspaceimage1.plot()
sd_sw_fspaceimage2.plot()

random_value=np.random.randint(81)
sw_fspaceimage0[random_value].hist(bins=81)
sw_fspaceimage1[random_value].hist(bins=81)
sw_fspaceimage2[random_value].hist(bins=81)

frames = [fspaceimage0, fspaceimage1]
merge1 = pd.concat(frames)

index1 = np.arange(len(merge1))
random_merge1 = np.random.permutation(index1)

random_merge1=merge1.sample(frac=1).reset_index(drop=True)

random_merge1.to_csv('../data/csvfile/Merged_image0image1.csv', index=False)

frames2 = [random_merge1, fspaceimage2]
merge2 = pd.concat(frames2)

index2 = np.arange(len(merge2))
random_merge2 = np.random.permutation(index2)

random_merge2=merge2.sample(frac=1).reset_index(drop=True)

random_merge2.to_csv('../data/csvfile/Merged_image0image1image2.csv', index=False)


x=np.random.randint(81)
y=np.random.randint(81)
d = pd.DataFrame(random_merge2[[0,1,81]].values,columns=['First','Second','Labels'])

import seaborn as sns
sns.scatterplot(data=d,x='First',y='Second',hue='Labels')


import matplotlib
fig = plt.figure()
ax = plt.axes(projection='3d')
bx = random_merge2[22]
by = random_merge2[7]
bz = random_merge2[17]
ax.scatter3D(bx,by,bz,c=random_merge2[81],cmap=matplotlib.colors.ListedColormap(['red','blue','green']))
plt.show()
