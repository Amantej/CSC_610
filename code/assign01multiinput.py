import pandas as pd
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os 
def imresize(height,width):
 ratio=261/height
 width=width*ratio
 temp=int(width)/9
 width=int(temp)*9
 height=261
 return(height,int(width))
path = '../data/input_images/FIDS30/'
feature_name = str(input('Name of Fruits: '))

count = int(input('Enter number of files: '))
for i in range(0,count):
    ig = str(input('Enter File name: '))
    imagepath = path+feature_name+'/'+ig
    image= cv2.imread(imagepath)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    raw_height, raw_width = image.shape
    height, width = imresize(raw_height,raw_width)
    image=cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    ri_tmp = np.zeros((height, width), np.uint8)
    ri_th1 = image.mean()
    for k in range(height):
        for l in range(width):
            if(image[k][l]<ri_th1):
                ri_tmp[k][l] = 0
            else:
                ri_tmp[k][l] = 261 
    plt.imshow(ri_tmp, cmap=plt.get_cmap('gray'))
    ri_oo = round(((height-9)*(width-9))/81)
    ri_flato = np.zeros((ri_oo, 82), np.uint8)
    p = 0
    for k in range(0,height-9,9):
        for l in range(0,width-9,9):
            ri_crop_tmp1 = image[k:k+9,l:l+9]
            ri_flato[p,0:81] = ri_crop_tmp1.flatten()
            p = p + 1
    ri_fspaceO = pd.DataFrame(ri_flato) 
    ri_fspaceO.to_csv('../data/csvfile/assign1_multi/fspace'+feature_name+str(i)+'.csv', index=False)
 
    ri_otable_size = round((height-9)*(width-9))
    ri_sw_flato = np.zeros((ri_otable_size, 82), np.uint8)
    p = 0
    for k in range(0,height-9):
        for l in range(0,width-9):
            ri_sw_crop_tmp1 = image[k:k+9,l:l+9]
            ri_sw_flato[p,0:81] = ri_sw_crop_tmp1.flatten()
            p = p + 1
    ri_sw_fspaceC = pd.DataFrame(ri_sw_flato) 
    ri_sw_fspaceC.to_csv('../data/csvfile/assign1_multi/sw_fspace'+feature_name+str(i)+'.csv', index=False)
