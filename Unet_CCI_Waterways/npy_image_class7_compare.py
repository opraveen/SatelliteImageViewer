# -*- coding: utf-8 -*-
"""
Created on Sun Mar 05 22:35:21 2017

@author: user
"""
# libs
from __future__ import division
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
#%matplotlib inline
from skimage.transform import resize
from collections import defaultdict
import pandas as pd
import cv2
import os
import shapely
from shapely.geometry import MultiPolygon, Polygon

def stretch_8bit(bands, lower_percent=2, higher_percent=98):
    out = np.zeros_like(bands).astype(np.float32)
    for i in range(3):
        a = 0 
        b = 1 
        c = np.percentile(bands[:,:,i], lower_percent)
        d = np.percentile(bands[:,:,i], higher_percent)        
        t = a + (bands[:,:,i] - c) * (b - a) / (d - c)    
        t[t<a] = a
        t[t>b] = b
        out[:,:,i] =t
    return out.astype(np.float32)

def CCCI_index(m, rgb):
    RE  = resize(m[5,:,:], (rgb.shape[0], rgb.shape[1])) 
    MIR = resize(m[7,:,:], (rgb.shape[0], rgb.shape[1])) 
    R = rgb[:,:,0]
    # canopy chloropyll content index
    CCCI = (MIR-RE)/(MIR+RE)*(MIR-R)/(MIR+R)
    return CCCI    

data = pd.read_csv('../input/train_wkt_v4.csv')
data = data[data.MultipolygonWKT != 'MULTIPOLYGON EMPTY']
grid_sizes_fname = '../input/grid_sizes.csv'
wkt_fname = '../input/train_wkt_v4.csv'
image_fname = '../input/three_band/'



# take some pictures from test 
waterway_test = ['6080_4_3','6080_4_0',
                 '6080_1_3', '6080_1_1',
                 '6150_3_4', '6050_2_1']

img_array = np.load('E:/agavranis/DSTL/0.42 Keras/msk/10_6010_0_0.npy')
print("img_array.shape=",img_array.shape)
print("img_array.dtype",img_array.dtype)

#from matplotlib import pyplot as plt

        
#for IM_ID in waterway_test:
#    i=1
#    img_array = np.load('E:/agavranis/DSTL/0.42 Keras/msk/10_'+ IM_ID +'.npy')
#    j=7
    #for i in range(len(waterway_test)):
#    plt.figure(i)
#    plt.suptitle("Class "+ str(j) +" Mask")
#    plt.imshow(img_array[j-1], cmap='gray')
#    plt.savefig(IM_ID+ str(j)+'_1.png',dpi=300)
#    i+=1

for IM_ID in waterway_test:
    # read rgb and m bands
    rgb = tiff.imread('../input/three_band/{}.tif'.format(IM_ID))
    rgb = np.rollaxis(rgb, 0, 3)
    m = tiff.imread('../input/sixteen_band/{}_M.tif'.format(IM_ID))
    img_array = np.load('E:/agavranis/DSTL/0.42 Keras/msk/10_'+ IM_ID +'.npy')[6]
    # get our index
    CCCI = CCCI_index(m, rgb) 
    
    # you can look on histogram and pick your favorite threshold value(0.11 is my best)
    binary = (CCCI > 0.11).astype(np.float32)
    binary2 = resize(binary, (img_array.shape[0], img_array.shape[1]))
    print( binary.sum())
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    ax = axes.ravel()
    ax[0].imshow(stretch_8bit(rgb))
    ax[0].set_title('Image')
    ax[0].axis('off')
    ax[1].imshow(binary, cmap='gray')
    ax[1].set_title('CCI')
    ax[1].axis('off')
    ax[2].imshow(binary2, cmap='gray')
    ax[2].set_title('CCI_Resize')
    ax[2].axis('off')
    ax[3].imshow(img_array, cmap='gray')
    ax[3].set_title('Binary')
    ax[3].axis('off')
    plt.tight_layout()
    plt.savefig(IM_ID+'6_tot.png',dpi=300)
    plt.show()

      
       
       
       