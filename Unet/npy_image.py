# -*- coding: utf-8 -*-
"""
Created on Sun Mar 05 22:35:21 2017

@author: user
"""

import numpy as np

img_array = np.load('E:/agavranis/DSTL/0.42 Keras/msk/10_6010_0_0.npy')
print("img_array.shape=",img_array.shape)
print("img_array.dtype",img_array.dtype)

from matplotlib import pyplot as plt

for i in range(10):
    plt.figure(i)
    plt.suptitle("Class "+ str(i+1) +" Mask")
    plt.imshow(img_array[i], cmap='gray')
    plt.savefig('10_6010_0_0'+ str(i+1)+'.png',dpi=300)

