#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 08:00:38 2017

@author: joe
"""
from descartes.patch import PolygonPatch
import polygon_to_classes as ptc
import sat_image_class as sic
import matplotlib.pyplot as plt
import pandas as pd

image_id =  '6120_2_2'
class_id = 1
def run_19_pixelwise(image_id):
    stacked = ptc.build_19_deep_layer(image_id)
    
    print('stacked.shape[0]: ', stacked.shape[0])
    print('stacked.shape[1]: ', stacked.shape[1])
    print('stacked.shape[2]: ', stacked.shape[2])
    cnt = 0
    for x in range(0, stacked.shape[0]):
        for y in range(0, stacked.shape[1]):
            print(stacked[x,y,:])
            cnt +=1
            if cnt>10000:
                break

    
inDir = '../data'


# read the training data from train_wkt_v4.csv
df = pd.read_csv(inDir + '/train_wkt_v4.csv')

# grid size will also be needed later..
gs = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

mask = ptc.generate_mask_for_image_and_class((3348,3403),"6120_2_2",4,gs,df)
#cv2.imwrite("mask.png",mask*255)
#print(mask)
for i in range(1000):
    for j in range(1000):
        print(mask[i,j])
plt.imshow(mask*255)
