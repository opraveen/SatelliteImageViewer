#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 06:27:22 2017

@author: joe
"""
from sklearn.externals import joblib
import data_util as d_util
import random_forest_models as rfm
import mask_to_polygon as mtp
import matplotlib.pyplot as plt
import pandas as pd
import shapely
from shapely import affinity
import cv2
import numpy as np


def create_csv_submission_inner(multi_polygon, img_size, class_type, image_id):
    grid_sizes_panda = pd.read_csv('../data/grid_sizes.csv',
                                   names=['ImageId', 'Xmax', 'Ymin'],
                                   skiprows=1)
    xymax = d_util._get_xmax_ymin(grid_sizes_panda, image_id)
    [x_fact, y_fact] = mtp.get_xy_factor(img_size, xymax)
    all_polygons = []
    for polygon in multi_polygon:
        scaled = affinity.scale(polygon, xfact=x_fact, yfact=y_fact,
                                origin=(0, 0))
        all_polygons.append(scaled)
    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        #Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        #need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons


def get_all_masks():
    cnt = 0
    all_test_images = d_util.get_all_test_images_official()
    for class_id in [10]:
        print(class_id)
        model = joblib.load('models/rf_class_' + str(class_id) + '.pkl')
        for image_id in all_test_images:
            cnt += 1
            print(cnt)
            print(image_id)
            img = rfm.predict(image_id, class_id, model)
            plt.imsave('2000X2000Masks/{0}_{1}.png'.format(str(image_id),str(class_id)), img)
       #     multi_polygon = mtp.mask_to_polygons(img)
            del img
            #mtp.create_csv_submission_inner(multi_polygon,(HEIGHT, WIDTH),class_id, image_id)
            #del multi_polygon

def create_submission_from_masks():
    mtp.create_csv_initial()
    cnt = 0
    for class_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print(class_id)
        all_test_images = d_util.get_all_test_images_official()
        for image_id in all_test_images:
            cnt +=1
            print('cnt: ', cnt)
            print(class_id)
            print(image_id)
            path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
            img = cv2.imread(path)
            kernel = np.ones((2,2), np.uint8)
            img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            #print('Image: ', img.shape)
            del img
            img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, kernel)
            del img_close
            img_open = cv2.cvtColor(img_open, cv2.COLOR_BGR2GRAY)
            #print(img_open)
            img_open[img_open<16] = 0
            img_open[img_open>16] = 1
            multi_polygon = mtp.mask_to_polygons(img_open)
            #print(img_open)
            del img_open
            mtp.create_csv_submission_inner(multi_polygon,(1000, 1000),class_id, image_id)
            del multi_polygon

mtp.create_csv_initial()
create_submission_from_masks()
#get_all_masks()
print('Done')

#create_submission_from_masks()
#mtp.reorder_csv()

#image_id = '6010_0_0'
#class_id = 1
#classes = [class_id]
#image = d_util.overlay_polygons(image_id, classes)
#print('image', image)
#
#plt.imsave('testplot_given_verlay.png', image * 255)
#
#model = joblib.load('models/rf_class_' + str(class_id) + '.pkl')
#
#img = rfm.predict(image_id, class_id, model)
#
#plt.imsave('testplot_from_model.png', img)
#kernel = np.ones((3,3), np.uint8)
##image_from_model_polygon = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#image_from_model_polygon = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#print('image_from_model_polygon', image_from_model_polygon)
#plt.imsave('testplot_from_submit_poly2.png', image_from_model_polygon*255)
#print(1)
#polygons = mtp.mask_to_polygons(img)
#print('polygons', polygons)
#raster_size = (2000, 2000)
#print(1)
#grid_sizes_panda = pd.read_csv('../data/grid_sizes.csv',
#                                   names=['ImageId', 'Xmax', 'Ymin'],
#                                   skiprows=1)
#xymax = d_util._get_xmax_ymin(grid_sizes_panda, image_id)
#print('xymax', xymax)
#polygon_list = create_csv_submission_inner(polygons, raster_size, class_id, image_id)
#print('polygon_list', polygon_list)
#contours = d_util._get_and_convert_contours(polygon_list, raster_size, xymax)
#print('contours', contours)
#image_from_model_polygon = d_util._plot_mask_from_contours(raster_size,
#
#                                                      contours, 1)

