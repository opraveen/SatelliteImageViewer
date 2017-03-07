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
import time
import os


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


def get_all_masks(size):
    cnt = 0
    all_test_images = d_util.get_all_test_images_official()
    start = time.time()
    for class_id in [1,2,3,4,5,6,7,8,9,10]:
        print(class_id)
        model = joblib.load('models/{0}/rf_class_{1}.pkl'.format(str(size),str(class_id)))
        end = time.time()
        duration = end - start
        print('seconds program has been running for: ', duration)
        for image_id in all_test_images:
            cnt += 1
            print(cnt)
            print(image_id)
            img = rfm.predict(image_id, class_id, model)
            plt.imsave('{0}Masks/{1}_{2}.png'.format(str(size),str(image_id),str(class_id)), img)
            #time.sleep(1.5)
       #     multi_polygon = mtp.mask_to_polygons(img)
            del img
            #time.sleep(1.5)
            #mtp.create_csv_submission_inner(multi_polygon,(HEIGHT, WIDTH),class_id, image_id)
            #del multi_polygon
def mask_image(img_target):
    img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
    #print(img_target)
    img_target[img_target<18] = 0
    img_target[img_target>18] = 1
    return img_target.astype(bool)


def and_not_image(image_id, class_id):
    path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_target = cv2.imread(path)
    img_target = cv2.resize(img_target, (1000, 1000))
    img_target = mask_image(img_target)

    path_tree = '2000X2000Masks/{0}_5.png'.format(str(image_id),str(class_id))
    img_tree = cv2.imread(path_tree)
    img_tree = cv2.resize(img_tree, (1000, 1000))
    img_tree = mask_image(img_tree)
    not_tree = np.logical_not(img_tree)
    #print(img_tree)
    path_crop = '2000X2000Masks/{0}_6.png'.format(str(image_id),str(class_id))
    img_crop = cv2.imread(path_crop)
    img_crop = cv2.resize(img_crop, (1000, 1000))
    img_crop = mask_image(img_crop)
    not_crop = np.logical_not(img_crop)

    path_water = '2000X2000Masks/{0}_7.png'.format(str(image_id),str(class_id))
    img_water = cv2.imread(path_water)
    img_water = cv2.resize(img_water, (1000, 1000))
    img_water = mask_image(img_water)
    not_water = np.logical_not(img_water)

    path_water2 = '2000X2000Masks/{0}_8.png'.format(str(image_id),str(class_id))
    img_water2 = cv2.imread(path_water2)
    img_water2 = cv2.resize(img_water2, (1000, 1000))
    img_water2 = mask_image(img_water2)
    not_water2 = np.logical_not(img_water2)
    ret_img = np.logical_and(img_target,not_tree)
    ret_img = np.logical_and(ret_img,not_crop)
    ret_img = np.logical_and(ret_img,not_water)
    ret_img = np.logical_and(ret_img,not_water2).astype(int)
    #print('ret_img', ret_img)
    #return ret_img
    plt.imshow(ret_img*255)
    return ret_img
    
#image_id = '6030_3_0'
#class_id = '4'
#path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),
#                                                           str(class_id))
#img = cv2.imread(path)
#img = cv2.resize(img, (300, 300))
#img = mask_image(img)
#plt.imshow(img*255)
#and_not_image(image_id, class_id)
def getSize(filename):
    st = os.stat(filename)
    return st.st_size
def create_submission_from_masks():
    mtp.create_csv_initial()
    cnt = 0
    all_test_images = d_util.get_all_test_images_official()
    start = time.time()
    for image_id in all_test_images:
        end = time.time()
        duration = end - start
        print('seconds program has been running for: ', duration)
        percent_done = cnt/4290
        filesize = float(getSize('submission_file.csv'))/1024.0
        print('filesize: ', filesize)
        print('Estimated completed files size:', filesize+(1-percent_done)*(filesize/(percent_done+.001)))
        print('Estimated seconds until done:', (1-percent_done)*(duration/(percent_done+.001)))
        print('Estimated hours until done:', (1-percent_done)*(duration/(percent_done+.001))/3600)
        for class_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            cnt +=1
            print('cnt: ', cnt)
            print('class_id: ', class_id)
            print(image_id)
#            path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
#            img_open = cv2.imread(path)
#            img = cv2.imread(path)
#            kernel = np.ones((2,2), np.uint8)
            path = '849X836combinedMasks/{0}_{1}.png'.format(str(image_id),
                                                       str(class_id))
            img = cv2.imread(path)
            img = cv2.resize(img, (1000, 1000))
            img = mask_image(img)
            multi_polygon = mtp.mask_to_polygons(img, class_id)
            mtp.create_csv_submission_inner(multi_polygon,(img.shape[1],img.shape[0]),
                                            class_id, image_id)
#            if class_id == 1:
#                img = and_not_image(image_id, class_id)
#                multi_polygon = mtp.mask_to_polygons(img, class_id)
#                mtp.create_csv_submission_inner(multi_polygon,(1000, 1000),
#                                                class_id, image_id)
#            if class_id == 2:
#                path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),
#                                                           str(class_id))
#                img = cv2.imread(path)
#                img = cv2.resize(img, (1000, 1000))
#                img = mask_image(img)
#                multi_polygon = mtp.mask_to_polygons(img, class_id)
#                mtp.create_csv_submission_inner(multi_polygon,(1000, 1000),
#                                                class_id, image_id)
#            if class_id == 3:
#                path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),
#                                                           str(class_id))
#                img = cv2.imread(path)
#                img = cv2.resize(img, (1000, 1000))
#                img = mask_image(img)
#                multi_polygon = mtp.mask_to_polygons(img, class_id)
#                mtp.create_csv_submission_inner(multi_polygon,(1000, 1000),
#                                                class_id, image_id)
#            if class_id == 4:
#                path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),
#                                                           str(class_id))
#                img = cv2.imread(path)
#                img = cv2.resize(img, (1000, 1000))
#                img = mask_image(img)
#                multi_polygon = mtp.mask_to_polygons(img, class_id)
#                mtp.create_csv_submission_inner(multi_polygon,(1000, 1000),
#                                                class_id, image_id)
#            if class_id == 5:
#                path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),
#                                                           str(class_id))
#                img = cv2.imread(path)
#                img = cv2.resize(img, (1000, 1000))
#                img = mask_image(img)
#                multi_polygon = mtp.mask_to_polygons(img, class_id)
#                mtp.create_csv_submission_inner(multi_polygon,(1000, 1000),
#                                                class_id, image_id)
#            if class_id == 6:
#                path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),
#                                                           str(class_id))
#                img = cv2.imread(path)
#                img = cv2.resize(img, (1000, 1000))
#                img = mask_image(img)
#                multi_polygon = mtp.mask_to_polygons(img, class_id)
#                mtp.create_csv_submission_inner(multi_polygon,(1000, 1000),
#                                                class_id, image_id)
#            if class_id == 7:
#                path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),
#                                                           str(class_id))
#                img = cv2.imread(path)
#                img = cv2.resize(img, (1000, 1000))
#                img = mask_image(img)
#                multi_polygon = mtp.mask_to_polygons(img, class_id)
#                mtp.create_csv_submission_inner(multi_polygon,(1000, 1000),
#                                                class_id, image_id)
#            if class_id == 8:
#                path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),
#                                                           str(class_id))
#                img = cv2.imread(path)
#                img = cv2.resize(img, (1000, 1000))
#                img = mask_image(img)
#                multi_polygon = mtp.mask_to_polygons(img, class_id)
#                mtp.create_csv_submission_inner(multi_polygon,(1000, 1000),
#                                                class_id, image_id)
#            if class_id == 9:
#                path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),
#                                                           str(class_id))
#                img = cv2.imread(path)
#                img = mask_image(img)
#                multi_polygon = mtp.mask_to_polygons(img, class_id)
#                mtp.create_csv_submission_inner(multi_polygon,(2000, 2000),
#                                                class_id, image_id)
#            if class_id == 10:
#                path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),
#                                                           str(class_id))
#                img = cv2.imread(path)
#                img = mask_image(img)
#                multi_polygon = mtp.mask_to_polygons(img, class_id)
#                mtp.create_csv_submission_inner(multi_polygon,(2000, 2000),
#                                                class_id, image_id)
#            img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#            #print('Image: ', img.shape)
#            del img
#            img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, kernel)
#            del img_close
#            img_open = cv2.cvtColor(img_open, cv2.COLOR_BGR2GRAY)
#            #img_open = img_close
#            #img_open = cv2.cvtColor(img_open, cv2.COLOR_BGR2GRAY)
#            #print(img_open)
#            img_open[img_open<16] = 0
#            img_open[img_open>16] = 1
#            multi_polygon = mtp.mask_to_polygons(img_open, class_id)
#            #print(img_open)
#            del img_open
#            mtp.create_csv_submission_inner(multi_polygon,(2000, 2000),class_id, image_id)
#            del multi_polygon

def numpy_to_mask(class_id, image_id):
    
    img_array = np.load('unetMasks/10_{0}.npy'.format(image_id))
#    print("img_array.shape=",img_array.shape)
#    print("img_array.dtype",img_array.dtype)
#
#    plt.figure(class_id)
#    plt.suptitle("Class "+ str(class_id+1) +" Mask")
    #plt.imshow(img_array[class_id], cmap='gray')
    path = '849X836UnetMasks/{0}_{1}.png'.format(str(image_id),
                                                          str(class_id + 1))

    #plt.imsave(
    plt.imsave(path, img_array[class_id])

def get_all_unet_masks():
    all_test_images = d_util.get_all_test_images_official()
    for i in [0,1,2,3,4,5,6,7,8,9]:
        print('i', i)
        for image_id in all_test_images:
            print('image_id', image_id)
            numpy_to_mask(i, image_id)
            
def combine_mask_rule_class_1(image_id):
    """
    get masks as imgs.
    resize and make int 1, 0.
    do logic
    save in folder.
    """
    class_id = 1
    path = '849X836UnetMasks/{0}_{1}.png'.format(str(image_id),str(class_id))
    unet = cv2.imread(path)#.astype(int)
    #print('unet', unet)
    unetmask = mask_image(unet).astype(int)

    path = '100X100Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_100 = cv2.imread(path)
    mask_100 = cv2.resize(img_100, (unet.shape[1],unet.shape[0]))
    mask_100 = mask_image(mask_100).astype(int)

    path = '250X250Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_250 = cv2.imread(path)
    mask_250 = cv2.resize(img_250, (unet.shape[1],unet.shape[0]))
    mask_250 = mask_image(mask_250).astype(int)

    path = '500X500Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_500 = cv2.imread(path)
    mask_500 = cv2.resize(img_500, (unet.shape[1],unet.shape[0]))
    mask_500 = mask_image(mask_500).astype(int)

    path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_2000 = cv2.imread(path)
    mask_2000 = cv2.resize(img_2000, (unet.shape[1],unet.shape[0]))
    mask_2000 = mask_image(mask_2000).astype(int)

    mask_sum = unetmask# + mask_100 + mask_250 + mask_500 + mask_2000
#    mask_sum[mask_sum<11] = 0
#    mask_sum[mask_sum>10] = 1
#849X836combinedMasks
    path = '849X836combinedMasks/{0}_{1}.png'.format(str(image_id),
                                                          str(class_id))

    #plt.imsave(
    plt.imsave(path, mask_sum*255)
    #plt.imshow(mask_sum*255)
    #print('mask_sum', mask_sum)
#    print('unetmask.shape', unetmask.shape)
#    print('mask_100.shape', mask_100.shape)
#    for i in range(0, 800):
#        for j in range(1, 800):
#            if mask_sum[i][j]>0:
#                print('mask_sum as mask', mask_sum[i][j])

def combine_mask_rule_class_2_3(image_id,class_id):
    """
    get masks as imgs.
    resize and make int 1, 0.
    do logic
    save in folder.
    """
    path = '849X836UnetMasks/{0}_{1}.png'.format(str(image_id),str(class_id))
    unet = cv2.imread(path)#.astype(int)
    #print('unet', unet)
    unetmask = mask_image(unet).astype(int)

    path = '100X100Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_100 = cv2.imread(path)
    mask_100 = cv2.resize(img_100, (unet.shape[1],unet.shape[0]))
    mask_100 = mask_image(mask_100).astype(int)

    path = '250X250Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_250 = cv2.imread(path)
    mask_250 = cv2.resize(img_250, (unet.shape[1],unet.shape[0]))
    mask_250 = mask_image(mask_250).astype(int)

    path = '500X500Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_500 = cv2.imread(path)
    mask_500 = cv2.resize(img_500, (unet.shape[1],unet.shape[0]))
    mask_500 = mask_image(mask_500).astype(int)

    path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_2000 = cv2.imread(path)
    mask_2000 = cv2.resize(img_2000, (unet.shape[1],unet.shape[0]))
    mask_2000 = mask_image(mask_2000).astype(int)

    mask_sum = unetmask * 3 + mask_100 + mask_250 + mask_500 + mask_2000
    mask_sum[mask_sum<=3] = 0
    mask_sum[mask_sum>2] = 1
#849X836combinedMasks
    path = '849X836combinedMasks/{0}_{1}.png'.format(str(image_id),
                                                          str(class_id))

    #plt.imsave(
    plt.imsave(path, mask_sum*255)
    #plt.imshow(mask_sum*255)
    #print('mask_sum', mask_sum)
#    print('unetmask.shape', unetmask.shape)
#    print('mask_100.shape', mask_100.shape)
#    for i in range(0, 800):
#        for j in range(1, 800):
#            if mask_sum[i][j]>0:
#                print('mask_sum as mask', mask_sum[i][j])
def combine_mask_rule_class_4(image_id,class_id):
    """
    get masks as imgs.
    resize and make int 1, 0.
    do logic
    save in folder.
    """
    path = '849X836UnetMasks/{0}_{1}.png'.format(str(image_id),str(class_id))
    unet = cv2.imread(path)#.astype(int)
    #print('unet', unet)
    unetmask = mask_image(unet).astype(int)

    path = '100X100Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_100 = cv2.imread(path)
    mask_100 = cv2.resize(img_100, (unet.shape[1],unet.shape[0]))
    mask_100 = mask_image(mask_100).astype(int)

    path = '250X250Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_250 = cv2.imread(path)
    mask_250 = cv2.resize(img_250, (unet.shape[1],unet.shape[0]))
    mask_250 = mask_image(mask_250).astype(int)

    path = '500X500Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_500 = cv2.imread(path)
    mask_500 = cv2.resize(img_500, (unet.shape[1],unet.shape[0]))
    mask_500 = mask_image(mask_500).astype(int)

    path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_2000 = cv2.imread(path)
    mask_2000 = cv2.resize(img_2000, (unet.shape[1],unet.shape[0]))
    mask_2000 = mask_image(mask_2000).astype(int)

    mask_sum = unetmask * 3 + mask_100 + mask_250 + mask_500 + mask_2000
    mask_sum[mask_sum<=3] = 0
    mask_sum[mask_sum>2] = 1
#849X836combinedMasks
    path = '849X836combinedMasks/{0}_{1}.png'.format(str(image_id),
                                                          str(class_id))

    #plt.imsave(
    plt.imsave(path, mask_sum*255)

def combine_mask_rule_class_5_6_7_8(image_id,class_id):
    """
    get masks as imgs.
    resize and make int 1, 0.
    do logic
    save in folder.
    """
    path = '849X836UnetMasks/{0}_{1}.png'.format(str(image_id),str(class_id))
    unet = cv2.imread(path)#.astype(int)
    #print('unet', unet)
    unetmask = mask_image(unet).astype(int)

    path = '100X100Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_100 = cv2.imread(path)
    mask_100 = cv2.resize(img_100, (unet.shape[1],unet.shape[0]))
    mask_100 = mask_image(mask_100).astype(int)

    path = '250X250Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_250 = cv2.imread(path)
    mask_250 = cv2.resize(img_250, (unet.shape[1],unet.shape[0]))
    mask_250 = mask_image(mask_250).astype(int)

    path = '500X500Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_500 = cv2.imread(path)
    mask_500 = cv2.resize(img_500, (unet.shape[1],unet.shape[0]))
    mask_500 = mask_image(mask_500).astype(int)

    path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_2000 = cv2.imread(path)
    mask_2000 = cv2.resize(img_2000, (unet.shape[1],unet.shape[0]))
    mask_2000 = mask_image(mask_2000).astype(int)

    mask_sum = unetmask * 2 + mask_100 + mask_250 + mask_500 + mask_2000
    mask_sum[mask_sum<=3] = 0
    mask_sum[mask_sum>2] = 1
#849X836combinedMasks
    path = '849X836combinedMasks/{0}_{1}.png'.format(str(image_id),
                                                          str(class_id))

    #plt.imsave(
    plt.imsave(path, mask_sum*255)
    #plt.imshow(mask_sum*255)
    #print('mask_sum', mask_sum)
#    print('unetmask.shape', unetmask.shape)
#    print('mask_100.shape', mask_100.shape)
#    for i in range(0, 800):
#        for j in range(1, 800):
#            if mask_sum[i][j]>0:
#                print('mask_sum as mask', mask_sum[i][j])

def combine_mask_rule_class_9_10(image_id,class_id):
    """
    get masks as imgs.
    resize and make int 1, 0.
    do logic
    save in folder.
    """
    path = '849X836UnetMasks/{0}_{1}.png'.format(str(image_id),str(class_id))
    unet = cv2.imread(path)#.astype(int)
    #print('unet', unet)
    unetmask = mask_image(unet).astype(int)

#    path = '100X100Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
#    img_100 = cv2.imread(path)
#    mask_100 = cv2.resize(img_100, (unet.shape[1],unet.shape[0]))
#    mask_100 = mask_image(mask_100).astype(int)
#
#    path = '250X250Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
#    img_250 = cv2.imread(path)
#    mask_250 = cv2.resize(img_250, (unet.shape[1],unet.shape[0]))
#    mask_250 = mask_image(mask_250).astype(int)

    path = '500X500Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_500 = cv2.imread(path)
    mask_500 = cv2.resize(img_500, (unet.shape[1],unet.shape[0]))
    mask_500 = mask_image(mask_500).astype(int)

    path = '2000X2000Masks/{0}_{1}.png'.format(str(image_id),str(class_id))
    img_2000 = cv2.imread(path)
    mask_2000 = cv2.resize(img_2000, (unet.shape[1],unet.shape[0]))
    mask_2000 = mask_image(mask_2000).astype(int)

    mask_sum = unetmask + mask_500 + mask_2000
    #mask_sum[mask_sum<=4] = 0
    mask_sum[mask_sum>.5] = 1
#849X836combinedMasks
    path = '849X836combinedMasks/{0}_{1}.png'.format(str(image_id),
                                                          str(class_id))

    #plt.imsave(
    plt.imsave(path, mask_sum*255)

def save_class9_10_combined_masks():
    all_test_images = d_util.get_all_test_images_official()
    for image_id in all_test_images:
        for class_id in [9,10]:
            print('image_id', image_id)
            combine_mask_rule_class_9_10(image_id, class_id)
#save_class9_10_combined_masks()

def save_class5_6_7_8_combined_masks():
    all_test_images = d_util.get_all_test_images_official()
    for image_id in all_test_images:
        for class_id in [5,6,7,8]:
            print('image_id', image_id)
            combine_mask_rule_class_5_6_7_8(image_id, class_id)
#save_class5_6_7_8_combined_masks()

def save_class4_combined_masks():
    all_test_images = d_util.get_all_test_images_official()
    for image_id in all_test_images:
        for class_id in [4]:
            print('image_id', image_id)
            combine_mask_rule_class_4(image_id, class_id)
#save_class4_combined_masks()

def save_class2_3_combined_masks():
    all_test_images = d_util.get_all_test_images_official()
    for image_id in all_test_images:
        for class_id in [2,3]:
            print('image_id', image_id)
            combine_mask_rule_class_2_3(image_id, class_id)
#save_class2_3_combined_masks()

def save_class1_combined_masks():
    all_test_images = d_util.get_all_test_images_official()
    for image_id in all_test_images:
        print('image_id', image_id)
        combine_mask_rule_class_1(image_id)
#save_class1_combined_masks()

#numpy_to_mask(0, '6010_0_0')
#get_all_unet_masks()
#    plt.savefig('10_6010_0_0'+ str(i+1)+'.png',dpi=300)
mtp.create_csv_initial()
create_submission_from_masks()
##get_all_masks()
mtp.reorder_csv()
#size = '500X500'
#get_all_masks(size)
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

