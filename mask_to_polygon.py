#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 07:16:34 2017
I took the mask_to_polygons from kernel posted on Kaggle:
https://www.kaggle.com/ceperaang/dstl-satellite-imagery-feature-detection/
lb-0-42-ultimate-full-solution-run-on-your-hw

by Sergey Mushinskiy

@author: joe
"""

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
import shapely
from shapely import affinity
import rasterio
from rasterio import features
import data_util as datut
import pandas as pd
import csv
import sys
from collections import defaultdict

csv.field_size_limit(sys.maxsize)

def mask_to_polygons(mask, epsilon=1, min_area=1.):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons
#SUBMISSION_FILE = pd.read_csv('submission_file.csv',
#                         names=['ImageId','ClassType', 'MultipolygonWKT'], skiprows=1)
def mask_to_polygons2(mask, class_id):
    all_polygons=[]
    for shape, value in features.shapes(mask.astype(np.int16),
                                mask = (mask==1),
                                transform = rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        del value
        shape1 = shapely.geometry.shape(shape)
#        del shape
#        if class_id == 1:
#            simp = 15
#        if class_id == 2:
#            simp = 15
#        if class_id == 3:
#            simp = 15
#        if class_id == 4:
#            simp = 15
#        if class_id == 5:
#            simp = 15
#        if class_id == 6:
#            simp = 15
#        if class_id == 7:
#            simp = 15
#        if class_id == 8:
#            simp = 15
#        if class_id == 9:
#            simp = 2
#        if class_id == 10:
#            simp = 2
        shape1 = shape1.simplify(1, preserve_topology=True)
        all_polygons.append(shape1)
        del shape1

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        if all_polygons.type == 'Polygon':
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons

def get_xy_factor(img_size, xymax):
    x_max, y_max = xymax
    height, width = img_size
    out_x = x_max / width
    out_y = y_max / height
    return [out_x, out_y]

def create_csv_initial():
    with open('submission_file.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(['ImageId', 'ClassType', 'MultipolygonWKT'])


def create_csv_submission_inner(multi_polygon, img_size, class_type, image_id):
    grid_sizes_panda = pd.read_csv('../data/grid_sizes.csv',
                                   names=['ImageId', 'Xmax', 'Ymin'],
                                   skiprows=1)
    xymax = datut._get_xmax_ymin(grid_sizes_panda, image_id)
    del grid_sizes_panda
    [x_fact, y_fact] = get_xy_factor(img_size, xymax)
    all_polygons=[]
    for polygon in multi_polygon:
        scaled = affinity.scale(polygon, xfact=x_fact, yfact=y_fact, origin=(0,0))
        del polygon
        all_polygons.append(scaled)
        #print('scaled: ', scaled)
    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        if all_polygons.type == 'Polygon':
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    with open('submission_file.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow([image_id, class_type, all_polygons])
        del all_polygons
    return True
    
def count_rows_in_submission():
    file = open("submission_file.csv")
    numline = len(file.readlines())
    print ("submission_file Length: ", numline)
IN_DIR = '../data'

def reorder_csv():
    with open('submission_file3.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(['ImageId', 'ClassType', 'MultipolygonWKT'])
    #all_test_images = datut.get_all_test_images_official()
    with open(IN_DIR + '/sample_submission.csv', 'rt') as f:
        reader = csv.reader(f)
        csv_list = list(reader)
    
    with open('submission_file.csv', 'rt') as f:
        reader = csv.reader(f)
        csv_list2 = list(reader)

    for row in csv_list[1:]:
        image_id = str(row[0]).strip() 
        print('image_id: ', image_id)
        class_id = str(row[1]).strip() 
        print('class_id: ', class_id)
        with open('submission_file3.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')

            all_p = [z for x, y, z in csv_list2 if x == image_id and y == class_id]
            #print(all_p[0])
#            writer.writerow(get_row(image_id, class_id))
            writer.writerow([image_id,class_id,all_p[0]])

def get_row(image_id, class_id):
    with open('submission_file.csv', 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0]==image_id and row[1]==class_id:
                return row

#def smooth_data(precision):
#    with open('submission_file4.csv', 'w') as csvfile:
#        writer = csv.writer(csvfile, lineterminator='\n')
#        writer.writerow(['ImageId', 'ClassType', 'MultipolygonWKT'])
#
#    with open('submission_file3.csv', 'rt') as f:
#        reader = csv.reader(f)
#        csv_list = list(reader)
#    for row in csv_list[1:]:
#        cnt = 0
#        with open('submission_file4.csv', 'a') as csvfile:
#            writer = csv.writer(csvfile, lineterminator='\n')
#            all_polygons=[]
#            polygons = row[2]
#            for shape in polygons:
#                print(shape)
#                shape1 = shapely.geometry.shape(shape)
#                shape1 = shape1.simplify(precision, preserve_topology=True)
#                all_polygons.append(shape1)
#        
#                all_polygons = shapely.geometry.MultiPolygon(all_polygons)
#                if not all_polygons.is_valid:
#                    all_polygons = all_polygons.buffer(0)
#                    #Sometimes buffer() converts a simple Multipolygon to just a Polygon,
#                    #need to keep it a Multi throughout
#                    if all_polygons.type == 'Polygon':
#                        all_polygons = shapely.geometry.MultiPolygon([all_polygons])
#                        writer = csv.writer(csvfile, lineterminator='\n')
#            writer.writerow([row[0],row[1],all_polygons])
#precision = 1
#smooth_data(precision)   