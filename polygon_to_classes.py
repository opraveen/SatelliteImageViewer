#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 07:56:48 2017
This Scritp Show Traning Polygons on Satalite İmages for each Class
Classes
        'Buildings'        :1,
        'Structures '      :2,
        'Road'             :3,
        'Track'            :4,
        'Trees'            :5,
        'Crops'            :6,
        'Waterway'         :7,
        'StandingWater'    :8,
        'VehicleLarge'     :9,
        'VehicleSmall'     :10,

@author: joe
"""
import os
import cv2
import tifffile
import pandas as pd
import numpy as np
from shapely.wkt import loads as wkt_loads
#from subprocess import check_output
#print(check_output(["ls", "../data"]).decode("utf8"))


#def _convert_coordinates_to_raster(coords, img_size, xymax):
#    '''
#    Return an xy coordinate mapped to image size.
#    xymax info is learned from gridsizes csv file.
#    pre width is a transformation given from Kaggle tutorial W′=W⋅W/(W+1)
#    pre height is the same.
#    coords[:, 0] given from Kaggle tutorial x'=(x/x_max)*W'
#
#    '''
#    x_max, y_max = xymax
#    height, width = img_size
#    pre_width = width * width / (width + 1)
#    pre_height = height * height / (height + 1)
#    coords[:, 0] *= pre_width / x_max
#    coords[:, 1] *= pre_height / y_max
#    return np.round(coords).astype(np.int32)


#def _get_xmax_ymin(grid_sizes_panda, image_id):
#    '''
#    Return tuple (float, float) for (xmax, ymin)
#
#    '''
#    gsp = grid_sizes_panda
#    xmax, ymin = gsp[gsp.ImageId == image_id].iloc[0, 1:].astype(float)
#    return (xmax, ymin)


#def _get_polygon_list(wkt_list_pandas, image_id, class_type):
#    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == image_id]
#    multipoly_def = df_image[df_image.ClassType == class_type].MultipolygonWKT
#    polygon_list = None
#    if len(multipoly_def) > 0:
#        assert len(multipoly_def) == 1
#        polygon_list = wkt_loads(multipoly_def.values[0])
#    return polygon_list


#def _get_and_convert_contours(polygon_list, raster_img_size, xymax):
#    perim_list = []
#    interior_list = []
#    if polygon_list is None:
#        return None
#    for k, _ in enumerate(polygon_list):
#        poly = polygon_list[k]
#        perim = np.array(list(poly.exterior.coords))
#        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
#        perim_list.append(perim_c)
#        for polygon_interior in poly.interiors:
#            interior = np.array(list(polygon_interior.coords))
#            interior_c = _convert_coordinates_to_raster(interior,
#                                                        raster_img_size, xymax)
#            interior_list.append(interior_c)
#    return perim_list, interior_list


#def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
#    img_mask = np.zeros(raster_img_size, np.uint8)
#    if contours is None:
#        return img_mask
#    perim_list, interior_list = contours
#    cv2.fillPoly(img_mask, perim_list, class_value)
#    cv2.fillPoly(img_mask, interior_list, 0)
#    return img_mask
#
#
#def generate_mask(raster_size, image_id, class_type, grid_sizes_panda,
#                  wkt_list_pandas):
#    '''
#    return numpy.ndarray. Serves as labels for predictors.
#    get xymax from gridsizes.
#    create list of polygons from train_wkt_v4.
#    for each polygon in class for image get fill interior.
#    fill the interior of a numpy array the size of image with polygons.
#    is class == 1
#    is not class == 0
#
#    '''
#    xymax = _get_xmax_ymin(grid_sizes_panda, image_id)
#    polygon_list = _get_polygon_list(wkt_list_pandas, image_id, class_type)
#    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
#    return _plot_mask_from_contours(raster_size, contours, 1)
#
#
#def get_mask(image_id, class_id, size):
#    '''
#    return numpy.ndarray. Serves as labels for predictors.
#    is class == 1
#    is not class == 0
#
#    '''
#    wkt_list_pandas = pd.read_csv('../data/train_wkt_v4.csv')
#    grid_sizes_panda = pd.read_csv('../data/grid_sizes.csv',
#                                   names=['ImageId', 'Xmax', 'Ymin'],
#                                   skiprows=1)
#
#    return generate_mask(size, image_id, class_id, grid_sizes_panda,
#                         wkt_list_pandas)
#
#
#def build_19_deep_layer(image_id, size):
#    '''
#    Return Numpy Array.
#    Height and Width determined by input size parameter.
#    Depth is 19 Layers
#    '''
#    rgbfile = os.path.join('..', 'data', 'three_band',
#                           '{}.tif'.format(image_id))
#    rgb = tifffile.imread(rgbfile)
#    rgb = np.rollaxis(rgb, 0, 3)
#    rgb = cv2.resize(rgb, size)
#
#    mfile = os.path.join('..', 'data', 'sixteen_band',
#                         '{}_M.tif'.format(image_id))
#    img_m = tifffile.imread(mfile)
#    img_m = np.rollaxis(img_m, 0, 3)
#    img_m = cv2.resize(img_m, size)
#
#    afile = os.path.join('..', 'data', 'sixteen_band',
#                         '{}_A.tif'.format(image_id))
#    img_a = tifffile.imread(afile)
#    img_a = np.rollaxis(img_a, 0, 3)
#    img_a = cv2.resize(img_a, size)
#
#    stacked_array = np.zeros((rgb.shape[0], rgb.shape[1], 19))
#    stacked_array[:, :, 0] = rgb[:, :, 0]
#    stacked_array[:, :, 1] = rgb[:, :, 1]
#    stacked_array[:, :, 2] = rgb[:, :, 2]
#    stacked_array[:, :, 3] = img_m[:, :, 0]
#    stacked_array[:, :, 4] = img_m[:, :, 1]
#    stacked_array[:, :, 5] = img_m[:, :, 2]
#    stacked_array[:, :, 6] = img_m[:, :, 3]
#    stacked_array[:, :, 7] = img_m[:, :, 4]
#    stacked_array[:, :, 8] = img_m[:, :, 5]
#    stacked_array[:, :, 9] = img_m[:, :, 6]
#    stacked_array[:, :, 10] = img_m[:, :, 7]
#    stacked_array[:, :, 11] = img_a[:, :, 0]
#    stacked_array[:, :, 12] = img_a[:, :, 1]
#    stacked_array[:, :, 13] = img_a[:, :, 2]
#    stacked_array[:, :, 14] = img_a[:, :, 3]
#    stacked_array[:, :, 15] = img_a[:, :, 4]
#    stacked_array[:, :, 16] = img_a[:, :, 5]
#    stacked_array[:, :, 17] = img_a[:, :, 6]
#    stacked_array[:, :, 18] = img_a[:, :, 7]
#    return stacked_array
