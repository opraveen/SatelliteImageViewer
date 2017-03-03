#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:33:11 2017

@author: joe
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.wkt import loads
import shapely.geometry
import tifffile as tiff
import os
import cv2
from skimage.segmentation import slic, mark_boundaries
from shapely.wkt import loads as wkt_loads
import glob


IN_DIR = '../data'
GRID_SIZES = pd.read_csv(IN_DIR + '/grid_sizes.csv',
                         names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SAMPLE_SUBMISSION = pd.read_csv(IN_DIR + '/sample_submission.csv',
                         names=['ImageId','ClassType', 'MultipolygonWKT'], skiprows=1)
df = pd.read_csv(IN_DIR + '/train_wkt_v4.csv')
df.head()


def get_all_test_images_official():
    return list(set(SAMPLE_SUBMISSION.ImageId.tolist()))

#print(get_all_test_images_official())

def get_all_test_images():
    '''
    Return list of all image_id's that are not part of training set.
    Grab list of all image_id's and list of all Training image_id's.
    Return the list of not intersected.
    '''
    path = IN_DIR + '/three_band/*.tif'  
    files=glob.glob(path)   
    images = []
    for file in files:
        image_id = str(file).replace('../data/three_band/','')
        image_id = str(image_id).replace('.tif','')
        #print(image_id)
        images.append(image_id)
    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    training_images = list(set(get_images_with_classes(classes)))
    #print('training_images: ', training_images)

    return list(set(images) - set(training_images))
#print(get_all_test_images())

def mask_to_polygons(mask):
    all_polygons=[]
    for shape, value in features.shapes(mask.astype(np.int16),
                                mask = (mask==1),
                                transform = rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):

        all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        #Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        #need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons

def get_xy_factor(img_size, xymax):
    x_max, y_max = xymax
    height, width = img_size
    out_x = x_max / width
    out_y = y_max / height
    return [out_x, out_y]
    
def resize_multi_polygon(multi_polygon, img_size, xymax):
    [x_fact, y_fact] = get_xy_factor(img_size, xymax)
    for polygon in multi_polygon:
        print('polygon:', polygon)
        scaled = affinity.scale(polygon, xfact=1/500, yfact=1/500, origin=(0,0))
        print('scaled: ', scaled)

def _get_xmax_ymin(grid_sizes_panda, image_id):
    '''
    Return tuple (float, float) for (xmax, ymin)

    '''
    gsp = grid_sizes_panda
    xmax, ymin = gsp[gsp.ImageId == image_id].iloc[0, 1:].astype(float)
    return (xmax, ymin)


def _get_polygon_list(wkt_list_pandas, image_id, class_type):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == image_id]
    multipoly_def = df_image[df_image.ClassType == class_type].MultipolygonWKT
    polygon_list = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygon_list = wkt_loads(multipoly_def.values[0])
    return polygon_list


def _get_and_convert_contours(polygon_list, raster_img_size, xymax):
    perim_list = []
    interior_list = []
    if polygon_list is None:
        return None
    for k, _ in enumerate(polygon_list):
        poly = polygon_list[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for polygon_interior in poly.interiors:
            interior = np.array(list(polygon_interior.coords))
            interior_c = _convert_coordinates_to_raster(interior,
                                                        raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list


def generate_mask(raster_size, image_id, class_type, grid_sizes_panda,
                  wkt_list_pandas):
    '''
    return numpy.ndarray. Serves as labels for predictors.
    get xymax from gridsizes.
    create list of polygons from train_wkt_v4.
    for each polygon in class for image get fill interior.
    fill the interior of a numpy array the size of image with polygons.
    is class == 1
    is not class == 0

    '''
    xymax = _get_xmax_ymin(grid_sizes_panda, image_id)
    polygon_list = _get_polygon_list(wkt_list_pandas, image_id, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    return _plot_mask_from_contours(raster_size, contours, 1)


def get_mask(image_id, class_id, size):
    '''
    return numpy.ndarray. Serves as labels for predictors.
    is class == 1
    is not class == 0

    '''
    wkt_list_pandas = pd.read_csv('../data/train_wkt_v4.csv')
    grid_sizes_panda = pd.read_csv('../data/grid_sizes.csv',
                                   names=['ImageId', 'Xmax', 'Ymin'],
                                   skiprows=1)

    return generate_mask(size, image_id, class_id, grid_sizes_panda,
                         wkt_list_pandas)

def get_ims(image_id, size):
    rgbfile = os.path.join('..', 'data', 'three_band',
                           '{}.tif'.format(image_id))
    rgb = tiff.imread(rgbfile)
    rgb = np.rollaxis(rgb, 0, 3)
    rgb = cv2.resize(rgb, size)

    mfile = os.path.join('..', 'data', 'sixteen_band',
                         '{}_M.tif'.format(image_id))
    img_m = tiff.imread(mfile)
    img_m = np.rollaxis(img_m, 0, 3)
    img_m = cv2.resize(img_m, size)

    afile = os.path.join('..', 'data', 'sixteen_band',
                         '{}_A.tif'.format(image_id))
    img_a = tiff.imread(afile)
    img_a = np.rollaxis(img_a, 0, 3)
    img_a = cv2.resize(img_a, size)
    #ccci_index = get_spectral_analysis(IM_ID)
    return [rgb, img_m, img_a]

def build_19_deep_layer(image_id, size):
    '''
    Return Numpy Array.
    Height and Width determined by input size parameter.
    Depth is 19 Layers
    '''
    [rgb, img_m, img_a] = get_ims(image_id, size)

    stacked_array = np.zeros((rgb.shape[0], rgb.shape[1], 19))
    stacked_array[:, :, 0] = rgb[:, :, 0]
    stacked_array[:, :, 1] = rgb[:, :, 1]
    stacked_array[:, :, 2] = rgb[:, :, 2]
    stacked_array[:, :, 3] = img_m[:, :, 0]
    stacked_array[:, :, 4] = img_m[:, :, 1]
    stacked_array[:, :, 5] = img_m[:, :, 2]
    stacked_array[:, :, 6] = img_m[:, :, 3]
    stacked_array[:, :, 7] = img_m[:, :, 4]
    stacked_array[:, :, 8] = img_m[:, :, 5]
    stacked_array[:, :, 9] = img_m[:, :, 6]
    stacked_array[:, :, 10] = img_m[:, :, 7]
    stacked_array[:, :, 11] = img_a[:, :, 0]
    stacked_array[:, :, 12] = img_a[:, :, 1]
    stacked_array[:, :, 13] = img_a[:, :, 2]
    stacked_array[:, :, 14] = img_a[:, :, 3]
    stacked_array[:, :, 15] = img_a[:, :, 4]
    stacked_array[:, :, 16] = img_a[:, :, 5]
    stacked_array[:, :, 17] = img_a[:, :, 6]
    stacked_array[:, :, 18] = img_a[:, :, 7]
    return stacked_array

def ccci_index(m, rgb):
    RE  = cv2.resize(m[5,:,:], (rgb.shape[0], rgb.shape[1])) 
    MIR = cv2.resize(m[7,:,:], (rgb.shape[0], rgb.shape[1])) 
    R = rgb[:,:,0]
    # canopy chloropyll content index
    return (MIR-RE)/(MIR+RE)*(MIR-R)/(MIR+R)

#def stretch_8bit(bands, lower_percent=2, higher_percent=98):
#    out = np.zeros_like(bands).astype(np.float32)
#    for i in range(3):
#        a = 0 
#        b = 1 
#        c = np.percentile(bands[:,:,i], lower_percent)
#        d = np.percentile(bands[:,:,i], higher_percent)        
#        t = a + (bands[:,:,i] - c) * (b - a) / (d - c)    
#        t[t<a] = a
#        t[t>b] = b
#        out[:,:,i] =t
#    return out.astype(np.float32)



def get_spectral_analysis(IM_ID):
    # read rgb and m bands
    rgb = tiff.imread('../data/three_band/{}.tif'.format(IM_ID))
    rgb = np.rollaxis(rgb, 0, 3)
    m = tiff.imread('../data/sixteen_band/{}_M.tif'.format(IM_ID))
    
    # get our index
    return ccci_index(m, rgb) 
    
    # you can look on histogram and pick your favorite threshold value(0.11 is my best)
    #binary = (CCCI > 0.11).astype(np.float32)
    
def build_9_deep_layer_class_1(image_id, size):
    '''
    Return Numpy Array.
    Height and Width determined by input size parameter.
    Depth is 9 Layers
    most important Features:
    1. feature 3 (0.269457)
    2. feature 10 (0.175592)
    3. feature 11 (0.068455)
    4. feature 4 (0.063344)
    5. feature 9 (0.050757)
    6. feature 6 (0.044132)
    7. feature 7 (0.042846)
    8. feature 0 (0.039941)
    9. feature 2 (0.039165)
    10. feature 8 (0.031848) remove
    11. feature 5 (0.027490)remove
    12. feature 16 (0.027265)remove
    13. feature 1 (0.024750)remove
    14. feature 14 (0.019526)remove
    15. feature 18 (0.018274)remove
    16. feature 13 (0.018085)remove
    17. feature 17 (0.013249)remove
    18. feature 12 (0.013172)remove
    19. feature 15 (0.012651)remove
    '''
    [rgb, img_m, img_a] = get_ims(image_id, size)

    stacked_array = np.zeros((rgb.shape[0], rgb.shape[1], 9))
    stacked_array[:, :, 0] = rgb[:, :, 0]
    #stacked_array[:, :, 1] = rgb[:, :, 1]
    stacked_array[:, :, 1] = rgb[:, :, 2]
    stacked_array[:, :, 2] = img_m[:, :, 0]
    stacked_array[:, :, 3] = img_m[:, :, 1]
    #stacked_array[:, :, 5] = img_m[:, :, 2]
    stacked_array[:, :, 4] = img_m[:, :, 3]
    stacked_array[:, :, 5] = img_m[:, :, 4]
    #stacked_array[:, :, 8] = img_m[:, :, 5]
    stacked_array[:, :, 6] = img_m[:, :, 6]
    stacked_array[:, :, 7] = img_m[:, :, 7]
    stacked_array[:, :, 8] = img_a[:, :, 0]
    #stacked_array[:, :, 12] = img_a[:, :, 1]
    #stacked_array[:, :, 13] = img_a[:, :, 2]
    #stacked_array[:, :, 14] = img_a[:, :, 3]
    #stacked_array[:, :, 15] = img_a[:, :, 4]
    #stacked_array[:, :, 16] = img_a[:, :, 5]
    #stacked_array[:, :, 17] = img_a[:, :, 6]
    #stacked_array[:, :, 18] = img_a[:, :, 7]
    return stacked_array

def build_9_deep_layer_class_2(image_id, size):
    '''
    Return Numpy Array.
    Height and Width determined by input size parameter.
    Depth is 9 Layers
    most important Features:
    1. feature 10 (0.108337)
    2. feature 9 (0.105984)
    3. feature 18 (0.076958)
    4. feature 0 (0.064350)
    5. feature 1 (0.059704)
    6. feature 11 (0.055858)
    7. feature 3 (0.054730)
    8. feature 7 (0.048406)
    9. feature 2 (0.048255)
    10. feature 4 (0.046878)remove
    11. feature 6 (0.044442)remove
    12. feature 16 (0.044235)remove
    13. feature 5 (0.041137)remove
    14. feature 17 (0.038147)remove
    15. feature 8 (0.037792)remove
    16. feature 15 (0.034890)remove
    17. feature 14 (0.034820)remove
    18. feature 12 (0.028320)remove
    19. feature 13 (0.026759)remove
    '''
    [rgb, img_m, img_a] = get_ims(image_id, size)

    stacked_array = np.zeros((rgb.shape[0], rgb.shape[1], 9))
    stacked_array[:, :, 0] = rgb[:, :, 0]
    stacked_array[:, :, 1] = rgb[:, :, 1]
    stacked_array[:, :, 2] = rgb[:, :, 2]
    stacked_array[:, :, 3] = img_m[:, :, 0]
    #stacked_array[:, :, 4] = img_m[:, :, 1]
    #stacked_array[:, :, 5] = img_m[:, :, 2]
    #stacked_array[:, :, 6] = img_m[:, :, 3]
    stacked_array[:, :, 4] = img_m[:, :, 4]
    #stacked_array[:, :, 8] = img_m[:, :, 5]
    stacked_array[:, :, 5] = img_m[:, :, 6]
    stacked_array[:, :, 6] = img_m[:, :, 7]
    stacked_array[:, :, 7] = img_a[:, :, 0]
    #stacked_array[:, :, 12] = img_a[:, :, 1]
    #stacked_array[:, :, 13] = img_a[:, :, 2]
    #stacked_array[:, :, 14] = img_a[:, :, 3]
    #stacked_array[:, :, 15] = img_a[:, :, 4]
    #stacked_array[:, :, 16] = img_a[:, :, 5]
    #stacked_array[:, :, 17] = img_a[:, :, 6]
    stacked_array[:, :, 8] = img_a[:, :, 7]
    return stacked_array

def build_9_deep_layer_class_3(image_id, size):
    '''
    Return Numpy Array.
    Height and Width determined by input size parameter.
    Depth is 9 Layers
    Feature ranking:
    1. feature 10 (0.166690)
    2. feature 6 (0.088873)
    3. feature 0 (0.085471)
    4. feature 9 (0.072393)
    5. feature 3 (0.068775)
    6. feature 2 (0.063699)
    7. feature 11 (0.054214)
    8. feature 4 (0.048185)
    9. feature 16 (0.047797)
    10. feature 7 (0.045747)remove
    11. feature 8 (0.041771)remove
    12. feature 1 (0.038706)remove
    13. feature 5 (0.030713)remove
    14. feature 18 (0.030528)remove
    15. feature 14 (0.029737)remove
    16. feature 17 (0.025076)remove
    17. feature 13 (0.023516)remove
    18. feature 15 (0.020040)remove
    19. feature 12 (0.018069)remove
    '''
    [rgb, img_m, img_a] = get_ims(image_id, size)

    stacked_array = np.zeros((rgb.shape[0], rgb.shape[1], 9))
    stacked_array[:, :, 0] = rgb[:, :, 0]
    #stacked_array[:, :, 1] = rgb[:, :, 1]
    stacked_array[:, :, 1] = rgb[:, :, 2]
    stacked_array[:, :, 2] = img_m[:, :, 0]
    stacked_array[:, :, 3] = img_m[:, :, 1]
    #stacked_array[:, :, 5] = img_m[:, :, 2]
    stacked_array[:, :, 4] = img_m[:, :, 3]
    #stacked_array[:, :, 7] = img_m[:, :, 4]
    #stacked_array[:, :, 8] = img_m[:, :, 5]
    stacked_array[:, :, 5] = img_m[:, :, 6]
    stacked_array[:, :, 6] = img_m[:, :, 7]
    stacked_array[:, :, 7] = img_a[:, :, 0]
    #stacked_array[:, :, 12] = img_a[:, :, 1]
    #stacked_array[:, :, 13] = img_a[:, :, 2]
    #stacked_array[:, :, 14] = img_a[:, :, 3]
    #stacked_array[:, :, 15] = img_a[:, :, 4]
    stacked_array[:, :, 8] = img_a[:, :, 5]
    #stacked_array[:, :, 17] = img_a[:, :, 6]
    #stacked_array[:, :, 18] = img_a[:, :, 7]
    return stacked_array

def build_9_deep_layer_class_4(image_id, size):
    '''
    Return Numpy Array.
    Height and Width determined by input size parameter.
    Depth is 9 Layers
    Feature ranking:
    1. feature 9 (0.099009)
    2. feature 11 (0.075683)
    3. feature 0 (0.071631)
    4. feature 6 (0.064858)
    5. feature 8 (0.062835)
    6. feature 4 (0.058600)
    7. feature 18 (0.057539)
    8. feature 3 (0.057299)
    9. feature 10 (0.051589)
    10. feature 2 (0.047413)remove
    11. feature 1 (0.046885)remove
    12. feature 17 (0.045164)remove
    13. feature 7 (0.043489)remove
    14. feature 5 (0.042154)remove
    15. feature 14 (0.038454)remove
    16. feature 12 (0.036900)remove
    17. feature 13 (0.035172)remove
    18. feature 16 (0.033595)remove
    19. feature 15 (0.031731)remove
    '''
    [rgb, img_m, img_a] = get_ims(image_id, size)

    stacked_array = np.zeros((rgb.shape[0], rgb.shape[1], 9))
    stacked_array[:, :, 0] = rgb[:, :, 0]
    #stacked_array[:, :, 1] = rgb[:, :, 1]
    #stacked_array[:, :, 2] = rgb[:, :, 2]
    stacked_array[:, :, 1] = img_m[:, :, 0]
    stacked_array[:, :, 2] = img_m[:, :, 1]
    #stacked_array[:, :, 5] = img_m[:, :, 2]
    stacked_array[:, :, 3] = img_m[:, :, 3]
    #stacked_array[:, :, 7] = img_m[:, :, 4]
    stacked_array[:, :, 4] = img_m[:, :, 5]
    stacked_array[:, :, 5] = img_m[:, :, 6]
    stacked_array[:, :, 6] = img_m[:, :, 7]
    stacked_array[:, :, 7] = img_a[:, :, 0]
    #stacked_array[:, :, 12] = img_a[:, :, 1]
    #stacked_array[:, :, 13] = img_a[:, :, 2]
    #stacked_array[:, :, 14] = img_a[:, :, 3]
    #stacked_array[:, :, 15] = img_a[:, :, 4]
    #stacked_array[:, :, 16] = img_a[:, :, 5]
    #stacked_array[:, :, 17] = img_a[:, :, 6]
    stacked_array[:, :, 8] = img_a[:, :, 7]
    return stacked_array

def build_9_deep_layer_class_5(image_id, size):
    '''
    Return Numpy Array.
    Height and Width determined by input size parameter.
    Depth is 9 Layers
    Feature ranking:
    1. feature 6 (0.126013)
    2. feature 7 (0.081846)
    3. feature 0 (0.079264)
    4. feature 11 (0.074018)
    5. feature 9 (0.073754)
    6. feature 5 (0.068345)
    7. feature 1 (0.065472)
    8. feature 3 (0.065141)
    9. feature 4 (0.061818)
    10. feature 2 (0.057855)remove
    11. feature 10 (0.045017)remove
    12. feature 18 (0.039035)remove
    13. feature 8 (0.029005)remove
    14. feature 14 (0.025850)remove
    15. feature 13 (0.024511)remove
    16. feature 12 (0.023831)remove
    17. feature 16 (0.023195)remove
    18. feature 17 (0.018316)remove
    19. feature 15 (0.017714)remove
    '''
    [rgb, img_m, img_a] = get_ims(image_id, size)

    stacked_array = np.zeros((rgb.shape[0], rgb.shape[1], 9))
    stacked_array[:, :, 0] = rgb[:, :, 0]
    stacked_array[:, :, 1] = rgb[:, :, 1]
    #stacked_array[:, :, 2] = rgb[:, :, 2]
    stacked_array[:, :, 2] = img_m[:, :, 0]
    stacked_array[:, :, 3] = img_m[:, :, 1]
    stacked_array[:, :, 4] = img_m[:, :, 2]
    stacked_array[:, :, 5] = img_m[:, :, 3]
    stacked_array[:, :, 6] = img_m[:, :, 4]
    #stacked_array[:, :, 8] = img_m[:, :, 5]
    stacked_array[:, :, 7] = img_m[:, :, 6]
    #stacked_array[:, :, 10] = img_m[:, :, 7]
    stacked_array[:, :, 8] = img_a[:, :, 0]
    #stacked_array[:, :, 12] = img_a[:, :, 1]
    #stacked_array[:, :, 13] = img_a[:, :, 2]
    #stacked_array[:, :, 14] = img_a[:, :, 3]
    #stacked_array[:, :, 15] = img_a[:, :, 4]
    #stacked_array[:, :, 16] = img_a[:, :, 5]
    #stacked_array[:, :, 17] = img_a[:, :, 6]
    #stacked_array[:, :, 18] = img_a[:, :, 7]
    return stacked_array

def build_9_deep_layer_class_6(image_id, size):
    '''
    Return Numpy Array.
    Height and Width determined by input size parameter.
    Depth is 19 Layers
    Feature ranking:
    1. feature 11 (0.173882)
    2. feature 9 (0.076466)
    3. feature 16 (0.071707)
    4. feature 10 (0.067565)
    5. feature 6 (0.062374)
    6. feature 15 (0.060454)
    7. feature 3 (0.059901)
    8. feature 18 (0.057314)
    9. feature 12 (0.043994)
    10. feature 14 (0.037934)remove
    11. feature 7 (0.037176)remove
    12. feature 13 (0.036722)remove
    13. feature 17 (0.035562)remove
    14. feature 4 (0.034169)remove
    15. feature 5 (0.033643)remove
    16. feature 0 (0.032371)remove
    17. feature 8 (0.027910)remove
    18. feature 1 (0.027177)remove
    19. feature 2 (0.023680)remove
    '''
    [rgb, img_m, img_a] = get_ims(image_id, size)

    stacked_array = np.zeros((rgb.shape[0], rgb.shape[1], 9))
    #stacked_array[:, :, 0] = rgb[:, :, 0]
    #stacked_array[:, :, 1] = rgb[:, :, 1]
    #stacked_array[:, :, 2] = rgb[:, :, 2]
    stacked_array[:, :, 0] = img_m[:, :, 0]
    #stacked_array[:, :, 4] = img_m[:, :, 1]
    #stacked_array[:, :, 5] = img_m[:, :, 2]
    stacked_array[:, :, 1] = img_m[:, :, 3]
    #stacked_array[:, :, 7] = img_m[:, :, 4]
    #stacked_array[:, :, 8] = img_m[:, :, 5]
    stacked_array[:, :, 2] = img_m[:, :, 6]
    stacked_array[:, :, 3] = img_m[:, :, 7]
    stacked_array[:, :, 4] = img_a[:, :, 0]
    stacked_array[:, :, 5] = img_a[:, :, 1]
    #stacked_array[:, :, 13] = img_a[:, :, 2]
    #stacked_array[:, :, 14] = img_a[:, :, 3]
    stacked_array[:, :, 6] = img_a[:, :, 4]
    stacked_array[:, :, 7] = img_a[:, :, 5]
    #stacked_array[:, :, 17] = img_a[:, :, 6]
    stacked_array[:, :, 8] = img_a[:, :, 7]
    return stacked_array

def build_9_deep_layer_class_7(image_id, size):
    '''
    Return Numpy Array.
    Height and Width determined by input size parameter.
    Depth is 19 Layers
    Feature ranking:
    1. feature 4 (0.241473)
    2. feature 10 (0.169648)
    3. feature 2 (0.130939)
    4. feature 3 (0.091364)
    5. feature 11 (0.068482)
    6. feature 9 (0.041832)
    7. feature 5 (0.041223)
    8. feature 13 (0.034988)
    9. feature 14 (0.025267)
    10. feature 18 (0.018525)remove
    11. feature 1 (0.017937)remove
    12. feature 15 (0.017327)remove
    13. feature 12 (0.016472)remove
    14. feature 6 (0.016456)remove
    15. feature 8 (0.015975)remove
    16. feature 17 (0.014198)remove
    17. feature 7 (0.013897)remove
    18. feature 16 (0.013255)remove
    19. feature 0 (0.010742)remove
    '''
    [rgb, img_m, img_a] = get_ims(image_id, size)

    stacked_array = np.zeros((rgb.shape[0], rgb.shape[1], 9))
    #stacked_array[:, :, 0] = rgb[:, :, 0]
    #stacked_array[:, :, 1] = rgb[:, :, 1]
    stacked_array[:, :, 0] = rgb[:, :, 2]
    stacked_array[:, :, 1] = img_m[:, :, 0]
    stacked_array[:, :, 2] = img_m[:, :, 1]
    stacked_array[:, :, 3] = img_m[:, :, 2]
    #stacked_array[:, :, 6] = img_m[:, :, 3]
    #stacked_array[:, :, 7] = img_m[:, :, 4]
    #stacked_array[:, :, 8] = img_m[:, :, 5]
    stacked_array[:, :, 4] = img_m[:, :, 6]
    stacked_array[:, :, 5] = img_m[:, :, 7]
    stacked_array[:, :, 6] = img_a[:, :, 0]
    #stacked_array[:, :, 12] = img_a[:, :, 1]
    stacked_array[:, :, 7] = img_a[:, :, 2]
    stacked_array[:, :, 8] = img_a[:, :, 3]
    #stacked_array[:, :, 15] = img_a[:, :, 4]
    #stacked_array[:, :, 16] = img_a[:, :, 5]
    #stacked_array[:, :, 17] = img_a[:, :, 6]
    #stacked_array[:, :, 18] = img_a[:, :, 7]
    return stacked_array

def build_9_deep_layer_class_8(image_id, size):
    '''
    Return Numpy Array.
    Height and Width determined by input size parameter.
    Depth is 19 Layers
    Feature ranking:
    1. feature 10 (0.267770)
    2. feature 9 (0.091547)
    3. feature 4 (0.086734)
    4. feature 2 (0.070864)
    5. feature 7 (0.066686)
    6. feature 3 (0.055117)
    7. feature 14 (0.050949)
    8. feature 11 (0.038394)
    9. feature 8 (0.037389)
    10. feature 12 (0.036403)remove
    11. feature 13 (0.030980)remove
    12. feature 16 (0.030661)remove
    13. feature 0 (0.026833)remove
    14. feature 5 (0.023638)remove
    15. feature 18 (0.022599)remove
    16. feature 1 (0.018675)remove
    17. feature 6 (0.016653)remove
    18. feature 17 (0.015878)remove
    19. feature 15 (0.012230)remove
    '''
    [rgb, img_m, img_a] = get_ims(image_id, size)

    stacked_array = np.zeros((rgb.shape[0], rgb.shape[1], 9))
    #stacked_array[:, :, 0] = rgb[:, :, 0]
    #stacked_array[:, :, 1] = rgb[:, :, 1]
    stacked_array[:, :, 0] = rgb[:, :, 2]
    stacked_array[:, :, 1] = img_m[:, :, 0]
    stacked_array[:, :, 2] = img_m[:, :, 1]
    #stacked_array[:, :, 5] = img_m[:, :, 2]
    #stacked_array[:, :, 6] = img_m[:, :, 3]
    stacked_array[:, :, 3] = img_m[:, :, 4]
    stacked_array[:, :, 4] = img_m[:, :, 5]
    stacked_array[:, :, 5] = img_m[:, :, 6]
    stacked_array[:, :, 6] = img_m[:, :, 7]
    stacked_array[:, :, 7] = img_a[:, :, 0]
    #stacked_array[:, :, 12] = img_a[:, :, 1]
    #stacked_array[:, :, 13] = img_a[:, :, 2]
    stacked_array[:, :, 8] = img_a[:, :, 3]
    #stacked_array[:, :, 15] = img_a[:, :, 4]
    #stacked_array[:, :, 16] = img_a[:, :, 5]
    #stacked_array[:, :, 17] = img_a[:, :, 6]
    #stacked_array[:, :, 18] = img_a[:, :, 7]
    return stacked_array
def build_9_deep_layer_class_9(image_id, size):
    '''
    Return Numpy Array.
    Height and Width determined by input size parameter.
    Depth is 19 Layers
    Feature ranking:
    1. feature 10 (0.198984)
    2. feature 11 (0.120227)
    3. feature 3 (0.088417)
    4. feature 18 (0.057698)
    5. feature 12 (0.051582)
    6. feature 13 (0.048079)
    7. feature 6 (0.046951)
    8. feature 17 (0.044557)
    9. feature 7 (0.044444)
    10. feature 14 (0.041786)remove
    11. feature 16 (0.041302)remove
    12. feature 9 (0.037124)remove
    13. feature 4 (0.035811)remove
    14. feature 15 (0.033760)remove
    15. feature 8 (0.027676)remove
    16. feature 2 (0.025126)remove
    17. feature 0 (0.023295)remove
    18. feature 1 (0.017317)remove
    19. feature 5 (0.015864)remove
    '''
    [rgb, img_m, img_a] = get_ims(image_id, size)

    stacked_array = np.zeros((rgb.shape[0], rgb.shape[1], 9))
    #stacked_array[:, :, 0] = rgb[:, :, 0]
    #stacked_array[:, :, 1] = rgb[:, :, 1]
    #stacked_array[:, :, 2] = rgb[:, :, 2]
    stacked_array[:, :, 0] = img_m[:, :, 0]
    #stacked_array[:, :, 4] = img_m[:, :, 1]
    #stacked_array[:, :, 5] = img_m[:, :, 2]
    stacked_array[:, :, 1] = img_m[:, :, 3]
    stacked_array[:, :, 2] = img_m[:, :, 4]
    #stacked_array[:, :, 8] = img_m[:, :, 5]
    #stacked_array[:, :, 9] = img_m[:, :, 6]
    stacked_array[:, :, 3] = img_m[:, :, 7]
    stacked_array[:, :, 4] = img_a[:, :, 0]
    stacked_array[:, :, 5] = img_a[:, :, 1]
    stacked_array[:, :, 6] = img_a[:, :, 2]
    #stacked_array[:, :, 14] = img_a[:, :, 3]
    #stacked_array[:, :, 15] = img_a[:, :, 4]
    #stacked_array[:, :, 16] = img_a[:, :, 5]
    stacked_array[:, :, 7] = img_a[:, :, 6]
    stacked_array[:, :, 8] = img_a[:, :, 7]
    return stacked_array

def build_9_deep_layer_class_10(image_id, size):
    '''
    Return Numpy Array.
    Height and Width determined by input size parameter.
    Depth is 19 Layers
    1. feature 9 (0.095195)
    2. feature 10 (0.085717)
    3. feature 11 (0.069927)
    4. feature 3 (0.065219)
    5. feature 12 (0.053188)
    6. feature 14 (0.051998)
    7. feature 4 (0.051198)
    8. feature 13 (0.050835)
    9. feature 7 (0.050009)
    10. feature 6 (0.049671)remove
    11. feature 0 (0.046510)remove
    12. feature 18 (0.044545)remove
    13. feature 8 (0.044031)remove
    14. feature 16 (0.042715)remove
    15. feature 15 (0.042282)remove
    16. feature 2 (0.040344)remove
    17. feature 1 (0.039613)remove
    18. feature 5 (0.039079)remove
    19. feature 17 (0.037925)remove
    '''
    [rgb, img_m, img_a] = get_ims(image_id, size)

    stacked_array = np.zeros((rgb.shape[0], rgb.shape[1], 9))
    #stacked_array[:, :, 0] = rgb[:, :, 0]
    #stacked_array[:, :, 1] = rgb[:, :, 1]
    #stacked_array[:, :, 2] = rgb[:, :, 2]
    stacked_array[:, :, 0] = img_m[:, :, 0]
    stacked_array[:, :, 1] = img_m[:, :, 1]
    #stacked_array[:, :, 5] = img_m[:, :, 2]
    #stacked_array[:, :, 6] = img_m[:, :, 3]
    stacked_array[:, :, 2] = img_m[:, :, 4]
    #stacked_array[:, :, 8] = img_m[:, :, 5]
    stacked_array[:, :, 3] = img_m[:, :, 6]
    stacked_array[:, :, 4] = img_m[:, :, 7]
    stacked_array[:, :, 5] = img_a[:, :, 0]
    stacked_array[:, :, 6] = img_a[:, :, 1]
    stacked_array[:, :, 7] = img_a[:, :, 2]
    stacked_array[:, :, 8] = img_a[:, :, 3]
    #stacked_array[:, :, 15] = img_a[:, :, 4]
    #stacked_array[:, :, 16] = img_a[:, :, 5]
    #stacked_array[:, :, 17] = img_a[:, :, 6]
    #stacked_array[:, :, 18] = img_a[:, :, 7]
    return stacked_array
def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask

    
def _convert_coordinates_to_raster(coords, img_size, xymax):
    '''
    Return an xy coordinate mapped to image size.
    xymax info is learned from gridsizes csv file.
    pre width is a transformation given from Kaggle tutorial W′=W⋅W/(W+1)
    pre height is the same.
    coords[:, 0] given from Kaggle tutorial x'=(x/x_max)*W'

    '''
    x_max, y_max = xymax
    height, width = img_size
    pre_width = width * width / (width + 1)
    pre_height = height * height / (height + 1)
    coords[:, 0] *= pre_width / x_max
    coords[:, 1] *= pre_height / y_max
    return np.round(coords).astype(np.int32)

def _convert_coord_to_csv_size(coords, img_size, xymax):
    '''
    Return an xy coordinate mapped to image size.
    xymax info is learned from gridsizes csv file.
    pre width is a transformation given from Kaggle tutorial W′=W⋅W/(W+1)
    pre height is the same.
    coords[:, 0] given from Kaggle tutorial x'=(x/x_max)*W'

    '''
    x_max, y_max = xymax
    coords_split = str(coords).split(' ')
    x_prime = coords_split[0]
    y_prime = coords_split[1]
    height, width = img_size
    out_x = x_prime * (x_max / width)
    out_y = y_prime * (y_max / height)
    return (out_x + ' ' + out_y)

def get_list_image_id():
    """Return list of image ID's"""
    return GRID_SIZES.ImageId


def stretch_8bit(bands, lower_percent=2, higher_percent=90):
    """Return np.uint8 for bands after stretch."""
    out = np.zeros_like(bands)
    for i in range(3):
        a = 0
        b = 255
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.uint8)


def scale_percentile(matrix):
    """Fixes the pixel value range to 2%-98% original distribution of values"""
    orig_shape = matrix.shape
    mat = matrix
    matrix = np.reshape(mat, [mat.shape[0] * mat.shape[1], 3]).astype(float)

    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins

    mat = (mat - mins[None, :]) / maxs[None, :]
    mat = np.reshape(mat, orig_shape)
    mat = mat.clip(0, 1)
    return mat


def overlay_polygons(image_id, classes):
    # Use just first image
    polygonsList = {}
    image = df[df.ImageId == image_id]
    print("classes: ", classes)
    for cType in classes:
        lds = loads(image[image.ClassType == cType].MultipolygonWKT.values[0])
        polygonsList[cType] = lds
    # plot using matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    print('ax', ax)
    # plotting, color by class type
    for p in polygonsList:
        for polygon in polygonsList[p]:
            mpl_poly = Polygon(np.array(polygon.exterior),
                               color=plt.cm.Set1(p*10), lw=0, alpha=0.3)
            ax.add_patch(mpl_poly)

    ax.relim()
    ax.autoscale_view()
    print('ax', ax)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.tight_layout()
    print('ax', ax)
    print('fig', fig)
    fig.savefig('testplot.png', bbox_inches=extent)


def images_with_polygons():
    return df.ImageId.unique()


def get_classes_in_image(image_id):
    # number of objects on the image by type
    '''
    1. Buildings
    2. Misc. Manmade structures
    3. Road
    4. Track - poor/dirt/cart track, footpath/trail
    5. Trees - woodland, hedgerows, groups of trees, standalone trees
    6. Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes,
    turnips) crops
    7. Waterway
    8. Standing water
    9. Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle
    10. Vehicle Small - small vehicle (car, van), motorbike
    '''
    polygonsList = {}
    l_class = []
    image = df[df.ImageId == image_id]
    for cType in image.ClassType.unique():
        lds = loads(image[image.ClassType == cType].MultipolygonWKT.values[0])
        polygonsList[cType] = lds
    for p in polygonsList:
        if len(polygonsList[p].geoms) > 0:
            msg = "Type: {:4d}, obs: {}".format(p, len(polygonsList[p].geoms))
            l_class.append(msg)
    return l_class


def get_images_with_classes(classes):
    # convert to shapely, get geometries and pivot

    df['polygons'] = df.apply(lambda row: loads(row.MultipolygonWKT), axis=1)
    df['nPolygons'] = df.apply(lambda row: len(row['polygons'].geoms), axis=1)

    pvt = df.pivot(index='ImageId', columns='ClassType', values='nPolygons')
    image_ids = []
    for class_ in classes:
        img_ids = pvt.loc[pvt[class_] != 0].index
        image_ids.extend(img_ids)
    return image_ids
# print(get_images_with_classes([1, 2, 3, 4, 5]))


def tiff_img_3_band(image_id):
    """
    Return Image given image_id.
    scoped to three_band data.
    """
    filename = os.path.join('..', 'data',
                            'three_band', '{}.tif'.format(image_id))
    tif_data = tiff.imread(filename).transpose([1, 2, 0])
    fixed_im = scale_percentile(tif_data)
    return fixed_im


def tiff_img_a_band(image_id):
    """
    Return Image given image_id.
    scoped to sixteen_band a region data.
    """
    filename = os.path.join('..', 'data',
                            'sixteen_band', '{}_A.tif'.format(image_id))
    tif_data = tiff.imread(filename).transpose([1, 2, 0])
    return tif_data


def tiff_img_m_band(image_id):
    """
    Return Image given image_id.
    scoped to sixteen_band m region data.
    """
    filename = os.path.join('..', 'data',
                            'sixteen_band', '{}_M.tif'.format(image_id))
    tif_data = tiff.imread(filename).transpose([1, 2, 0])
    return tif_data


def tiff_img_p_band(image_id):
    """
    Return Image given image_id.
    scoped to sixteen_band p data.
    """
    filename = os.path.join('..', 'data',
                            'sixteen_band', '{}_P.tif'.format(image_id))
    tif_data = tiff.imread(filename)
    return tif_data


def get_p_image(image_id):
    """Save p_image to testplot.png given image_id"""
    t_img = tiff_img_p_band(image_id)
    plt.imsave('testplot.png', t_img)


def get_a_image(image_id, layer):
    t_img = tiff_img_a_band(image_id)
    t_img_sh = t_img.shape
    img = np.zeros((t_img_sh[0], t_img_sh[1]))
    img[:, :] = t_img[:, :, layer]
    plt.imsave('testplot.png', img)


def get_a_1_image(image_id):
    """Save a_1_image to testplot.png given image_id"""
    get_a_image(image_id, 0)


def get_a_2_image(image_id):
    """Save a_2_image to testplot.png given image_id"""
    get_a_image(image_id, 1)


def get_a_3_image(image_id):
    """Save a_3_image to testplot.png given image_id"""
    get_a_image(image_id, 2)


def get_a_4_image(image_id):
    """Save a_4_image to testplot.png given image_id"""
    get_a_image(image_id, 3)


def get_a_5_image(image_id):
    """Save a_5_image to testplot.png given image_id"""
    get_a_image(image_id, 4)


def get_a_6_image(image_id):
    """Save a_6_image to testplot.png given image_id"""
    get_a_image(image_id, 5)


def get_a_7_image(image_id):
    """Save a_7_image to testplot.png given image_id"""
    get_a_image(image_id, 6)


def get_a_8_image(image_id):
    """Save a_8_image to testplot.png given image_id"""
    get_a_image(image_id, 7)


def get_m_image(image_id, layer):
    t_img = tiff_img_m_band(image_id)
    t_img_sh = t_img.shape
    img = np.zeros((t_img_sh[0], t_img_sh[1]))
    img[:, :] = t_img[:, :, layer]
    plt.imsave('testplot.png', img)


def get_m_1_image(image_id):
    """Save m_1_image to testplot.png given image_id"""
    get_m_image(image_id, 0)


def get_m_2_image(image_id):
    """Save m_2_image to testplot.png given image_id"""
    get_m_image(image_id, 1)


def get_m_3_image(image_id):
    """Save m_3_image to testplot.png given image_id"""
    get_m_image(image_id, 2)


def get_m_4_image(image_id):
    """Save m_4_image to testplot.png given image_id"""
    get_m_image(image_id, 3)


def get_m_5_image(image_id):
    """Save m_5_image to testplot.png given image_id"""
    get_m_image(image_id, 4)


def get_m_6_image(image_id):
    """Save m_6_image to testplot.png given image_id"""
    get_m_image(image_id, 5)


def get_m_7_image(image_id):
    """Save m_7_image to testplot.png given image_id"""
    get_m_image(image_id, 6)


def get_m_8_image(image_id):
    """Save m_8_image to testplot.png given image_id"""
    get_m_image(image_id, 7)


def get_rbg_image(image_id, layer):
    t_img = tiff_img_3_band(image_id)
    t_img_sh = t_img.shape
    img = np.zeros((t_img_sh[0], t_img_sh[1], 3))
    img[:, :, layer] = t_img[:, :, layer]
    plt.imsave('testplot.png', img)

def get_rbg_image_all(image_id):
    plt.imsave('testplot.png', tiff_img_3_band(image_id))


def get_red_image(image_id):
    """Save red_image to testplot.png given image_id"""
    get_rbg_image(image_id, 0)


def get_green_image(image_id):
    """Save green_image to testplot.png given image_id"""
    get_rbg_image(image_id, 1)


def get_blue_image(image_id):
    """Save blue_image to testplot.png given image_id"""
    get_rbg_image(image_id, 2)


def rbg(image_id):
    filename = os.path.join('..', 'data', 'three_band',
                            '{}.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img


def m_image(image_id):
    filename = os.path.join('..', 'data', 'sixteen_band',
                            '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img


def boundries(image_id):
    y1, y2, x1, x2 = 1000, 1600, 2000, 2600
    rgb = rbg(image_id)
    m_img = m_image(image_id)
    m_img = cv2.resize(m_img, tuple(reversed(rgb.shape[:2])))
    img = np.zeros_like(rgb)
    img[:, :, 0] = m_img[:, :, 6]
    img[:, :, 1] = m_img[:, :, 4]
    img[:, :, 2] = m_img[:, :, 2]
    img = stretch_8bit(img)
    region = img[y1:y2, x1:x2, :]
    segments = slic(region, n_segments=100, compactness=20.0,
                    max_iter=10, sigma=5, spacing=None, multichannel=True,
                    convert2lab=True, enforce_connectivity=False,
                    min_size_factor=10, max_size_factor=3, slic_zero=False)
    boundaries = mark_boundaries(region, segments, color=(0, 255, 0))
    plt.figure()
    plt.imsave('testplot.png', boundaries)
