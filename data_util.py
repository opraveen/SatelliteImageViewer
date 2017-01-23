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

IN_DIR = '../data'
GRID_SIZES = pd.read_csv(IN_DIR + '/grid_sizes.csv',
                         names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
df = pd.read_csv(IN_DIR + '/train_wkt_v4.csv')
df.head()



def _convert_coordinates_to_raster(coords, img_size, xymax):
    x_max, y_max = xymax
    height, width = img_size
    width1 = 1.0 * width * width / (width + 1)
    height1 = 1.0 * height * height / (height + 1)
    x_f = width1 / x_max
    y_f = height1 / y_max
    coords[:, 1] *= x_f
    coords[:, 0] *= y_f
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


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

#overlay_polygons('6070_2_3', [7])
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
