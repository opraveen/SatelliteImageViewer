#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 06:27:22 2017

@author: joe
"""

import data_util as d_util

image_id = '6110_1_2'
classes = [1]
d_util.overlay_polygons(image_id, classes)