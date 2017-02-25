#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 05:52:55 2017

@author: joe (modified for cnn by Asif and Matthias)
1.The first dictionaries are for best hyperparameters as we find them
I built a function called random_search_for_hyperparameters which will find
the best out of 20 random trials. this should get the random forest very close
to optimum.

2. I built 2 functions for training the Random Forest one is to make your life
easier, the other is a little more honest. with random forest and only random
forest I believe the slight dishonest will not hurt you too much.

    a.train_random_forest(training_images, class_id)
    to use this in the honest way. get a random split of images. save those
    images to never be touched except for testing. with the rest find the
    best hyperparameters with random_search_for_hyperparameters function. Then
    test with your testset.

    b. A tool to just Train and Test is
    train_random_forest_auto(class_id, num_test_images). Beware, if you have
    a different set of hold out images for finding the hyperparameters, and
    the Trained model... That is not exactly honest. some test knowledge can
    seep into the model which is techincally not good, but also remeber the
    only real test is the Kaggle test, which we do not have access to except
    through the Submission system.

3. I built a test Function called predict_random_forest. This is for testing
our holdout data. Not the Kaggle data. It is for our testing our model.

4. I built an inference function just called predict. It will run any image
regardless of whether it is our training, test, and will return a Mask.


"""

import time
import random
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cnn_data_util

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Permute
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import ZeroPadding2D
#from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
# from pylab import *


image_width = 3321
image_height = 3321
num_channels = 3
num_labels = 2

RUN_DIR = '.'

CLASS_1 = {}
CLASS_1[0] = 2692557
CLASS_1[1] = 307443
PARAMETERS_BY_CLASS = {}
PARAMETERS_BY_CLASS[1] = CLASS_1



def sliding_window_9_9(images, step_size, win_shape, class_id):
    '''slide a window across the image'''
    while True:
        for image_id in images:
            print(image_id)
            size = (image_width, image_height)
            stacked = cnn_data_util.build_3_deep_layer_cnn(image_id, size)
    #        size = (stacked.shape[0], stacked.shape[1])
            mask = cnn_data_util.get_mask(image_id, class_id, size)
            for y in range(0, image_width - 9, step_size):
                for x in range(0, image_height - 9, step_size):
                    x_data_l = np.empty((0, 9, 9, 3), float)
                    x_data = stacked[y:y + win_shape[1], x:x + win_shape[0]]
                    y_data = mask[y:y + win_shape[1], x:x + win_shape[0]]
                    x_data_l = np.append(x_data_l, [x_data], axis=0)
                    y_data_l = np.array([round(np.average(y_data))])
                    yield (x_data_l, y_data_l)


def sliding_window_27_27(images, step_size, win_shape, class_id):
    '''slide a window across the image'''
    while True:
        for image_id in images:
            print(image_id)
            size = (image_width, image_height)
            stacked = cnn_data_util.build_3_deep_layer_cnn(image_id, size)
            mask = cnn_data_util.get_mask(image_id, class_id, size)
            x_data_l = []# np.empty((0, 27, 27, 3), float)
            y_data_l =  []#np.empty((0, 2), float)
            cnt = 0
            for y in range(0, image_width - 27, step_size):
                for x in range(0, image_height - 27, step_size):
                    x_data = stacked[y:y + win_shape[1], x:x + win_shape[0]]
                    y_data = mask[y:y + win_shape[1], x:x + win_shape[0]]
                    x_data_l.append(x_data)
                    del x_data
                    y_data_p = np.array([round(np.average(y_data))])
                    del y_data
                    y_cat = to_categorical(y_data_p,2)[0]
                    y_data_l.append(y_cat)
                    del y_cat
                    cnt += 1
                    if cnt==2684:
                        ret_x = np.array(x_data_l)
                        ret_y = np.array(y_data_l)
                        x_data_l = []
                        y_data_l =  []
                        cnt = 0
                        yield (ret_x, ret_y)


def sliding_window_81_81(images, step_size, win_shape, class_id):
    '''slide a window across the image'''
    while True:
        for image_id in images:
            print(image_id)
            size = (image_width, image_height)
            stacked = cnn_data_util.build_3_deep_layer_cnn(image_id, size)
            mask = cnn_data_util.get_mask(image_id, class_id, size)
            
            x_data_l = []# np.empty((0, 27, 27, 3), float)
            y_data_l =  []#np.empty((0, 2), float)
            cnt = 0
            for y in range(0, image_width - 81, step_size):
                for x in range(0, image_height - 81, step_size):
                    x_data = stacked[y:y + win_shape[1], x:x + win_shape[0]]
                    y_data = mask[y:y + win_shape[1], x:x + win_shape[0]]
                    x_data_l.append(x_data)
                    del x_data
                    y_val = 0
                    if np.average(y_data)> .05:
                        y_val = 1

                    y_data_p = np.array([y_val])
                    del y_data
                    y_cat = to_categorical(y_data_p,2)[0]
                    y_data_l.append(y_cat)
                    del y_cat
                    cnt += 1
                    if cnt==550:
                        ret_x = np.array(x_data_l)
                        ret_y = np.array(y_data_l)
                        x_data_l = []
                        y_data_l =  []
                        cnt = 0
                        yield (ret_x, ret_y)


def sliding_window_81_81_pred(image_id, step_size, win_shape):
    '''slide a window across the image'''
#    while True:
#        for image_id in images:
    print(image_id)
    size = (image_width, image_height)
    stacked = cnn_data_util.build_3_deep_layer_cnn(image_id, size)
    #        size = (stacked.shape[0], stacked.shape[1])
    x_data_l = []
    for y in range(0, image_width, step_size):
        for x in range(0, image_height, step_size):
            #x_data_l = np.empty((0, 27, 27, 3), float)
            x_data = stacked[y:y + win_shape[1], x:x + win_shape[0]]
            #x_data_l = np.append(x_data_l, [x_data], axis=0)
            x_data_l.append(x_data)
    return np.array(x_data_l)


def sliding_window_27_27_pred(image_id, step_size, win_shape):
    '''slide a window across the image'''
#    while True:
#        for image_id in images:
    print(image_id)
    size = (image_width, image_height)
    stacked = cnn_data_util.build_3_deep_layer_cnn(image_id, size)
    #        size = (stacked.shape[0], stacked.shape[1])
    x_data_l = []
    for y in range(0, image_width, step_size):
        for x in range(0, image_height, step_size):
            #x_data_l = np.empty((0, 27, 27, 3), float)
            x_data = stacked[y:y + win_shape[1], x:x + win_shape[0]]
            #x_data_l = np.append(x_data_l, [x_data], axis=0)
            x_data_l.append(x_data)
    return np.array(x_data_l)


                    
def sliding_window_9_9_pred(images, step_size, win_shape):
    '''slide a window across the image'''
    while True:
        for image_id in images:
            print(image_id)
            size = (image_width, image_height)
            stacked = cnn_data_util.build_3_deep_layer_cnn(image_id, size)
    #        size = (stacked.shape[0], stacked.shape[1])

            for y in range(0, image_width - step_size, step_size):
                for x in range(0, image_height - step_size, step_size):
                    x_data_l = np.empty((0, 9, 9, 3), float)
                    x_data = stacked[y:y + win_shape[1], x:x + win_shape[0]]
                    x_data_l = np.append(x_data_l, [x_data], axis=0)
                    yield x_data_l

#generator = sliding_window_9_9(['6110_1_2', '6120_2_0'], 3, [9,9], 1)
#next(generator)

def random_split(seq, number):
    '''
    Return tuple of 2 lists.
    list[0] is of length number randomly selected from seq.
    list[1] is not in list[0] from seq.

    '''
    ind = set(random.sample(range(len(seq)), number))
    split = ([], [])
    for number, val in enumerate(seq):
        split[number not in ind].append(val)
    return split


def cnn_model_9_9():
    img_rows, img_cols = 9, 9
    # number of convolutional filters to use
    nb_filters = 128
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(img_rows, img_cols, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    
    model.add(BatchNormalization())
    model.add(ZeroPadding2D(padding=(1, 1)))
#    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
#    model.add(Dropout(0.05))

    model.add(Flatten())
#    model.add(Dense(10, init='normal', activation='relu'))
#    model.add(BatchNormalization())
#    #model.add(Activation('relu'))
##    model.add(Dropout(0.5))
#    model.add(Dense(10, init='normal', activation='relu'))
#    model.add(BatchNormalization())
    model.add(Dense(1, init='normal', activation='sigmoid'))
    #model.add(Activation('softmax'))
#    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(loss='binary_crossentropy',optimizer= sgd,metrics=['mean_squared_error'])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['mean_squared_error'])

    return model

def cnn_model_27_27():
    img_rows, img_cols = 27, 27
    # number of convolutional filters to use
    nb_filters = 128
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(img_rows, img_cols, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    #model.add(ZeroPadding2D(padding=(1, 1)))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))  
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    #model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))  
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Flatten())
    model.add(Dense(50, init='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(50, init='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(2, init='normal', activation='softmax'))
    #model.add(Activation('softmax'))
#    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(loss='binary_crossentropy',optimizer= sgd,metrics=['mean_squared_error'])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def cnn_model_81_81():
    img_rows, img_cols = 81, 81
    # number of convolutional filters to use
    nb_filters = 128
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(img_rows, img_cols, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))  
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))  
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))  
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Flatten())
    model.add(Dense(200, init='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(200, init='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(2, init='normal', activation='softmax'))
    #model.add(Activation('softmax'))
#    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(loss='binary_crossentropy',optimizer= sgd,metrics=['mean_squared_error'])
    _metrics = ['accuracy', 'binary_accuracy', 'fmeasure', 'precision', 'recall']
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=_metrics)

    return model

def train_cnn_auto_9_9(class_id, num_test_images, training_epochs):
    '''
    this is a quick train sequence. makes some assumptions, but if you just
    want to train up a class this is the function for you.
    '''
    images = cnn_data_util.get_images_with_classes([class_id])

    split = random_split(images, num_test_images)
    print('images:', images)
    test_images = split[0]
    print('test_images:', test_images)
    training_images = split[1]
    print('training_images:', training_images)
    train_cnn_9_9(training_images, class_id, training_epochs)
    print('These are your test images.')
    for image in test_images:
        print(image)


def train_cnn_auto_27_27(class_id, num_test_images, training_epochs):
    '''
    this is a quick train sequence. makes some assumptions, but if you just
    want to train up a class this is the function for you.
    '''
    images = cnn_data_util.get_images_with_classes([class_id])

    split = random_split(images, num_test_images)
    print('images:', images)
    test_images = split[0]
    print('test_images:', test_images)
    training_images = split[1]
    print('training_images:', training_images)
    train_cnn_27_27(training_images, class_id, training_epochs)
    print('These are your test images.')
    for image in test_images:
        print(image)
    



def train_cnn_auto_81_81(class_id, num_test_images, training_epochs):
    '''
    this is a quick train sequence. makes some assumptions, but if you just
    want to train up a class this is the function for you.
    '''
    images = cnn_data_util.get_images_with_classes([class_id])

    split = random_split(images, num_test_images)
    print('images:', images)
    test_images = split[0]
    print('test_images:', test_images)
    training_images = split[1]
    print('training_images:', training_images)
    train_cnn_81_81(training_images, class_id, training_epochs)
    print('These are your test images.')
    for image in test_images:
        print(image)

def train_cnn_9_9(training_images, class_id, training_epochs):
    '''
    This is a more honest trainer, but a little more work in doing your prep
    work. That does not mean you need to change the function, but you need to
    pull out you own test data, and provide this function with a list of
    image_ids
    '''
    start = time.time()
    generator = sliding_window_9_9(training_images, 9, [9, 9], class_id)
    model = cnn_model_9_9()
    model.fit_generator(generator, samples_per_epoch=136161,
                        nb_epoch=training_epochs, verbose=2,
#                        nb_val_samples=200,
                        class_weight=None)
    model_json = model.to_json()
    with open('models/cnn_9_9_class_' + str(class_id) + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    path = 'models/cnn_3_9_9_class_' + str(class_id) + 'cnn_weights.h5'
    model.save_weights(path)
    print("Saved model to disk")
    print('It took', time.time()-start, 'seconds.')


def train_cnn_27_27(training_images, class_id, training_epochs):
    '''
    This is a more honest trainer, but a little more work in doing your prep
    work. That does not mean you need to change the function, but you need to
    pull out you own test data, and provide this function with a list of
    image_ids
    '''
    start = time.time()
    generator = sliding_window_27_27(training_images, 27, [27, 27], class_id)
    model = cnn_model_27_27()
    #uncomment load_weights to resume training. Model must not have changed. duh
    #model.load_weights('models/cnn_27_27_class_' + str(class_id) + 'cnn_weights.h5')
    class_weight_d = {0:1, 1:50}
    model.fit_generator(generator, samples_per_epoch=2684,
                        nb_epoch=training_epochs, verbose=2,
#                        nb_val_samples=200,
                        class_weight=class_weight_d)
    model_json = model.to_json()
    with open('models/cnn_3_27_27_class_' + str(class_id) + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    path = 'models/cnn_3_27_27_class_' + str(class_id) + 'cnn_weights.h5'
    model.save_weights(path)
    print("Saved model to disk")
    print('It took', time.time()-start, 'seconds.')

def train_cnn_81_81(training_images, class_id, training_epochs):
    '''
    This is a more honest trainer, but a little more work in doing your prep
    work. That does not mean you need to change the function, but you need to
    pull out you own test data, and provide this function with a list of
    image_ids
    '''
    start = time.time()
    generator = sliding_window_81_81(training_images, 81, [81, 81], class_id)
    model = cnn_model_81_81()
    #uncomment load_weights to resume training. Model must not have changed. duh
    #model.load_weights('models/cnn_3_81_81_class_' + str(class_id) + 'cnn_weights.h5')
    class_weight_d = {0:1, 1:42}
    model.fit_generator(generator, samples_per_epoch=550,
                        nb_epoch=training_epochs, verbose=2,
#                        nb_val_samples=200,
                        class_weight=class_weight_d)
    model_json = model.to_json()
    with open('models/cnn_3_81_81_class_' + str(class_id) + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    path = 'models/cnn_3_81_81_class_' + str(class_id) + 'cnn_weights.h5'
    model.save_weights(path)
    print("Saved model to disk")
    print('It took', time.time()-start, 'seconds.')
    
def evaluate_cnn(image_id, class_id):
    '''
    This is to test your holdout images.
    '''
    pass


def predict_cnn_proba(image_id, class_id):
    '''
    This is to run inference on any of the images probablities(psuedo).
    '''
    pass


def predict_cnn_9_9(image_id, class_id):
    '''
    This is to run inference on any of the images.
    '''
    start = time.time()
    generator = sliding_window_9_9_pred([image_id], 9, [9, 9])
    json_file = open('models/cnn_3_9_9_class_' + str(class_id) + '.json', 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    json_file.close()
    loaded_model.load_weights('models/cnn_3_9_9_class_' + str(class_id) + 'cnn_weights.h5')
    print("Loaded model from disk")
    y_pred = []
    mask = cnn_data_util.get_mask(image_id, class_id, (369,369))
    mask = mask.reshape(136161)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(0, 136161):
        pred = loaded_model.predict(next(generator))[0][0]
        if round(pred) == 0 and mask[i] == 0:
            TN +=1
        if round(pred) == 1 and mask[i] == 1:
            TP +=1
        if round(pred) == 1 and mask[i] == 0:
            FP +=1
        if round(pred) == 0 and mask[i] == 1:
            FN +=1
        y_pred.append(round(pred))
    y_pred = np.asarray(y_pred)
    print('TP', TP)
    print('FP', FP)
    print('TN', TN)
    print('FN', FN)
    print(y_pred.shape)
    pred_image = y_pred.reshape(369,369)
    print('It took', time.time()-start, 'seconds.')
    return pred_image

def predict_cnn_27_27(image_id, class_id):
    '''
    This is to run inference on any of the images.
    '''
    start = time.time()
    x_data = sliding_window_27_27_pred(image_id, 27, [27, 27])
    json_file = open('models/cnn_3_27_27_class_' + str(class_id) + '.json', 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    json_file.close()
    loaded_model.load_weights('models/cnn_3_27_27_class_' + str(class_id) + 'cnn_weights.h5')
    print("Loaded model from disk")
    y_pred = []
    mask = cnn_data_util.get_mask(image_id, class_id, (123,123))
    mask = mask.reshape(15129)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    pred_i = loaded_model.predict(x_data)
    for i in range(0, 15129):
        #print('pred_i: ', pred_i)
        pred = pred_i[i][0] #-.35
        #print('pred: ', pred)
        if round(pred) == 0 and mask[i] == 0:
            TN +=1
        if round(pred) == 1 and mask[i] == 1:
            TP +=1
        if round(pred) == 1 and mask[i] == 0:
            FP +=1
        if round(pred) == 0 and mask[i] == 1:
            FN +=1
        #y_pred.append(round(pred))
        y_pred.append(pred)
    y_pred = np.asarray(y_pred)
    print('TP', TP)
    print('FP', FP)
    print('TN', TN)
    print('FN', FN)
    print(y_pred.shape)
    pred_image = y_pred.reshape(123,123)
    print('It took', time.time()-start, 'seconds.')
    return pred_image


def predict_cnn_81_81(image_id, class_id):
    '''
    This is to run inference on any of the images.
    '''
    start = time.time()
    x_data = sliding_window_81_81_pred(image_id, 81, [81, 81])
    json_file = open('models/cnn_3_81_81_class_' + str(class_id) + '.json', 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    json_file.close()
    loaded_model.load_weights('models/cnn_3_81_81_class_' + str(class_id) + 'cnn_weights.h5')
    print("Loaded model from disk")
    y_pred = []
    mask = cnn_data_util.get_mask(image_id, class_id, (41,41))
    mask = mask.reshape(1681)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    pred_i = loaded_model.predict_classes(x_data)
    #pred_i = loaded_model.predict(x_data)
    for i in range(0, 1681):
        #print('pred_i: ', pred_i)
        pred = pred_i[i]#[1]
        #print('pred: ', pred)
        if round(pred) == 0 and mask[i] == 0:
            TN +=1
        if round(pred) == 1 and mask[i] == 1:
            TP +=1
        if round(pred) == 1 and mask[i] == 0:
            FP +=1
        if round(pred) == 0 and mask[i] == 1:
            FN +=1
        #y_pred.append(round(pred))
        y_pred.append(pred)
    y_pred = np.asarray(y_pred)
    print('TP', TP)
    print('FP', FP)
    print('TN', TN)
    print('FN', FN)
    print(y_pred.shape)
    pred_image = y_pred.reshape(41,41)
    print('It took', time.time()-start, 'seconds.')
    return pred_image


def report(results, n_top=3):
    """
    Utility function to report best scores
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def perf_measure(y_actual, y_pred):
    '''
    return confusion matrix parameters (TP, FP, TN, FN)
    '''
    conf_mat = confusion_matrix(y_actual, y_pred)
    true_neg = conf_mat[0][0]
    false_neg = conf_mat[1][0]
    true_pos = conf_mat[1][1]
    false_pos = conf_mat[0][1]
    print(true_pos, '\t', false_pos, '\t', true_neg, '\t', false_neg)
    return(true_pos, false_pos, true_neg, false_neg)

def retrain_all():
    for i in range(1,10):
        train_random_forest_auto(i, 0)


class_id = 3
test_sample_size = 2
training_epochs = 400
#image_id = '6110_1_2'
#size = (3500,3500)
#stacked = cnn_data_util.build_3_deep_layer_cnn(image_id, size)
#stacked += 1.2
#mask = cnn_data_util.get_mask(image_id, class_id, size)
#plt.imshow(stacked * 255)
train_cnn_auto_81_81(class_id, test_sample_size, training_epochs)

#train_cnn(class_1_ids_train, class_id)
#training_images: ['6110_1_2', '6120_2_0']
#training_images = cnn_data_util.get_images_with_classes([class_id])
#train_cnn_9_9(['6110_1_2', '6120_2_0'], class_id, training_epochs)
#train_cnn_9_9(['6110_1_2','6120_2_2','6060_2_3','6120_2_0','6150_0_3',
#               '6100_1_3','6140_1_2'], class_id, training_epochs)
pred_im = predict_cnn_81_81('6120_2_0', class_id)
##pred_im = predict_cnn_27_27('6100_2_2', class_id)
plt.imsave('pred_im_{0}'.format('6120_2_0'), pred_im * 255)
plt.imshow(pred_im * 255)
#mask = cnn_data_util.get_mask('6100_1_3', class_id, (123, 123))
#plt.imshow(mask * 255)
#6100_1_3
#697s - loss: 0.3051 - mean_squared_error: 0.0829
#Saved model to disk
#It took 17322.916904449463 seconds.
#These are your test images.
#6070_2_3
#6120_2_2
#6140_3_1





#train_cnn_auto_9_9(class_id, test_sample_size, training_epochs)
#image_id = '6110_1_2'
#image_id = '6120_2_2'
#image_id = '6060_2_3'
#image_id = '6150_0_3'
#image_id = '6100_1_3'
#image_id = '6120_2_2'
#image_id = '6140_1_2'

#pred_im = predict_cnn(image_id, class_id)
##img_w_mask = create_dataset_cnn(image_id, class_id)
#plt.imshow(pred_im * 255)

#pred_im = predict_cnn_proba(image_id, class_id)
#img_w_mask = create_dataset_cnn(image_id, class_id)
#plt.imshow(pred_im * 255)
#plt.show()
#plt.imshow(img_w_mask[1])
#plt.show()
#plt.imshow(img_w_mask[0][:,:,1])
#plt.show()

#evaluate_cnn(image_id, class_id)