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
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score # leave import for later.
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import cnn_data_util

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Permute
from keras.layers import Convolution2D
from keras.layers import Deconvolution2D
from keras.layers import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Lambda
from keras.optimizers import SGD
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model
from keras.utils.np_utils import to_categorical
# from pylab import *




image_width = 3321
image_height = 3321
num_channels = 19
num_labels = 2

RUN_DIR = '.'

CLASS_1 = {}
CLASS_1[0] = 2692557
CLASS_1[1] = 307443
PARAMETERS_BY_CLASS = {}
PARAMETERS_BY_CLASS[1] = CLASS_1
def create_dataset_cnn(stacked, mask, win_width, win_height):
    '''
    Return Dataset for one image both X and y data as a list.


    '''
    x_and_y = np.zeros(win_width, win_height, 20)
    x_and_y[:, :, 0] = stacked[win_width, win_height, 0]
    x_and_y[:, :, 1] = stacked[win_width, win_height, 1]
    x_and_y[:, :, 2] = stacked[win_width, win_height, 2]
    x_and_y[:, :, 3] = stacked[win_width, win_height, 3]
    x_and_y[:, :, 4] = stacked[win_width, win_height, 4]
    x_and_y[:, :, 5] = stacked[win_width, win_height, 5]
    x_and_y[:, :, 6] = stacked[win_width, win_height, 6]
    x_and_y[:, :, 7] = stacked[win_width, win_height, 7]
    x_and_y[:, :, 8] = stacked[win_width, win_height, 8]
    x_and_y[:, :, 9] = stacked[win_width, win_height, 9]
    x_and_y[:, :, 10] = stacked[win_width, win_height, 10]
    x_and_y[:, :, 11] = stacked[win_width, win_height, 11]
    x_and_y[:, :, 12] = stacked[win_width, win_height, 12]
    x_and_y[:, :, 13] = stacked[win_width, win_height, 13]
    x_and_y[:, :, 14] = stacked[win_width, win_height, 14]
    x_and_y[:, :, 15] = stacked[win_width, win_height, 15]
    x_and_y[:, :, 16] = stacked[win_width, win_height, 16]
    x_and_y[:, :, 17] = stacked[win_width, win_height, 17]
    x_and_y[:, :, 18] = stacked[win_width, win_height, 18]
    x_and_y[:, :, 19] = mask[win_width, win_height] # label layer
    del stacked
    del mask
    y_data = x_and_y[:,:,19]
    x_data = x_and_y[:,:,0:19]
    del x_and_y
    return [x_data, y_data]

def sliding_window_9_9(images, step_size, win_shape, class_id):
    '''slide a window across the image'''
    while True:
        for image_id in images:
            print(image_id)
            size = (image_width, image_height)
            stacked = cnn_data_util.build_19_deep_layer_cnn(image_id, size)
    #        size = (stacked.shape[0], stacked.shape[1])
            mask = cnn_data_util.get_mask(image_id, class_id, size)
    
            for y in range(0, image_width - 9, step_size):
                for x in range(0, image_height - 9, step_size):
                    x_data_l = np.empty((0, 9, 9, 19), float)
                    x_data = stacked[y:y + win_shape[1], x:x + win_shape[0]]
                    y_data = mask[y:y + win_shape[1], x:x + win_shape[0]]
                    x_data_l = np.append(x_data_l,[x_data],axis=0)
                    y_data_l = np.array([round(np.average(y_data))])
                    yield (x_data_l, y_data_l)
                    
def sliding_window_9_9_pred(images, step_size, win_shape):
    '''slide a window across the image'''
    while True:
        for image_id in images:
            print(image_id)
            size = (image_width, image_height)
            stacked = cnn_data_util.build_19_deep_layer_cnn(image_id, size)
    #        size = (stacked.shape[0], stacked.shape[1])

            for y in range(0, image_width - step_size, step_size):
                for x in range(0, image_height - step_size, step_size):
                    x_data_l = np.empty((0, 9, 9, 19), float)
                    x_data = stacked[y:y + win_shape[1], x:x + win_shape[0]]
                    x_data_l = np.append(x_data_l,[x_data],axis=0)
                    yield (x_data_l)
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
    nb_filters = 64
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(img_rows, img_cols, 19)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(20, init='normal', activation='relu'))
    model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, init='normal', activation='softmax'))
    #model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    return model

def train_cnn_auto(class_id, num_test_images, training_epochs):
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



def train_cnn_9_9(training_images, class_id, training_epochs):
    '''
    This is a more honest trainer, but a little more work in doing your prep
    work. That does not mean you need to change the function, but you need to
    pull out you own test data, and provide this function with a list of
    image_ids
    '''
    start = time.time()
    generator = sliding_window_9_9(training_images, 9, [9, 9], class_id)
    model=cnn_model_9_9()
#    print('x_data shape', x_data.shape)
    
    #model.fit(x_data, y_data, validation_split=.25, batch_size=1,nb_epoch=training_epochs)
    model.fit_generator(generator, samples_per_epoch=34040,
                        nb_epoch = training_epochs, verbose=2,
#                        nb_val_samples=200,
                        class_weight=None)
    # serialize weights to HDF5
    model.save(RUN_DIR + '/cnn_model.h5')
    model.save_weights(RUN_DIR + '/cnn_weights.h5')
    print("Saved model to disk")

    print('It took', time.time()-start, 'seconds.')



def evaluate_cnn(image_id, class_id):
    '''
    This is to test your holdout images.
    '''
    x_data = np.empty((0,image_width,image_height,19), float)
    y_data = np.empty((0,image_width*image_height,2), float)
    y_tmp = np.zeros((image_width*image_height,2),int)

    [x_part, y_part] = create_dataset_cnn(image_id, class_id)
    print(np.count_nonzero(y_part))
    print(np.count_nonzero(y_part==0))
    y_part=y_part.astype(int)
    y_part=y_part.reshape(image_width*image_height)
    y_tmp=to_categorical(y_part,2)
    x_data=np.append(x_data,[x_part],axis=0)
    y_data=np.append(y_data,[y_tmp],axis=0)

    model=load_model(RUN_DIR + '/cnn_model.h5')
    print("Loaded model from disk")


    results=model.evaluate(x_data,y_data,batch_size=1)
    return results



def predict_cnn_proba(image_id, class_id):
    '''
    This is to run inference on any of the images.
    '''
    sliding_window_9_9_pred(images, step_size, win_shape)

    model=load_model(RUN_DIR + '/cnn_model.h5')
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print("Loaded model from disk")
    y_data_p = model.predict(x_data,batch_size=1)
    pred_image_1 = np.zeros((image_width*image_height))
    pred_image_1 = y_data_p[:,:,0]
    pred_image = pred_image_1.reshape(image_width,image_height)
#    for row in pred_image:
#        print(row)
#    pred_image[pred_image <.95]=0
#    pred_image[pred_image >.95]=1

    return pred_image * 255


def predict_cnn_9_9(image_id, class_id):
    '''
    This is to run inference on any of the images.
    '''
    start = time.time()
    generator = sliding_window_9_9_pred([image_id], 18, [9, 9])
    
    model=load_model(RUN_DIR + '/cnn_model.h5')
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print("Loaded model from disk")
    y_pred = []
    for i in range(0, 34040):
        pred = model.predict(next(generator))[0][0]
        if i%1000 == 0:
            print(i)
            print('pred: ', pred)
        #print('pred: ', pred)
        y_pred.append(round(pred))
        
    y_pred = np.asarray(y_pred)
    print(y_pred.shape)
    pred_image = y_pred.reshape(123,123)
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


class_id = 1
test_sample_size = 10
training_epochs = 14

#train_cnn_auto(class_id, test_sample_size, training_epochs)

#train_cnn(class_1_ids_train, class_id)
#training_images: ['6110_1_2', '6120_2_0']
#training_images = cnn_data_util.get_images_with_classes([class_id])
#train_cnn_9_9(['6110_1_2', '6120_2_0'], class_id, training_epochs)
#train_cnn_9_9(['6110_1_2','6120_2_2','6060_2_3','6120_2_0','6150_0_3',
#               '6100_1_3','6140_1_2'], class_id, training_epochs)
pred_im = predict_cnn_9_9('6140_1_2', class_id)
plt.imshow(pred_im * 255)
image_id = '6110_1_2'
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