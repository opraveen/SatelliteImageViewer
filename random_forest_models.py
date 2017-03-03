#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 05:52:55 2017

@author: joe
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
    seep into the model which is technically not good, but also remember the
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
from random import randint
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score # leave import for later.
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import data_util
import mask_to_polygon as mtp
np.random.seed(seed=42)
random.seed(42)
# Tested = True 20 iterations of randomized search.
PARAMETERS_BY_CLASS = {}
CLASS_1 = {}
CLASS_1['n_estimators'] = 100
CLASS_1['max_depth'] = None
CLASS_1['min_samples_split'] = 5
CLASS_1['min_samples_leaf'] = 2
CLASS_1['max_features'] = 9
CLASS_1['criterion'] = 'entropy'
CLASS_1['bootstrap'] = True
PARAMETERS_BY_CLASS[1] = CLASS_1

# Tested = False
CLASS_2 = {}
CLASS_2['n_estimators'] = 100
CLASS_2['max_depth'] = None
CLASS_2['min_samples_split'] = 5
CLASS_2['min_samples_leaf'] = 2
CLASS_2['max_features'] = 9
CLASS_2['criterion'] = 'entropy'
CLASS_2['bootstrap'] = True
PARAMETERS_BY_CLASS[2] = CLASS_2

# Tested = False
CLASS_3 = {}
CLASS_3['n_estimators'] = 100
CLASS_3['max_depth'] = None
CLASS_3['min_samples_split'] = 5
CLASS_3['min_samples_leaf'] = 2
CLASS_3['max_features'] = 9
CLASS_3['criterion'] = 'entropy'
CLASS_3['bootstrap'] = True
PARAMETERS_BY_CLASS[3] = CLASS_3

# Tested = False
CLASS_4 = {}
CLASS_4['n_estimators'] = 100
CLASS_4['max_depth'] = None
CLASS_4['min_samples_split'] = 5
CLASS_4['min_samples_leaf'] = 2
CLASS_4['max_features'] = 9
CLASS_4['criterion'] = 'entropy'
CLASS_4['bootstrap'] = True
PARAMETERS_BY_CLASS[4] = CLASS_4

# Tested = False
CLASS_5 = {}
CLASS_5['n_estimators'] = 100
CLASS_5['max_depth'] = None
CLASS_5['min_samples_split'] = 5
CLASS_5['min_samples_leaf'] = 2
CLASS_5['max_features'] = 9
CLASS_5['criterion'] = 'entropy'
CLASS_5['bootstrap'] = True
PARAMETERS_BY_CLASS[5] = CLASS_5

# Tested = False
CLASS_6 = {}
CLASS_6['n_estimators'] = 100
CLASS_6['max_depth'] = None
CLASS_6['min_samples_split'] = 5
CLASS_6['min_samples_leaf'] = 2
CLASS_6['max_features'] = 9
CLASS_6['criterion'] = 'entropy'
CLASS_6['bootstrap'] = True
PARAMETERS_BY_CLASS[6] = CLASS_6

# Tested = False
CLASS_7 = {}
CLASS_7['n_estimators'] = 100
CLASS_7['max_depth'] = None
CLASS_7['min_samples_split'] = 5
CLASS_7['min_samples_leaf'] = 2
CLASS_7['max_features'] = 9
CLASS_7['criterion'] = 'entropy'
CLASS_7['bootstrap'] = True
PARAMETERS_BY_CLASS[7] = CLASS_7

# Tested = False
CLASS_8 = {}
CLASS_8['n_estimators'] = 100
CLASS_8['max_depth'] = None
CLASS_8['min_samples_split'] = 5
CLASS_8['min_samples_leaf'] = 2
CLASS_8['max_features'] = 9
CLASS_8['criterion'] = 'entropy'
CLASS_8['bootstrap'] = True
PARAMETERS_BY_CLASS[8] = CLASS_8

# Tested = False
CLASS_9 = {}
CLASS_9['n_estimators'] = 100
CLASS_9['max_depth'] = None
CLASS_9['min_samples_split'] = 5
CLASS_9['min_samples_leaf'] = 2
CLASS_9['max_features'] = 9
CLASS_9['criterion'] = 'entropy'
CLASS_9['bootstrap'] = True
PARAMETERS_BY_CLASS[9] = CLASS_9

# Tested = False
CLASS_10 = {}
CLASS_10['n_estimators'] = 100
CLASS_10['max_depth'] = None
CLASS_10['min_samples_split'] = 5
CLASS_10['min_samples_leaf'] = 2
CLASS_10['max_features'] = 9
CLASS_10['criterion'] = 'entropy'
CLASS_10['bootstrap'] = True
PARAMETERS_BY_CLASS[10] = CLASS_10

# Rescaled height and width
#HEIGHT = 3350
#WIDTH = 3350
#HEIGHT = 1000
#WIDTH = 1000
HEIGHT = 2000
WIDTH = 2000
#HEIGHT = 500
#WIDTH = 500

CLASS_1_ONES = 307443
CLASS_1_ZEROS = 2692557
def get_stack_by_class(image_id, class_id, size):
    if class_id ==1:
        return data_util.build_9_deep_layer_class_1(image_id, size)
    if class_id ==2:
        return data_util.build_9_deep_layer_class_2(image_id, size)
    if class_id ==3:
        return data_util.build_9_deep_layer_class_3(image_id, size)
    if class_id ==4:
        return data_util.build_9_deep_layer_class_4(image_id, size)  
    if class_id ==5:
        return data_util.build_9_deep_layer_class_5(image_id, size)
    if class_id ==6:
        return data_util.build_9_deep_layer_class_6(image_id, size)     
    if class_id ==7:
        return data_util.build_9_deep_layer_class_7(image_id, size)    
    if class_id ==8:
        return data_util.build_9_deep_layer_class_8(image_id, size)
    if class_id ==9:
        return data_util.build_9_deep_layer_class_9(image_id, size)
    if class_id ==10:
        return data_util.build_9_deep_layer_class_10(image_id, size)

def create_dataset(image_id, class_id):
    '''
    Return Dataset for one image both X and y data as a list.


    '''
    size = (HEIGHT, WIDTH)
    stacked = get_stack_by_class(image_id, class_id, size)
    size = (stacked.shape[0], stacked.shape[1])
    print('size', size)
    mask = data_util.get_mask(image_id, class_id, size)
    x_and_y = np.zeros((stacked.shape[0], stacked.shape[1], 10))
    x_and_y[:, :, 0] = stacked[:, :, 0]
    x_and_y[:, :, 1] = stacked[:, :, 1]
    x_and_y[:, :, 2] = stacked[:, :, 2]
    x_and_y[:, :, 3] = stacked[:, :, 3]
    x_and_y[:, :, 4] = stacked[:, :, 4]
    x_and_y[:, :, 5] = stacked[:, :, 5]
    x_and_y[:, :, 6] = stacked[:, :, 6]
    x_and_y[:, :, 7] = stacked[:, :, 7]
    x_and_y[:, :, 8] = stacked[:, :, 8]
#    x_and_y[:, :, 9] = stacked[:, :, 9]
#    x_and_y[:, :, 10] = stacked[:, :, 10]
#    x_and_y[:, :, 11] = stacked[:, :, 11]
#    x_and_y[:, :, 12] = stacked[:, :, 12]
#    x_and_y[:, :, 13] = stacked[:, :, 13]
#    x_and_y[:, :, 14] = stacked[:, :, 14]
#    x_and_y[:, :, 15] = stacked[:, :, 15]
#    x_and_y[:, :, 16] = stacked[:, :, 16]
#    x_and_y[:, :, 17] = stacked[:, :, 17]
#    x_and_y[:, :, 18] = stacked[:, :, 18]
    x_and_y[:, :, 9] = mask[:, :]

    x_and_y_re = x_and_y.reshape(x_and_y.shape[0] * x_and_y.shape[1], 10)
    y_data = x_and_y_re[:, 9]
    x_data = x_and_y_re[:, 0:9]
    return [x_data.tolist(), y_data.tolist()]


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


def train_random_forest_auto(class_id, num_test_images):
    '''
    this is a quick train sequence. makes some assumptions, but if you just
    want to train up a class this is the function for you.
    '''
    images = data_util.get_images_with_classes([class_id])
    split = random_split(images, num_test_images)
    print('images:', images)
    test_images = split[0]
    print('test_images:', test_images)
    training_images = split[1]
    print('training_images:', training_images)
    train_random_forest(training_images, class_id)
    print('These are your test images.')
    for image in test_images:
        print(image)
        

def train_random_forest(training_images, class_id):
    '''
    This is a more honest trainer, but a little more work in doing your prep
    work. That does not mean you need to change the function, but you need to
    pull out you own test data, and provide this function with a list of
    image_ids
    '''
    x_data = []
    y_data = []

    for image_id in training_images:
        [x_part, y_part] = create_dataset(image_id, class_id)
        for i in range(0, len(y_part)):
            evenup_data(x_data, y_data, x_part[i], y_part[i], class_id)
        del x_part
        del y_part
    
    num_ones = [y for y in y_data if y==1]
    num_zeros = [y for y in y_data if y==0]
    print('num_ones: ', len(num_ones))
    print('num_zeros: ', len(num_zeros))

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    param = PARAMETERS_BY_CLASS[class_id]
    model = RandomForestClassifier(n_estimators=param['n_estimators'],
                                   max_depth=param['max_depth'],
                                   min_samples_split=param['min_samples_split'],
                                   min_samples_leaf=param['min_samples_leaf'],
                                   max_features=param['max_features'],
                                   criterion=param['criterion'],
                                   bootstrap=param['bootstrap'],
                                   random_state=0,
                                   n_jobs=-1)

    print('About to start Training')
    start = time.time()
    model.fit(x_data, y_data)
    joblib.dump(model, 'models/rf_class_' + str(class_id) + '.pkl')
    print('It took', time.time()-start, 'seconds.')

    
def evenup_data(x_data, y_data, x_datum, y_datum, class_id):
    if class_id == 1 and y_datum == 0:
        if random.random() < 0.083:#random.random() < 0.0083:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 1 and y_datum == 1:
        if random.random() < 0.83:#random.random() < 0.082:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 2 and y_datum == 0:
        if random.random() < 0.054:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 2 and y_datum == 1:
        if True:# random.random() < 0.26:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 3 and y_datum == 0:
        if random.random() < 0.083:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 3 and y_datum == 1:
        if True:# random.random() < 0.43:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 4 and y_datum == 0:
        if random.random() < 0.04:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 4 and y_datum == 1:
        if True:# random.random() < 0.14:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 5 and y_datum == 0:
        if random.random() < 0.042:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 5 and y_datum == 1:
        if random.random() < 0.27:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 6 and y_datum == 0:
        if random.random() < 0.13:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 6 and y_datum == 1:
        if random.random() < 0.15:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 7 and y_datum == 0:
        if True:#random.random() < 0.048:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 7 and y_datum == 1:
        if True:#random.random() < 0.65:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True


    if class_id == 8 and y_datum == 0:
        if random.random() < 0.069:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 8 and y_datum == 1:
        if True:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True
#            if random.random() < 0.32:
#                return True



    if class_id == 9 and y_datum == 0:
        if random.random() < 0.11:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 9 and y_datum == 1:
        if True:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True
#            if random.random() < 0.12:
#                return True


    if class_id == 10 and y_datum == 0:
        if random.random() < 0.069:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True

    if class_id == 10 and y_datum == 1:
        if True:
            x_data.append(gen_noise_x_data(x_datum))
            y_data.append(y_datum)
        return True
#            if random.random() < 0.082:
#                return True

            
def gen_noise_x_data(x_data):
    #x_data = np.random.normal(x_data, 2)
    return x_data
    
def determine_one_zero_distribution(class_id):
    training_images = data_util.get_images_with_classes([class_id])
    num_ones = 0
    num_zeros = 0
    for image_id in training_images:
        print (image_id)
        [x_part, y_part] = create_dataset(image_id, class_id)
        num_ones += len([y for y in y_part if y==1])
        num_zeros += len([y for y in y_part if y==0])
        print('num_ones: ', num_ones)
        print('num_zeros: ', num_zeros)
    print('final num_ones: ', num_ones)
    print('final num_zeros: ', num_zeros)

    
def predict_random_forest(image_id, class_id):
    '''
    This is to test you holdout images.
    '''
    [x_data, y_data] = create_dataset(image_id, class_id)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    model = joblib.load('models/rf_class_' + str(class_id) + '.pkl')
    start2 = time.time()
    y_pred = model.predict(x_data)
    print('prediction took', time.time() - start2, 'seconds.')

    # Print the feature ranking
    print("about to measure TP, FP, TN, FN.")
    print(perf_measure(y_data, y_pred))
    print("Feature ranking:")

    # Print importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for feature in range(x_data.shape[1]):
        print("%d. feature %d (%f)" % (feature + 1, indices[feature],
                                       importances[indices[feature]]))

    # Show Image
    pred_image = y_pred.reshape(HEIGHT, WIDTH)
    plt.imshow(pred_image * 255)


def predict(image_id, class_id, model):
    '''
    This is to run inference on any of the images.
    '''
    size = (HEIGHT, WIDTH)
    stacked = get_stack_by_class(image_id, class_id, size)
    x_data = stacked.reshape(stacked.shape[0] * stacked.shape[1], 9)
    y_pred = model.predict(x_data)
    pred_image = y_pred.reshape(HEIGHT, WIDTH)
    #plt.imshow(pred_image * 255)
    #plt.imsave('testplot.png',pred_image * 255)
    return pred_image


def predict_proba(image_id, class_id):
    '''
    This is to run inference on any of the images.
    '''
    size = (HEIGHT, WIDTH)
    stacked = get_stack_by_class(image_id, class_id, size)
    x_data = stacked.reshape(stacked.shape[0] * stacked.shape[1], 9)
    print(x_data.shape)
    model = joblib.load('models/rf_class_' + str(class_id) + '.pkl')
    y_pred = model.predict_proba(x_data)
    print(y_pred[:, 1])
    pred_image = y_pred[:, 1].reshape(HEIGHT, WIDTH)
    plt.imshow(pred_image * 255)
    plt.imsave('testplot.png', pred_image * 255)
    return pred_image

PARAM_DIST = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


def random_search_for_hyperpara(training_images, class_id):
    '''
    This is a methdo to search for good hyperparameters.
    '''
    x_data = []
    y_data = []
    for image_id in training_images:
        [x_part, y_part] = create_dataset(image_id, class_id)
        x_data.extend(x_part)
        y_data.extend(y_part)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    x_train, _, y_train, _ = train_test_split(x_data, y_data,
                                              test_size=0.96,
                                              random_state=42)
    n_iter_search = 20
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    random_search = RandomizedSearchCV(clf, param_distributions=PARAM_DIST,
                                       n_iter=n_iter_search)
    start = time.time()
    random_search.fit(x_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))
    report(random_search.cv_results_)


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
    pass
    conf_mat = confusion_matrix(y_actual, y_pred)
    true_neg = conf_mat[0][0]
    false_neg = conf_mat[1][0]
    true_pos = conf_mat[1][1]
    false_pos = conf_mat[0][1]
    print(true_pos, '\t', false_pos, '\t', true_neg, '\t', false_neg)
    return(true_pos, false_pos, true_neg, false_neg)



def retrain_all():
    for i in range(1, 10):
        train_random_forest_auto(i, 0)

#train_random_forest_auto(class_id, num_test_images)
#determine_one_zero_distribution(10)
#train_random_forest_auto(9, 0)

class_id = 10
test_sample_size = 0
#images = data_util.get_images_with_classes([class_id])
#random_search_for_hyperpara(images, class_id)
#train_random_forest(class_1_ids_train, class_id)
train_random_forest_auto(class_id, test_sample_size)
#
#6110_4_0
#6120_2_0
#6070_2_3
#image_id = '6060_2_4'
#img = predict_random_forest(image_id, class_id)
#plt.imshow(img*255)
#predict_proba(image_id, class_id)

#mtp.count_rows_in_submission()

#mtp.reorder_csv()
#
#mtp.create_csv_initial()
#all_test_images = data_util.get_all_test_images_official()
cnt = 0
#for class_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
#    print(class_id)
#    model = joblib.load('models/rf_class_' + str(class_id) + '.pkl')
#    for image_id in all_test_images:
#        cnt += 1
#        print(cnt)
#        print(image_id)
#        img = predict(image_id, class_id, model)
#        multi_polygon = mtp.mask_to_polygons(img)
#        del img
#        mtp.create_csv_submission_inner(multi_polygon,(HEIGHT, WIDTH),class_id, image_id)
#        del multi_polygon
#



      
#retrain_all()
#print('done predicting')