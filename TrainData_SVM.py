from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from sklearn import svm
from six.moves import cPickle as pickle
import scipy.io
from matplotlib import pyplot as plt
import random


# from tensorflow.python.client import device_lib
# print (device_lib.list_local_devices())

##################load data#####################
from sklearn.svm import LinearSVC, SVC

all_data = pickle.load(open('dataset_normalized_mfcc_specto_beats.pickle', 'rb'))
train_data = all_data['train_dataset']
test_data = all_data['test_dataset']

train_labels = all_data['train_labels']
test_labels = all_data['test_labels']

del all_data

#################Load train and test data###################


num_channels = 1  # grayscale
image_width = 41
image_height = 1400

def reformat(dataset):
    dataset = dataset.reshape(
        (-1, image_width*image_height)).astype(np.float32)
    return dataset



train_data = reformat(train_data)
test_data = reformat(test_data)


print('train_data shape is : %s' % (train_data.shape,))
print('test_data shape is : %s' % (test_data.shape,))


test_size = test_data.shape[0]
train_size = train_data.shape[0]


# clf = svm.LinearSVC()
# clf.fit(train_data, train_labels)
# LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
#      verbose=0)


clf = SVC(C=200, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
clf.fit(train_data, train_labels)





def accuracy(predictions, labels):
    batch_size = predictions.shape[0]
    sum = np.sum(predictions==labels)
    acc = (100.0 * sum) / batch_size
    return acc

train_predictions=clf.predict(train_data)

print("Train Accuracy: %.1f%%"%accuracy(train_predictions,train_labels))

test_predictions=clf.predict(test_data)

print("Test Accuracy: %.1f%%"%accuracy(test_predictions,test_labels))