from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import scipy.io
from matplotlib import pyplot as plt
import random


# from tensorflow.python.client import device_lib
# print (device_lib.list_local_devices())

##################load data#####################

all_data = pickle.load(open('dataset.pickle', 'rb'))
train_data = all_data['train_dataset']
test_data = all_data['test_dataset']

train_labels = all_data['train_labels']
test_labels = all_data['test_labels']

del all_data

#################Load train and test data###################


num_channels = 1  # grayscale
image_width = 20
image_height = 1400

def reformat(dataset):
    dataset = dataset.reshape(
        (-1, image_width, image_height, num_channels)).astype(np.float32)
    return dataset


train_data = reformat(train_data)
test_data = reformat(test_data)


print('train_data shape is : %s' % (train_data.shape,))
print('test_data shape is : %s' % (test_data.shape,))


test_size = test_data.shape[0]
train_size = train_data.shape[0]

########################Training###########################

num_classifiers = 1


def accuracy(predictions, labels):
    batch_size = predictions[0].shape[0]
    sum = np.sum(predictions==labels)
    acc = (100.0 * sum) / batch_size


    return acc

