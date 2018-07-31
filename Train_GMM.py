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
from sklearn.decomposition import PCA
from sklearn.mixture import GMM
from sklearn.svm import LinearSVC, SVC

all_data = pickle.load(open('dataset_normalized_all.pickle', 'rb'))
train_data = all_data['train_dataset']
test_data = all_data['test_dataset']

train_labels = all_data['train_labels']
test_labels = all_data['test_labels']

del all_data


start_index=0
end_index=87
train_data=train_data[:,start_index:end_index,:]
test_data=test_data[:,start_index:end_index,:]


num_channels = 1  # grayscale
image_width = train_data.shape[1]
image_height = train_data.shape[2]

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



pca=PCA(copy=True, iterated_power='auto', n_components=750, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)

pca.fit(train_data)

train_data=pca.transform(train_data)

test_data=pca.transform(test_data)

n_classes = 10

# Try GMMs using different types of covariances.
classifiers = dict((covar_type, GMM(n_components=n_classes,
                    covariance_type=covar_type, init_params='wc', n_iter=20))
                   for covar_type in ['full'])


def accuracy(predictions, labels):
    batch_size = predictions.shape[0]
    sum = np.sum(predictions == labels)
    acc = (100.0 * sum) / batch_size
    return acc

for index, (name, classifier) in enumerate(classifiers.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    classifier.means_ = np.array([train_data[train_labels == i].mean(axis=0)
                                  for i in range(n_classes)])


    # Train the other parameters using the EM algorithm.
    classifier.fit(train_data)



    train_predictions = classifier.predict(train_data)

    print("Train Accuracy: %.1f%%" % accuracy(train_predictions, train_labels))

    test_predictions = classifier.predict(test_data)

    print("Test Accuracy: %.1f%%" % accuracy(test_predictions, test_labels))