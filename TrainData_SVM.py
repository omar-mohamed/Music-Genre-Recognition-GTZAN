from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
from sklearn.decomposition import PCA
from sklearn.svm import SVC

##################load data#####################

print("Loading data")

all_data = pickle.load(open('dataset_standarized_all_10.pickle', 'rb'))
train_data = all_data['train_dataset']
test_data = all_data['test_dataset']

train_labels = all_data['train_labels']
test_labels = all_data['test_labels']

del all_data

# in case you want to select a portion of the features
# start_index=0
# end_index=87
# train_data=train_data[:,start_index:end_index,:]
# test_data=test_data[:,start_index:end_index,:]


input_width = train_data.shape[1]
input_height = train_data.shape[2]


# vectorize the data
def reformat(dataset):
    dataset = dataset.reshape(
        (-1, input_width * input_height)).astype(np.float32)
    return dataset


train_data = reformat(train_data)
test_data = reformat(test_data)

print("Shape of training set after vectorization:")
print(train_data.shape)

print("Shape of test set after vectorization:")
print(test_data.shape)

################## PCA #####################

print("Running PCA")

# run pca to reduce vector size to 900
pca = PCA(copy=True, iterated_power='auto', n_components=900, random_state=None,
          svd_solver='auto', tol=0.0, whiten=False)

pca.fit(train_data)

train_data = pca.transform(train_data)

test_data = pca.transform(test_data)

print("Shape of training set after pca:")
print(train_data.shape)

print("Shape of test set after pca:")
print(test_data.shape)


################## SVM #####################


# computes accuracy given the predictions and real labels
def accuracy(predictions, labels):
    batch_size = predictions.shape[0]
    sum = np.sum(predictions == labels)
    acc = (100.0 * sum) / batch_size
    return acc


print("Training SVM")

# initializing SVM

clf = SVC(C=0.1, cache_size=200, class_weight=None, coef0=1000.0,
          decision_function_shape=None, degree=2, gamma=0.0001, kernel='linear',
          max_iter=-1, probability=False, random_state=None, shrinking=True,
          tol=0.001, verbose=False)

# training SVM
clf.fit(train_data, train_labels)

# getting predictions of training set
train_predictions = clf.predict(train_data)

print("Train Accuracy: %.1f%%" % accuracy(train_predictions, train_labels))

# getting predictions of test set
test_predictions = clf.predict(test_data)

print("Test Accuracy: %.1f%%" % accuracy(test_predictions, test_labels))
