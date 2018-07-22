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



genres_labels = 10      # the labels' length for a genres classifier
batch_size = 20         # the number of training images in a single iteration
test_batch_size = 50   # used to calculate test predictions over many iterations to avoid memory issues
patch_size = 5          # convolution filter size
depth1 = 16             # number of filters in first conv layer
depth2 = 32             # number of filters in second conv layer
depth3 = 64             # number of filters in third conv layer
num_hidden1 = 1024      # the size of the unrolled vector after convolution
num_hidden2 = 512       # the size of the hidden neurons in fully connected layer
num_hidden3 = 256       # the size of the hidden neurons in fully connected layer
# regularization_lambda=4e-4


graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_width, image_height, num_channels), name="train_dataset")

    #labels
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, genres_labels), name="train_labels")


    tf_test_dataset = tf.placeholder(tf.float32, shape=(test_batch_size, image_width, image_height, num_channels), name="test_labels")


    #to take one image and classify it (used in gui interface)
    tf_one_input = tf.placeholder(tf.float32, shape=(1, image_width, image_height, num_channels),name='one_input_placeholder')



    def get_conv_weight(name, shape):
        return tf.get_variable(name, shape=shape,
                               initializer=tf.contrib.layers.xavier_initializer_conv2d())


    def get_bias_variable(name,shape):
        return tf.Variable(tf.constant(1.0, shape=shape),name=name)


    def get_fully_connected_weight(name, shape):
        weights = tf.get_variable(name, shape=shape,
                                  initializer=tf.contrib.layers.xavier_initializer())
        return weights


    # Variables.

    conv1_weights = get_conv_weight('conv1_weights', [patch_size, patch_size, num_channels, depth1])
    conv1_biases = get_bias_variable("conv1_bias",[depth1])

    conv2_weights = get_conv_weight('conv2_weights', [patch_size, patch_size, depth1, depth2])
    conv2_biases = get_bias_variable("conv2_bias",[depth2])

    conv3_weights = get_conv_weight('conv3_weights', [patch_size, patch_size, depth2, depth3])
    conv3_biases = get_bias_variable("conv3_bias",[depth3])

    # genre classifier

    hidden1_weights_c1 = get_fully_connected_weight('hidden1_weights', [num_hidden1, num_hidden2])
    hidden1_biases_c1 = get_bias_variable("hidden1_bias",[num_hidden2])

    hidden2_weights_c1 = get_fully_connected_weight('hidden2_weights', [num_hidden2, num_hidden3])
    hidden2_biases_c1 = get_bias_variable("hidden2_bias",[num_hidden3])

    hidden3_weights_c1 = get_fully_connected_weight('hidden3_weights', [num_hidden3, genres_labels])
    hidden3_biases_c1 = get_bias_variable("hidden3_bias",[genres_labels])



    def get_logits(image_vector, hidden1_weights, hidden1_biases, hidden2_weights, hidden2_biases, hidden3_weights,
                   hidden3_biases, keep_dropout_rate=1):
        hidden = tf.nn.relu(tf.matmul(image_vector, hidden1_weights) + hidden1_biases)
        if keep_dropout_rate < 1:
            hidden = tf.nn.dropout(hidden, keep_dropout_rate)
        hidden = tf.nn.relu(tf.matmul(hidden, hidden2_weights) + hidden2_biases)
        if keep_dropout_rate < 1:
            hidden = tf.nn.dropout(hidden, keep_dropout_rate)
        return tf.matmul(hidden, hidden3_weights) + hidden3_biases


    def run_conv_layer(input, conv_weights, conv_biases):
        conv = tf.nn.conv2d(input, conv_weights, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.max_pool(value=conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.local_response_normalization(conv)
        return tf.nn.relu(conv + conv_biases)


    # Model.
    def model(data, keep_dropout_rate=1):
        # first conv block
        hidden = run_conv_layer(data, conv1_weights, conv1_biases)
        # second conv block
        hidden = run_conv_layer(hidden, conv2_weights, conv2_biases)
        # third conv block
        hidden = run_conv_layer(hidden, conv3_weights, conv3_biases)

        #flatten
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

        #  classifier
        logits = get_logits(reshape, hidden1_weights_c1, hidden1_biases_c1, hidden2_weights_c1, hidden2_biases_c1,
                             hidden3_weights_c1, hidden3_biases_c1, keep_dropout_rate)



        return logits


    # Training computation.
    logits = model(tf_train_dataset, 0.7)

    # regularizers=regularization_lambda*(tf.nn.l2_loss(hidden1_weights) + tf.nn.l2_loss(hidden1_biases))+regularization_lambda*(tf.nn.l2_loss(hidden2_weights) + tf.nn.l2_loss(hidden2_biases))+regularization_lambda*(tf.nn.l2_loss(hidden3_weights) + tf.nn.l2_loss(hidden3_biases))

   #sum loss of different classifiers

    loss =  tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))  # +regularizers

    # tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
    # decayed_learning_rate = learning_rate *decay_rate ^ (global_step / decay_steps)

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.001, global_step, 20000, 0.90, staircase=True)  #use learning rate decay
    # Optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      1.0)          #gradient clipping by 1
    optimize = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=global_step)


    # Predictions for the training and test data.


    train_prediction= logits


    test_prediction=model(tf_test_dataset)

    one_prediction=model(tf_one_input)
    one_prediction=tf.identity(one_prediction, name="one_prediction")



