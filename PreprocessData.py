#%matplotlib inline

from __future__ import print_function
import numpy as np

from six.moves import cPickle as pickle
from six.moves import range

import os
import sys
import tarfile
from IPython.display import display, Image

import h5py

import matplotlib.pyplot as plt

from PIL import Image
import random
import numpy as np
import librosa
import librosa.display
#http://opihi.cs.uvic.ca/sound/genres.tar.gz
#GTZAN genre collection.

y,sr=librosa.load('./genres/blues/blues.00001.au')

librosa.display.waveplot(y,sr)

D=librosa.stft(y)

log_power=librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.max)

librosa.display.specshow(log_power,x_axis='time',y_axis='log')
plt.colorbar()

mfcc=librosa.feature.mfcc(y=y,sr=sr)

print(mfcc)

def read_data(directory='./genres'):
    all_data_mfcc=np.zeros((1000,20,1400),dtype=float)
    all_labels=np.zeros(1000)
    label_index=0
    image_index=0
    for _, dirs,_ in os.walk(directory):
        for dir in dirs:
            for _, _, files in os.walk(directory+'/'+dir):
                for file in files:
                    y, sr = librosa.load(directory+'/'+dir+'/'+file)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr)

                    all_data_mfcc[image_index,:,0:mfcc.shape[1]]=mfcc
                    all_labels[image_index]=label_index
                    image_index=image_index+1
            label_index=label_index+1
    return all_data_mfcc,all_labels

all_mfcc,all_labels=read_data()

def normalize(x):
    xvar, xavg = x.var(axis=0), x.mean(axis=0)
    print(xavg)
    x = (x - xavg)
    xvar[xvar==0]=1
    x=x/xvar
    print("After normalization:")
    xmin, xmax = x.min(), x.max()
    print(x)
    return x


print(all_mfcc.shape)

all_mfcc=normalize(all_mfcc)

print(all_mfcc.shape)


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

all_mfcc,all_labels=randomize(all_mfcc,all_labels)


def split_data(dataset,labels,num_classes=10,test_images_for_class=25):
    test_counter=np.zeros(num_classes)
    test_set=np.zeros((num_classes*test_images_for_class,20,1400),dtype=float)
    test_labels=np.zeros(num_classes*test_images_for_class)
    deleted_index=[]
    test_index=0
    for i in range(labels.shape[0]):
        _class=int(labels[i])
        if test_counter[_class]>=test_images_for_class:
            continue
        test_counter[_class]=test_counter[_class]+1
        deleted_index.append(i)
        test_labels[test_index]=labels[i]
        test_set[test_index] = dataset[i]
        test_index=test_index+1
    dataset=np.delete(dataset, deleted_index, axis=0)
    labels=np.delete(labels, deleted_index, axis=0)

    return dataset,labels,test_set,test_labels

train_set,train_labels,test_set,test_labels=split_data(all_mfcc,all_labels)


pickle_file = 'dataset_normalized.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_set,
        'train_labels': train_labels,
        'test_dataset': test_set,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print("Done")
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

