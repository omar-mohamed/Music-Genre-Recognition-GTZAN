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

# rand=np.random.randint(11, size=(1000, 41,1400))

# def normalize_2(x):
#     xvar, xavg = x.var(axis=0), x.mean(axis=0)
#     print(xavg)
#     x = (x - xavg)
#     xvar_0=xvar==0
#     if(xvar_0!=False):
#         xvar[xvar_0]=1
#     x=x/xvar
#     print("After normalization:")
#     xmin, xmax = x.min(), x.max()
#     print(xmin)
#     print(xmax)
#
#     return x
#
# normalize_2(np.array([0,0,0,0,0,0,0,0,-20]))

y,sr=librosa.load('./genres/blues/blues.00001.au')

librosa.display.waveplot(y, sr)

D=librosa.stft(y)
#D_short = librosa.stft(y, hop_length=6400)
log_power=librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.max)

librosa.display.specshow(log_power,x_axis='time',y_axis='log')
plt.colorbar()

mfcc=librosa.feature.mfcc(y=y,sr=sr)
rmse=librosa.feature.rmse(y=y)
cent = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
chroma_stft=librosa.feature.chroma_stft(y=y, sr=sr)
chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
Stft = np.abs(librosa.stft(y))
contrast = librosa.feature.spectral_contrast(S=Stft, sr=sr)
print(mfcc)

def read_data(directory='./genres'):
    all_data_mfcc=np.zeros((1000,20,1400),dtype=float)
    all_data_spectrogram=np.zeros((1000,20,1400),dtype=float)
    all_data_beats=np.zeros((1000,1,1400),dtype=float)
    all_data_contrast = np.zeros((1000,7, 1400), dtype=float)
    all_data_cq = np.zeros((1000,12, 1400), dtype=float)
    all_data_cens = np.zeros((1000,12, 1400), dtype=float)
    all_data_stft = np.zeros((1000,12, 1400), dtype=float)
    all_data_centroid = np.zeros((1000,1, 1400), dtype=float)
    all_data_bandwidth = np.zeros((1000,1, 1400), dtype=float)
    all_data_rmse = np.zeros((1000,1, 1400), dtype=float)

    all_labels=np.zeros(1000)
    label_index=0
    image_index=0
    for _, dirs,_ in os.walk(directory):
        for dir in dirs:
            for _, _, files in os.walk(directory+'/'+dir):
                for file in files:
                    y, sr = librosa.load(directory+'/'+dir+'/'+file)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr)
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20, fmax=1400)
                    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

                    rmse = librosa.feature.rmse(y=y)
                    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
                    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
                    Stft = np.abs(librosa.stft(y))
                    contrast = librosa.feature.spectral_contrast(S=Stft, sr=sr)
                    #
                    all_data_spectrogram[image_index,:,0:S.shape[1]]=S
                    all_data_mfcc[image_index,:,0:mfcc.shape[1]]=mfcc
                    all_data_beats[image_index,:,0:beats.shape[0]]=beats
                    all_data_beats[image_index, :, beats.shape[0]+1] = tempo
                    #
                    all_data_rmse[image_index,:,0:rmse.shape[1]]=rmse
                    all_data_centroid[image_index,:,0:cent.shape[1]]=cent
                    all_data_bandwidth[image_index,:,0:spec_bw.shape[1]]=spec_bw
                    all_data_stft[image_index,:,0:chroma_stft.shape[1]]=chroma_stft
                    all_data_cens[image_index,:,0:chroma_cens.shape[1]]=chroma_cens
                    all_data_cq[image_index,:,0:chroma_cq.shape[1]]=chroma_cq
                    all_data_contrast[image_index,:,0:contrast.shape[1]]=contrast

                    all_labels[image_index]=label_index
                    image_index=image_index+1
            label_index=label_index+1
    all_data=np.concatenate((all_data_mfcc,all_data_spectrogram),axis=1)
    all_data=np.concatenate((all_data,all_data_beats),axis=1)
    all_data = np.concatenate((all_data, all_data_rmse), axis=1)
    all_data = np.concatenate((all_data, all_data_centroid), axis=1)
    all_data = np.concatenate((all_data, all_data_bandwidth), axis=1)
    all_data = np.concatenate((all_data, all_data_cens), axis=1)
    all_data = np.concatenate((all_data, all_data_stft), axis=1)
    all_data = np.concatenate((all_data, all_data_cq), axis=1)
    all_data = np.concatenate((all_data, all_data_contrast), axis=1)

    return all_data,all_labels

all_data,all_labels=read_data()

def normalize(x):
    # xvar, xavg = x.std(axis=0), x.mean(axis=0)
    xmin, xmax = x.min(axis=0), x.max(axis=0)
    # print(xavg)
    x = (x - xmin)
    diff=xmax-xmin
    # xvar_0=xvar==0
    diff_0 = diff == 0
    if(isinstance(diff_0,bool)==False):
        diff[diff_0]=1
    x=x/diff
    print("After normalization:")
    xmin, xmax = x.min(), x.max()
    print(xmin)
    print(xmax)

    return x


print(all_data.shape)

all_data=normalize(all_data)

print(all_data.shape)


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

all_data,all_labels=randomize(all_data,all_labels)


def split_data(dataset,labels,num_classes=10,test_images_for_class=25):
    test_counter=np.zeros(num_classes)
    test_set=np.zeros((num_classes*test_images_for_class,dataset.shape[1],dataset.shape[2]),dtype=float)
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

train_set,train_labels,test_set,test_labels=split_data(all_data,all_labels)


pickle_file = 'dataset_normalized_all.pickle'

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

