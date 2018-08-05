# %matplotlib inline

from __future__ import print_function
from six.moves import cPickle as pickle
from six.moves import range
import os
import numpy as np
import librosa
import librosa.display


# http://opihi.cs.uvic.ca/sound/genres.tar.gz
# GTZAN genre collection.

################## Data Loading #####################

print("Loading Data")
# function to read and concatenated features from the files
def read_data(directory='./genres'):
    all_data_mfcc = np.zeros((1000, 20, 1400), dtype=float)
    all_data_spectrogram = np.zeros((1000, 20, 1400), dtype=float)
    all_data_beats = np.zeros((1000, 1, 1400), dtype=float)
    all_data_contrast = np.zeros((1000, 7, 1400), dtype=float)
    all_data_cq = np.zeros((1000, 12, 1400), dtype=float)
    all_data_cens = np.zeros((1000, 12, 1400), dtype=float)
    all_data_stft = np.zeros((1000, 12, 1400), dtype=float)
    all_data_centroid = np.zeros((1000, 1, 1400), dtype=float)
    all_data_bandwidth = np.zeros((1000, 1, 1400), dtype=float)
    all_data_rmse = np.zeros((1000, 1, 1400), dtype=float)

    all_labels = np.zeros(1000)
    label_index = 0
    image_index = 0
    for _, dirs, _ in os.walk(directory):
        for dir in dirs:
            for _, _, files in os.walk(directory + '/' + dir):
                for file in files:
                    print("Loading features of: " + file)
                    y, sr = librosa.load(directory + '/' + dir + '/' + file)
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
                    all_data_spectrogram[image_index, :, 0:S.shape[1]] = S
                    all_data_mfcc[image_index, :, 0:mfcc.shape[1]] = mfcc
                    all_data_beats[image_index, :, 0:beats.shape[0]] = beats
                    all_data_beats[image_index, :, beats.shape[0] + 1] = tempo
                    #
                    all_data_rmse[image_index, :, 0:rmse.shape[1]] = rmse
                    all_data_centroid[image_index, :, 0:cent.shape[1]] = cent
                    all_data_bandwidth[image_index, :, 0:spec_bw.shape[1]] = spec_bw
                    all_data_stft[image_index, :, 0:chroma_stft.shape[1]] = chroma_stft
                    all_data_cens[image_index, :, 0:chroma_cens.shape[1]] = chroma_cens
                    all_data_cq[image_index, :, 0:chroma_cq.shape[1]] = chroma_cq
                    all_data_contrast[image_index, :, 0:contrast.shape[1]] = contrast

                    all_labels[image_index] = label_index
                    image_index = image_index + 1
            label_index = label_index + 1
    all_data = np.concatenate((all_data_mfcc, all_data_spectrogram), axis=1)
    all_data = np.concatenate((all_data, all_data_beats), axis=1)
    all_data = np.concatenate((all_data, all_data_rmse), axis=1)
    all_data = np.concatenate((all_data, all_data_centroid), axis=1)
    all_data = np.concatenate((all_data, all_data_bandwidth), axis=1)
    all_data = np.concatenate((all_data, all_data_cens), axis=1)
    all_data = np.concatenate((all_data, all_data_stft), axis=1)
    all_data = np.concatenate((all_data, all_data_cq), axis=1)
    all_data = np.concatenate((all_data, all_data_contrast), axis=1)

    return all_data, all_labels


all_data, all_labels = read_data()

print("Shape of data:")
print(all_data.shape)

print("Shape of labels:")
print(all_labels.shape)


################## Normalization #####################


# function to standarize the data x=(x-mean)/std

def standarize(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    diff = x - mean
    z_scores_np = np.divide(diff, std, out=np.zeros_like(diff), where=std != 0)

    print('Mean after standarization: %.1f%%' % z_scores_np.mean())

    print('std after standarization: %.1f%%' % z_scores_np.std())

    return z_scores_np


# function to min_max normalize *currently not used*
def min_max_normalize(x):
    min = x.min(axis=0)
    max = x.max(axis=0)
    diff = x - min
    range = max - min
    np_minmax = np.divide(diff, range, out=np.zeros_like(diff), where=range != 0)
    print('Min after normalization: %.1f%%' % np_minmax.min())

    print('Max after normalization: %.1f%%' % np_minmax.max())
    return np_minmax


# function to shuffle the data
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


print("Shuffling data")
all_data, all_labels = randomize(all_data, all_labels)

print("Standarizing data")
all_data_standard = standarize(all_data)


################## Splitting #####################


# function to split data into training and testing (default is 90% for training and 10% for testing)
def split_data(dataset, labels, num_classes=10, test_images_for_class=10):
    test_counter = np.zeros(num_classes)
    if dataset.ndim == 3:
        test_set = np.zeros((num_classes * test_images_for_class, dataset.shape[1], dataset.shape[2]), dtype=float)
    elif dataset.ndim == 2:
        test_set = np.zeros((num_classes * test_images_for_class, dataset.shape[1]), dtype=float)

    test_labels = np.zeros(num_classes * test_images_for_class)
    deleted_index = []
    test_index = 0
    for i in range(labels.shape[0]):
        _class = int(labels[i])
        if test_counter[_class] >= test_images_for_class:
            continue
        test_counter[_class] = test_counter[_class] + 1
        deleted_index.append(i)
        test_labels[test_index] = labels[i]
        test_set[test_index] = dataset[i]
        test_index = test_index + 1
    dataset = np.delete(dataset, deleted_index, axis=0)
    labels = np.delete(labels, deleted_index, axis=0)

    return dataset, labels, test_set, test_labels


print("Splitting data into training and testing")

train_set_std, train_labels_std, test_set_std, test_labels_std = split_data(all_data_standard, all_labels)

print("Shape of training set:")
print(train_set_std.shape)

print("Shape of training labels:")
print(train_labels_std.shape)

print("Shape of test set:")
print(test_set_std.shape)

print("Shape of test labels:")
print(test_labels_std.shape)

################## Saving data #####################


# save data to a pickle file to load when training

print("Saving data into pickle file")

pickle_file = 'dataset_standarized_all_10.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_set_std,
        'train_labels': train_labels_std,
        'test_dataset': test_set_std,
        'test_labels': test_labels_std,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print("Done")
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
