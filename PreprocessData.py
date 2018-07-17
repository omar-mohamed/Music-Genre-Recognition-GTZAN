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

import librosa
import librosa.display
#http://opihi.cs.uvic.ca/sound/genres.tar.gz
#GTZAN genre collection.

y,sr=librosa.load('./genres/blues/blues.00000.au')

librosa.display.waveplot(y,sr)

D=librosa.stft(y)

log_power=librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.max)

librosa.display.specshow(log_power,x_axis='time',y_axis='log')
plt.colorbar()

mfcc=librosa.feature.mfcc(y=y,sr=sr)

print(mfcc)