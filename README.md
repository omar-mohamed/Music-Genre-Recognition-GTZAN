# Music-Genre-Recognition-GTZAN
A deep learning project to recognize the genre of a specific song snippet.

# Problem:

Given a segment of a song, the task is to classify the genre of the song into one of these classes:
- Blues
- Classical
- Country
- Disco
- HipHop
- Jazz
- Metal
- Pop
- Reggae
- Rock


# Dataset:

GTZAN Genre Collection: The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.

Link: [GTZAN_Dataset](http://marsyasweb.appspot.com/download/data_sets/)

### Dataset Splitting:

- The original GTZAN dataset contains 1k 30-seconds .wav files.
- Each class has 100 30-seconds snippets from songs
- Used 90% of the data for training (900 sample).
- Used 10% for testing (100 sample)
- Each class has the same number of samples in training and testing (90 and 10 respectively)


# Preprocessing:

Extracted multiple features like:
- MFCC
![image](https://user-images.githubusercontent.com/6074821/43685392-8751dc74-98b2-11e8-8ab1-0d9dceb9e9db.png)

- Melspectrogram
![image](https://user-images.githubusercontent.com/6074821/43685404-d7ff8b94-98b2-11e8-9573-94846ca59158.png)

- Tempogram
![image](https://user-images.githubusercontent.com/6074821/43685440-675f1bce-98b3-11e8-91b2-b640761e221b.png)

And other similar features and concatenated them into a 87x1400 array for each image and normalizing over all images.

Used Librosa library for feature extraction
Link: [Librosa_features](https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html#librosa.feature.melspectrogram)


# Training Methods & Results:

## Guassian Mixture Models:

- First, vectorize the 87x1400 array of each sample
- Run PCA to reduce data to 900 element
- Run GMM for each class of the 10 
- Used GMM covariance type full
- On testing, get the probability for each class and maximize

- **Accuracy on training: 99.9%**
- **Accuracy on testing: 71%**

## SVM:

- First, vectorize the 87x1400 array of each sample
- Run PCA to reduce data to 900 element
- Run Linear SVM on training data 

- **Accuracy on training: 100%**
- **Accuracy on testing: 74%**

## Neural Networks:

- First, vectorize the 87x1400 array of each sample
- Run PCA to reduce data to 900 element
- NN is made of one hidden layer of 64 nodes
- Leaky Relu Used as activation 
- Used Batch Normalization
- Used Dropout 0.5

- **Accuracy on training: 100%**
- **Accuracy on testing: 50%**

## Convolutional Neural Networks:

- Two convolutional layers followed by fully connected
- Leaky Relu Used as activation 
- Used Batch Normalization
- Used Dropout 0.5

- **Accuracy on training: 99%**
- **Accuracy on testing: 35%**

## Notes & Modifications:

As you might have noticed from the results, all the training methods seem to struggle with overfitting as the data size is very small, only 90 song per class for training, especially neural networks which up till now achieving worse on test set. So here are a couple of things that may increase accuracy:

- Use SVM on convolutional neural network output (the output of the convolutional layers) 
- Use GMM on convolutional neural network output (the output of the convolutional layers)
- Try more NN architectures and hyper parameters 
- Try 3D convolutions (structure data in 3d way)
- Data augmentation 

# Environment Used:
- Python 3.6.1
- Tensorflow 1.9
