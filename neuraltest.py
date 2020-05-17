from keras.models import load_model
from keras.utils import np_utils
import librosa
import numpy as np
import os
import keras
from keras import models
from keras.layers import Dense, Dropout
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

model = load_model('D:/output/neuralmodel_normalizer.h5')

def Mfcc_Vectors(filename):      
    y, sr = librosa.load(filename,sr=None)
    
    v=librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr,n_mels=36))  
    vector=np.sum(v,1).reshape(-1)  
    
    print(vector)
    return [vector]

no=Normalizer()
# ss=StandardScaler()

test_X = Mfcc_Vectors("D:/output/test/4.mp3")

test_X = no.transform(test_X)

# test_X = ss.fit_transform(test_X)

pred = model.predict(test_X)
print(pred)
