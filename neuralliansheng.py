
import librosa
import numpy as np
import os
import keras
from keras import models
from keras.layers import Dense, Dropout
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from sklearn.preprocessing import Normalizer


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def Mfcc_Vectors(filename):      
    y, sr = librosa.load(filename,sr=None)
    
    v=librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr,n_mels=36))  
    vector=np.sum(v,1).reshape(-1)  
    
    
    return vector
    
    
path0 = "D:/output/0"
path1 = "D:/output/1"

files0= os.listdir(path0)
files1= os.listdir(path1)


def Mfcc_set(files,path):
    line=[]
    for file in files:
        if not os.path.isdir(file):

        
            line.append(Mfcc_Vectors(path+"/"+file))
           
       
    line=np.array(line)
    
    
    return line
set0=Mfcc_set(files0,path0)
set1=Mfcc_set(files1,path1)
train=np.vstack((set0,set1))
print(set0.shape[0])
print(set1.shape[0])
tag_0=np.zeros(set0.shape[0])
tag_1=np.ones(set1.shape[0])

ss=StandardScaler()
no=Normalizer()

train_tag= np_utils.to_categorical(np.append(tag_0,tag_1))

X_train, X_test, y_train, y_test = train_test_split(np.array(train, dtype='float64'),
                                                        np.array(train_tag, dtype='float64'), test_size=0.30,
                                                        random_state=0)
# X_train = ss.fit_transform(X_train)
X_train = no.transform(X_train)
# X_test = ss.transform(X_test)
X_test = no.transform(X_test)

def create_model():
    model = models.Sequential()
    
    model.add(Dense(36, activation='relu', input_shape=(X_train.shape[1],)))
  
    model.add(Dense(36, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model

model = create_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30, batch_size=8)
test_loss, test_acc = model.evaluate(X_test,y_test)
model.save('D:/output/neuralmodel_normalizer.h5')
print('test_acc: ',test_acc)