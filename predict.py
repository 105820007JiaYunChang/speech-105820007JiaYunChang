# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:51:56 2020

@author: CaesarYu
"""
import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from preprocess import *
from keras.models import load_model
import pandas as pd

#DATA_PATH="C://Users//CaesarYu//Desktop//SpeechRecognition//test//"
#
#
#save_test_data_to_array(DATA_PATH)
  # creates a HDF5 file 'model.h5'
##TRAIN_PATH="C://Users//CaesarYu//Desktop//SpeechRecognition//train//train//audio//_background_noise_"
#TRAIN_PATH="C://Users//CaesarYu//Desktop//SpeechRecognition//train//train//audio//"
aa=get_labels()[0]
#testdatas=np.load('TEST_NPY//test.npy')
#testdatas=np.load('TEST877//test.npy')
model=load_model('COOLAA.h5')

#output=[]
#size1=20
#size2=11
#i=1
#for testdata in testdatas:    
#    testdata_reshaped = testdata.reshape(1, size1, size2, 1)
##    print("predict=", np.argmax(model.predict(testdata_reshaped)))
#    ans=aa[np.argmax(model.predict(testdata_reshaped))]
#    output.append([i,ans[0:len(ans)-4]])
#    i=i+1
#output=pd.DataFrame(output)
#output.columns=['id','word']
#output.to_csv('AAAAA.csv',index=False)
##wave, sr = librosa.load(TRAIN_PATH+'//doing_the_dishes.wav', mono=True, sr=None)

##asd=get_labels()
##print("labels=", get_labels())