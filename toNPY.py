# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 00:03:20 2020

@author: CaesarYu
"""

from preprocess import *
import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from preprocess import *
from keras.models import load_model
import pandas as pd

TRAIN_PATH="C://Users//CaesarYu//Desktop//SpeechRecognition//train//train//"
#TEST_PATH="C://Users//CaesarYu//Desktop//SpeechRecognition//test//"
save_data_to_array(path=TRAIN_PATH,max_pad_len=11)
#save_test_data_to_array(path=TEST_PATH,max_pad_len=24)
