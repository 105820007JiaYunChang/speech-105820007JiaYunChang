import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np

#DATA_PATH="C://Users//CaesarYu//Desktop//SpeechRecognition//train//train//audio//"
DATA_PATH="C://Users//CaesarYu//Desktop//SpeechRecognition//TRAIN_NPY//"


# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


# Handy function to convert wav2mfcc
def wav2mfccV2(file_path, max_pad_len=12):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path+'\\', mono=True, sr=None)
    wave = wave[::3]
    
    # michael added to cut my audio file
    i=0
    # 訓練資料的長度
    wav_length=12000
    # 聲音檔過長，擷取片段
    if len(wave) > wav_length:
        # 尋找最大聲的點，取前後各半
        i=np.argmax(wave)
        if i > (wav_length):
            wave = wave[i-int(wav_length/2):i+int(wav_length/2)]
        else:
            # 聲音檔過長，取前面
            wave = wave[0:wav_length]
    
    mfcc = librosa.feature.mfcc(wave, sr=16000,n_mfcc=30)
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width < 0:
        pad_width = 0
#        mfcc = mfcc[:,:11]
        mfcc = mfcc[:,:max_pad_len]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc


def save_data_to_array(path=DATA_PATH, max_pad_len=11):
    labels, _, _ = get_labels(path)
    cc=1
    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
            print('%.3f' %(cc/(len(labels)*len(wavfiles))))
            cc=cc+1
        np.save(label + '.npy', mfcc_vectors)
def save_test_data_to_array(path=DATA_PATH, max_pad_len=11):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in sorted(os.listdir(path + '/' + label),key=lambda x:int(x[:-4]))]
        for wavfile in wavfiles:
            print(wavfile)
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)

def get_train_test(split_ratio=0.9, random_state=87):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)
    i=0
    for x in labels:
        labels[i]=x[0:len(x)-4]
        i=i+1
    
    # Getting first arrays
    X = np.load('.//TRAIN_NPY//'+labels[0] + '.npy')
    print(labels[0])
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        print(label)
        x = np.load('.//TRAIN_NPY//'+label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)



def prepare_dataset(path=DATA_PATH):
    labels, _, _ = get_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path  + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            # Downsampling
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data


def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]


# print(prepare_dataset(DATA_PATH))

