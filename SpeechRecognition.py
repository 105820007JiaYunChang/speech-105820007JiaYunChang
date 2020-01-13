# 導入函式庫
from preprocess import *
import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout,Activation, Flatten,Embedding,LSTM, Conv2D,GlobalMaxPool2D, MaxPooling2D,BatchNormalization,AveragePooling2D,GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adadelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.engine.input_layer import Input
# 載入 data 資料夾的訓練資料，並自動分為『訓練組』及『測試組』
size1=20
size2=11
X_train, X_test, y_train, y_test = get_train_test(split_ratio=0.7,)
    
X_train = X_train.reshape(X_train.shape[0], size1, size2, 1)
X_test = X_test.reshape(X_test.shape[0], size1 ,size2, 1)

# 類別變數轉為one-hot encoding
y_train_hot = to_categorical(y_train,)
y_test_hot = to_categorical(y_test,)
print("X_train.shape=", X_train.shape)

mReduceLROnPlateau=ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, 
                                     mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.0000001)
mEarlyStop=EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
 verbose=1, mode='auto', baseline=None, restore_best_weights=True)
# 建立簡單的線性執行的模型
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='elu', input_shape=(size1, size2, 1)))
model.add(Conv2D(48, kernel_size=(2, 2), activation='elu'))
model.add(Conv2D(64, kernel_size=(2, 2), activation='elu'))
model.add(Conv2D(128, kernel_size=(2, 2), activation='elu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))

#model = Sequential()
## 建立卷積層，filter=32,即 output size, Kernal Size: 2x2, activation function 採用 relu
#model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=(size1, size2, 1)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
#model.add(Conv2D(64, kernel_size=(2, 2), activation='relu',))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(32, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=2),
              metrics=['accuracy'])

# 進行訓練, 訓練過程會存在 train_history 變數中
model.fit(X_train, y_train_hot, batch_size=800, epochs=20,
          verbose=1, validation_data=(X_test, y_test_hot),callbacks=[mReduceLROnPlateau,mEarlyStop])


#X_train = X_train.reshape(X_train.shape[0], size1, size2, 1)
#X_test = X_test.reshape(X_test.shape[0], size1, size2, 1)
#score = model.evaluate(X_test, y_test_hot, verbose=1)
#
## 模型存檔
#from keras.models import load_model
#model.save('COOLAA.h5')
#
#
## 預測(prediction)
#mfcc = wav2mfcc('./data/happy/012c8314_nohash_0.wav')
#mfcc_reshaped = mfcc.reshape(1, 20, 11, 1)
#print("labels=", get_labels())
#print("predict=", np.argmax(model.predict(mfcc_reshaped)))
