# Machine Learning 
##### NTUT-classroom-2019-Autumn
#
#### Speech Recognitioon
#
#### 目的
#
Google recently released the Speech Commands Datasets. It includes 65,000 one-second long utterances of 30 short words, by thousands of different people.

In this competition, you're challenged to use the Speech Commands Dataset to build an algorithm that understands simple spoken commands. By improving the recognition accuracy of open-sourced voice interface tools, we can improve product effectiveness and their accessibility.

#### 流程圖
[!image](https://imgur.com/0tuLhdh.jpg)
#### 內容
#
將所有音檔進行裁切並利用librosa進行MFCC特徵提取，
將提取出的特徵 Reshape 至 20*11 大小，
利用簡易的 CNN Model 進行訓練。
######  Loss
[!image](https://imgur.com/y9sXYqq.jpg)

######  ACC
[!image](https://imgur.com/XCZbKi4.jpg)


最後訓練完成後使用 predict.py 程式進行預測並輸出結果。


