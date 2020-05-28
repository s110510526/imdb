import numpy as np
#import matplotlib.pyplot as plt

#讀入數據庫
from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)#只要最常出現的10000字

#資料前處理
from keras.preprocessing import sequence
x_train = sequence.pad_sequences(x_train, maxlen=100)#資料長度上限100字，不足補0
x_train = np.array(x_train)
#x_train = x_train.tolist()
x_test = sequence.pad_sequences(x_test, maxlen=100)#資料長度上限100字，不足補0
x_test = np.array(x_test)
#x_test = x_test.tolist()

#建構神經網路
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from keras import layers

model = Sequential() #空白神經網路機
model.add(Embedding(10000, 128)) #嵌入層(input維度,output維度)
model.add(Dropout(0.35))
model.add(LSTM(128))
model.add(Dense(units=256,activation='relu' ))
model.add(Dropout(0.35))
model.add(Dense(1, activation='sigmoid'))
'''
model = Sequential()
model.add(layers.Dense(50, activation = "relu", input_shape=(10000, )))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()
'''
#組裝
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#訓練
model.fit(x_train, y_train, batch_size=500, epochs=15)

#輸出模型各層參數
print(model.summary())

#評估
score = model.evaluate(x_test, y_test)
print('測試資料的loss', score[0])
print('測試資料的準確率', score[1])

#儲存訓練模型
model.save('imdb_modelb.h5')