

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical




#1. 데이터
data = np.load('./data/sdss_uv_s.npy', allow_pickle=True)
label = np.load('./data/sdss_label.npy', allow_pickle=True)

x = data[:, 2:]
y = data[:, 1]

from sklearn.preprocessing import StandardScaler,LabelEncoder

#labeling
encoder = LabelEncoder()
encoder.fit(label)
y = encoder.transform(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8
)


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#one hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델

model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(x_train.shape[1],))) 
model.add(Dense(150, activation='relu'))
model.add(Dense(110, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(19, activation='softmax'))


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, batch_size=1000, validation_split=0.2, callbacks=[early_stopping])



#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=1000)


print("=======DNN=======")
model.summary()
print("loss: ", loss)
print("acc: ", accuracy)

'''
=======DNN=======
loss:  0.6649184823036194
acc:  0.7659404277801514
'''