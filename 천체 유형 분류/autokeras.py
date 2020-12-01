#2020-11-26 
#autokeras

#pip install autokeras
#pip instal git+https://github.com/keras-team/keras-tuner.git@1.0.2rc4

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout


import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence
from sklearn.model_selection import train_test_split
import autokeras as ak


random_seed=66
data = np.load('./data/sdss_uv_s.npy', allow_pickle=True)

x = data[:, 2:]
y = data[:, 1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_seed
)


#initialize the image classifier
clf = ak.StructuredDataClassifier(
    overwrite=True,
    max_trials=3
)

#feed the image calssifier with training data
clf.fit(x_train, y_train, 'type', epochs=20) #epoch 바꿔 보기


#predict with the best model.
predicted_y = clf.predict(x_test)
print(predicted_y)

#evaluate the best model with testing data.
print(clf.evaluate(x_test, 'type'))
#clf.summary() 안 먹힌다 



#튜닝이 거의 필요 없다

