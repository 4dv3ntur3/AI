#npy load & PCA별로 압축(feature importance)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#콘솔창에 보여지는 행과 열의 크기 
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100


#seaborn:statistical data visualization
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings 
warnings.filterwarnings('ignore')


#1. 데이터

#npy로 작업
data = np.load('./data/sdss_uv_s.npy', allow_pickle=True)
# label = np.load('./data/sdss_label.npy', allow_pickle=True)



'''
#labeling 작업 이후에 x, y 분리
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
encoder = LabelEncoder()
encoder.fit(label)
label = encoder.transform(label)
# print(label)
# print(encoder.classes_)

#One Hot Encoding을 위해 2차원 배열로 변환
label = label.reshape(-1, 1)
oh = OneHotEncoder()
oh.fit(label)
oh_label = oh.transform(label)
oh_label.toarray()
'''


x = data[:20000, 3:] #fiberID, mjd 제외
y = data[:20000, 1]


#데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8
)


#전처리
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



# 2. 모델
# model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
# XGBooster는 CPU를 병렬로 사용해서 속도가 느린 편 
model = XGBClassifier(max_depth=10)




#3. 훈련
model.fit(x_train, y_train)


#4. 평가 및 예측
print("accuracy: ", model.score(x_test, y_test))

'''

'''



