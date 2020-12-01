#####import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100


#seaborn:statistical data visualization
import seaborn as sns         

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings 
warnings.filterwarnings('ignore')




#####data load

#vs code 좌측의 탐색기에서 연 폴더가 루트 폴더가 된다
#test에는 y값이 없으므로 column이 21임
train = pd.read_csv('./data/train.csv', index_col=0) #size of train data:  (199991, 22)
test = pd.read_csv('./data/test.csv', index_col=0) #size of test data:  (10009, 21) 


sample_submission = pd.read_csv('./data/sample_submission.csv', index_col=0)

train.info()
print(train.describe())





#데이터 type의 분포 확인
# fig = plt.figure(figsize=(18, 9))
# plt.grid()
# train['fiberID'].value_counts()[:100].plot(kind='bar', alpha=0.7)
# plt.title('fiber ID distribution')
# plt.show()

# print(train['fiberid'].value_counts()[:100])












# print('size of train data: ', train.shape)
# print('size of test data: ', test.shape)
# train.info()

'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 199991 entries, 0 to 199990
Data columns (total 22 columns):
 #   Column      Non-Null Count   Dtype
---  ------      --------------   -----
 0   type        199991 non-null  object
 1   fiberID     199991 non-null  int64
 2   psfMag_u    199991 non-null  float64
 3   psfMag_g    199991 non-null  float64
 4   psfMag_r    199991 non-null  float64
 5   psfMag_i    199991 non-null  float64
 6   psfMag_z    199991 non-null  float64
 7   fiberMag_u  199991 non-null  float64
 8   fiberMag_g  199991 non-null  float64
 9   fiberMag_r  199991 non-null  float64
 10  fiberMag_i  199991 non-null  float64
 11  fiberMag_z  199991 non-null  float64
 12  petroMag_u  199991 non-null  float64
 13  petroMag_g  199991 non-null  float64
 14  petroMag_r  199991 non-null  float64
 15  petroMag_i  199991 non-null  float64
 16  petroMag_z  199991 non-null  float64
 17  modelMag_u  199991 non-null  float64
 18  modelMag_g  199991 non-null  float64
 19  modelMag_r  199991 non-null  float64
 20  modelMag_i  199991 non-null  float64
 21  modelMag_z  199991 non-null  float64
dtypes: float64(20), int64(1), object(1)
memory usage: 35.1+ MB

'''

# print(train.describe())


#데이터 type의 분포 확인
# fig = plt.figure(figsize=(18, 9))
# plt.grid()
# train['type'].value_counts()[:100].plot(kind='bar', alpha=0.7)
# plt.title('train data distribution')
# plt.show()

# print(train['type'].value_counts()[:100].plot(kind='bar', alpha=0.7))



##### 1. 데이터

#train data의 type(정답) column을 sample_submission에 대응하는 가변수 형태로 변환.
#one hot encoding과 비슷

# column_number = {}
# for i, column in enumerate(sample_submission.columns):
#     column_number[column] = i

# def to_number(x, dic):
#     return dic[x]

# train['type_num'] = train['type'].apply(lambda x: to_number(x, column_number))



#dataset X, Y 분리
train.info()
y = train['type']
x = train.drop(columns=['type'], axis=1)

print(type(x))
print(type(y))

print(y.iloc[0])
print(type(y.iloc[0]))

# x = x.iloc[:10000, :]
# y = y.iloc[:10000]




#test에 정답이 없기 때문에 그냥 train 가지고 train_test_split 한다
# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, random_state=66
# )


# np.save('./data/npy/x_train.npy', arr=x_train)
# np.save('./data/npy/y_train.npy', arr=y_train)
# np.save('./data/npy/x_test.npy', arr=x_test)
# np.save('./data/npy/y_test.npy', arr=y_test)
# np.save('./data/npy/test.npy', arr=test)




'''
#randomforestclassifier
accuracy:  0.8784219605490138

#xgbooster
accuracy:  0.8385

'''