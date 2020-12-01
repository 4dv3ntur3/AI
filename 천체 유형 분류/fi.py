#feature importance 확인
#교차 검증
#gridsearch, randomsearch

random_seed=66


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#콘솔창에 보여지는 행과 열의 크기 
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings 
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA






#1. 데이터
#npy로 작업
data = np.load('./data/sdss_uv_s.npy', allow_pickle=True)

x = data[:10000, 2:] #mjd, type 제외
y = data[:10000, 1]


# print(x.shape) #(193452, 23)
# print(x[:10])


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_seed
)


# 2. 모델

# from lightgbm import LGBMClassifier
# model = LGBMClassifier()

# model = RandomForestClassifier()

model = XGBClassifier(tree_method='gpu_hist', 
                    predictor='gpu_predictor',
                    n_jobs=-1
                    )


#3. 훈련
model.fit(x_train, y_train)


#4. 평가 및 예측
print("acc: ", model.score(x_test, y_test))










'''
#feature importance 오름차순 정렬
thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    selection_model = XGBClassifier(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    #score
    score = accuracy_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, acc: %.2F%%" % (thresh, select_x_train.shape[1],
          score*100.0))
'''

# import matplotlib.pyplot as plt




'''
fi = model.best_estimator_.feature_importances_
print("feature importances: ", fi)

thresholds = np.argsort(fi)[::-1]
print(thresholds)
x_train = x_train[:, thresholds[:22]]
x_test = x_test[:, thresholds[:22]]

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("acc: ", score)
'''


'''
######## RandomForestClassifier
column 22개일 때 최대
10퍼센트로 테스트해 봄
accuracy:  0.7229258206254846
[0.02180588 0.02691884 0.02799914 0.03079707 0.03222677 0.03603209
 0.03968765 0.04083442 0.04121563 0.04439716 0.0449276  0.04756338
 0.04793087 0.04794028 0.04804041 0.04880131 0.04905387 0.04905821
 0.04933916 0.05192193 0.05251921 0.05350773 0.06748139]
Thresh=0.022, n=23, acc: 76.17%
Thresh=0.027, n=22, acc: 76.25%
Thresh=0.028, n=21, acc: 74.72%
Thresh=0.031, n=20, acc: 70.35%
Thresh=0.032, n=19, acc: 70.07%
Thresh=0.036, n=18, acc: 69.55%
Thresh=0.040, n=17, acc: 70.15%
Thresh=0.041, n=16, acc: 69.76%
Thresh=0.041, n=15, acc: 70.28%
Thresh=0.044, n=14, acc: 69.63%
Thresh=0.045, n=13, acc: 69.24%
Thresh=0.048, n=12, acc: 69.58%
Thresh=0.048, n=11, acc: 69.35%
Thresh=0.048, n=10, acc: 69.66%
Thresh=0.048, n=9, acc: 69.50%
Thresh=0.049, n=8, acc: 69.11%
Thresh=0.049, n=7, acc: 69.01%
Thresh=0.049, n=6, acc: 68.88%
Thresh=0.049, n=5, acc: 68.57%
Thresh=0.052, n=4, acc: 68.26%
Thresh=0.053, n=3, acc: 64.41%
Thresh=0.054, n=2, acc: 59.24%
Thresh=0.067, n=1, acc: 40.06%
'''


'''
######XGBClassifier
n=22 일 때 최대
accuracy:  0.7616955285603515
[0.00794394 0.01799276 0.01886998 0.01962239 0.02001533 0.0220001 
 0.02206289 0.02216259 0.03070869 0.03150214 0.03257238 0.03276909
 0.03533068 0.03768584 0.0385225  0.04716763 0.04957782 0.06366464
 0.07775965 0.08224916 0.08966011 0.09077578 0.1093841 ]
Thresh=0.008, n=23, acc: 76.17%
Thresh=0.018, n=22, acc: 76.25%
Thresh=0.019, n=21, acc: 74.72%
Thresh=0.020, n=20, acc: 75.19%
Thresh=0.022, n=18, acc: 70.04%
Thresh=0.022, n=17, acc: 69.94%
Thresh=0.022, n=16, acc: 69.84%
Thresh=0.031, n=15, acc: 70.33%
Thresh=0.032, n=14, acc: 70.46%
Thresh=0.033, n=13, acc: 70.38%
Thresh=0.033, n=12, acc: 70.04%
Thresh=0.035, n=11, acc: 70.12%
Thresh=0.038, n=10, acc: 68.88%
Thresh=0.039, n=9, acc: 69.53%
Thresh=0.047, n=8, acc: 70.12%
Thresh=0.050, n=7, acc: 69.37%
Thresh=0.064, n=6, acc: 69.14%
Thresh=0.078, n=5, acc: 68.86%
Thresh=0.082, n=4, acc: 67.90%
Thresh=0.090, n=3, acc: 65.06%
Thresh=0.091, n=2, acc: 57.77%
Thresh=0.109, n=1, acc: 35.33%
'''


# accuracy:  0.7616955285603515
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22]
# score:  0.751873869216852


###coulumn 1개 뺐더니 이 꼴이 나서 안 하기로 함 
#pyplot으로 feature importance 한 거 표 첨부하기... 
# PS D:\project>  cd 'd:\project'; & 'C:\Anaconda3\python.exe' 'c:\Users\bitcamp\.vscode\extensions\ms-python.python-2020.11.371526539\pythonFile539\pythonFiles\lib\python\debugpy\launcher' '50254' '--' 'd:\project\fi.py'
# accuracy:  0.7616955285603515
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22]
# score:  0.7616955285603515
