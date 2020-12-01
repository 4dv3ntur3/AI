#feature importance 확인
#교차 검증
#gridsearch, randomsearch

#콘솔창에 보여지는 행과 열의 크기 
# pd.options.display.max_rows = 100
# pd.options.display.max_columns = 100

# import matplotlib.pyplot as plt

label = np.load('./data/sdss_label.npy', allow_pickle=True)





random_seed=66

import numpy as np
import pandas as pd
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


x = data[:, 2:] #mjd, type 제외
y = data[:, 1]

print(x.shape) #(193452, 23)


pca = PCA()
pca.fit(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
d = np.argmax(cumsum >= 0.95) + 1
print(cumsum >= 0.95)
print(d)


'''
(193452, 23)
[0.9942214  0.99833267 0.99963311 0.99988818 0.99993882 0.99997348
 0.99998054 0.9999867  0.99999013 0.99999324 0.99999472 0.99999594
 0.99999704 0.99999773 0.9999983  0.99999875 0.99999911 0.99999941
 0.9999996  0.99999974 0.99999985 0.99999993 1.        ]
[ True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True]

1
'''

'''

import matplotlib.pyplot as plt

plt.plot(cumsum)
plt.grid() #격자에 넣어 줌
plt.show()


# pca = PCA(n_components=1)
# x2d = pca.fit_transform(x)
'''


from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_seed
)


# 2. 모델
model = XGBClassifier(tree_method='gpu_hist', 
                    predictor='gpu_predictor',
                    n_jobs=-1,
                    )

#3. 훈련
model.fit(x_train, y_train)


#4. 평가 및 예측
print("before: ", model.score(x_test, y_test))


thresholds = np.argsort(thresholds)
x_train = x_train[:, thresholds[:18]]
x_test = x_test[:, thresholds[:18]]

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("after: ", score)



'''
thresholds = np.sort(model.feature_importances_)
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


# '''
# 제거했더니 더 떨어짐
# pca 및 feature importance는 사용 안 하는 것으로..



# [13:04:02] WARNING: C:\Users\Administrator\workspace\xgboost-win64_release_1.2.0\src\learner.cc:516:
# Parameters: { num_classes } might not be used.

'''
before:  0.7906748339407097

after:  0.7192111860639425


  This may not be accurate due to some parameters are only used in language bindings but
  passed down to XGBoost core.  Or some parameters are not used but slip through this
  verification. Please open an issue if you find above cases.


'''


# '''

'''

from xgboost import plot_importance
plot_importance(model)
plt.show()
'''


# # feature importance
# import matplotlib.pyplot as plt
# import numpy as np
# def plot_feature_importances(model):
#     n_features = x_train.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#                 align='center')

#     plt.yticks(np.arange(n_features)) #아무것도 주지 않으면 0, 1, 2, 3(index)로 표기되어 나온다
#                                                               #아래 -> 위 순서임
#     plt.xlabel("feature importances")
#     plt.ylabel("features")
#     plt.ylim(-1, n_features)

# plot_feature_importances(model)
# plt.show()


# '''
# (193452, 23)
# [0.9942214  0.99833267 0.99963311 0.99988818 0.99993882 0.99997348
#  0.99998054 0.9999867  0.99999013 0.99999324 0.99999472 0.99999594
#  0.99999704 0.99999773 0.9999983  0.99999875 0.99999911 0.99999941
#  0.9999996  0.99999974 0.99999985 0.99999993 1.        ]
# [ True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True  True  True  True  True  True]
# 1
# [12:16:23] WARNING: C:\Users\Administrator\workspace\xgboost-win64_release_1.2.0\src\learner.cc:516: 
# Parameters: { num_classes } might not be used.

#   This may not be accurate due to some parameters are only used in language bindings but
#   passed down to XGBoost core.  Or some parameters are not used but slip through this
#   verification. Please open an issue if you find above cases.


'''
accuracy:  0.7906748339407097
[0.00307059 0.00916665 0.01460616 0.01575622 0.01593518 0.0181066
 0.01907608 0.01965153 0.01973368 0.02284114 0.02517437 0.03133697
 0.03574115 0.03637766 0.04063398 0.05132947 0.0539331  0.05563413
 0.07912228 0.0795078  0.11227353 0.11734547 0.1236463 ]

Thresh=0.003, n=23, acc: 78.91%
Thresh=0.009, n=22, acc: 79.12%
Thresh=0.015, n=21, acc: 79.13%
Thresh=0.016, n=20, acc: 79.11%
Thresh=0.016, n=19, acc: 79.14%
Thresh=0.018, n=18, acc: 79.33%
Thresh=0.019, n=17, acc: 77.22%
Thresh=0.020, n=16, acc: 72.01%
Thresh=0.020, n=15, acc: 72.17%
Thresh=0.023, n=14, acc: 72.15%
Thresh=0.025, n=13, acc: 72.17%
Thresh=0.031, n=12, acc: 72.05%
Thresh=0.036, n=11, acc: 71.68%
Thresh=0.036, n=10, acc: 71.78%
Thresh=0.041, n=9, acc: 71.52%
Thresh=0.051, n=8, acc: 71.70%
Thresh=0.054, n=7, acc: 71.67%
Thresh=0.056, n=6, acc: 71.45%
Thresh=0.079, n=5, acc: 68.13%
Thresh=0.080, n=4, acc: 67.10%
Thresh=0.112, n=3, acc: 65.61%
Thresh=0.117, n=2, acc: 51.94%
Thresh=0.124, n=1, acc: 36.74%
'''
# 
# '''