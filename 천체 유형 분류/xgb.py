#feature importance 확인
#교차 검증
#gridsearch, randomsearch
#콘솔창에 보여지는 행과 열의 크기 





random_seed = 66

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import warnings 
warnings.filterwarnings('ignore')


pd.options.display.max_rows = 100
pd.options.display.max_columns = 100



#1. 데이터

#npy로 작업
data = np.load('./data/sdss_uv_s.npy', allow_pickle=True)

x = data[:, 2:] #mjd 제외
y = data[:, 1]



#데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_seed
)


# 전처리
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(tree_method='gpu_hist', 
                    predictor='gpu_predictor',
                    n_jobs=-1,
                    objective='multi:softmax',
                    num_classes=19,
                    colsample_bytree=0.4199756517368983,
                    gamma=1.2750227387976651,
                    learning_rate=0.02301482147915011,
                    max_depth=9,
                    min_child_weight=8.483928333898685,
                    n_estimators=1092,
                    subsample=0.9186264420010433,
                    random_state=random_seed,
                    silent=False
                    )

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score


kfold = KFold(n_splits=5, shuffle=True)

model.fit(x_train, y_train, verbose=1)

scores = cross_val_score(model, x_train, y_train, cv=kfold)



y_pred_a = model.predict(x_test)

print("accuracy_score: ", accuracy_score(y_pred_a, y_test))
print(model, "\n cross_val_score:", scores)





'''
아무것도 없이(data 전부)
model.score:  0.7906748339407097


#1회차
model.score:  0.7633041275748882
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.79, gamma=4.11, gpu_id=0,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.0769, max_delta_step=0, max_depth=5,
              min_child_weight=1.566, missing=nan,
              monotone_constraints='(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)',
              n_estimators=179, n_jobs=-1, num_classes=19, num_parallel_tree=1,
              objective='multi:softprob', predictor='gpu_predictor',
              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
              subsample=0.553, tree_method='gpu_hist', validate_parameters=1,
              verbosity=None) : [0.76367396 0.76392479 0.75823856 0.76292324 0.75924011]

2회차)
|  24       |  0.7896   |  0.6133   |  0.6875   |  0.04476  |  9.946    |  4.227    |  391.4    |  0.7025   |

iter       target         colsample   gamma      learning    max_depth min_childe n_estimator sub

model.score:  0.7904939133131736
log_loss:  0.5753018443164244
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6133, gamma=0.6875,
              gpu_id=0, importance_type='gain', interaction_constraints='',
              learning_rate=0.04476, max_delta_step=0, max_depth=9,
              min_child_weight=4.227, missing=nan,
              monotone_constraints='(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)',
              n_estimators=391, n_jobs=-1, num_classes=19, num_parallel_tree=1,
              objective='multi:softprob', predictor='gpu_predictor',
              random_state=66, reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
              silent=False, subsample=0.7025, tree_method='gpu_hist',
              validate_parameters=1, verbosity=None) 's cross_val_score: [0.78489969 0.78518351 0.78763892 0.78689584 0.79041742]

3회차)
accuracy_score:  0.7921738905688661
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5446571258589146,
              gamma=0.9920672327805469, gpu_id=0, importance_type='gain',
              interaction_constraints='', learning_rate=0.02302281977496389,
              max_delta_step=0, max_depth=9, min_child_weight=1.130961524021024,
              missing=nan,
              monotone_constraints='(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)',
              n_estimators=998, n_jobs=-1, num_classes=19, num_parallel_tree=1,
              objective='multi:softprob', predictor='gpu_predictor',
              random_state=66, reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
              silent=False, subsample=0.8289098372479776,
              tree_method='gpu_hist', validate_parameters=1, verbosity=None)
 cross_val_score: [0.79294414 0.78663737 0.78783277 0.78883432 0.78670199]


4회차)
accuracy_score:  0.7935954097852214
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4199756517368983,
              gamma=1.2750227387976651, gpu_id=0, importance_type='gain',
              interaction_constraints='', learning_rate=0.02301482147915011,
              max_delta_step=0, max_depth=9, min_child_weight=8.483928333898685,
              missing=nan,
              monotone_constraints='(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)',
              n_estimators=1092, n_jobs=-1, num_classes=19, num_parallel_tree=1,
              objective='multi:softprob', predictor='gpu_predictor',
              random_state=66, reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
              silent=False, subsample=0.9186264420010433,
              tree_method='gpu_hist', validate_parameters=1, verbosity=None)
 cross_val_score: [0.79048881 0.78624968 0.78944818 0.78818816 0.78915741]


5회차)



'''

