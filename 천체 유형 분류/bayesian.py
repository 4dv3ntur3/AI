import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

RANDOM_SEED=66

data= np.load('./data/sdss_uv_s.npy', allow_pickle=True)

x = data[:, 2:]
y = data[:, 1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=RANDOM_SEED
)

def xgb_cv(
    learning_rate, 
    n_estimators, 
    gamma,
    min_child_weight, 
    subsample,
    colsample_bytree,
    silent=True,
    n_jobs= -1,
    ):

    model = XGBClassifier(max_depth=9,
                        learning_rate=learning_rate,
                        n_estimators=int(n_estimators),
                        gamma=gamma,
                        min_child_weight=min_child_weight,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        tree_method='gpu_hist', 
                        predictor='gpu_predictor',
                        n_jobs=-1,
                        objective='multi:softmax'
                        )


    #훈련
    model.fit(x_train, y_train)

    #예측값 출력
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)

    return acc

#베이지안 최적화 라이브러리
from bayes_opt import BayesianOptimization

#범위
test = {
    'learning_rate': (0.01, 0.025),
    'n_estimators':(1000, 1100),
    'gamma': (0.99, 2),
    'min_child_weight': (1, 10), 
    'subsample': (0.5, 1),
    'colsample_bytree': (0.1, 0.5),
}

# Bayesian optimization 객체 생성
# f : 탐색 대상 함수, pbounds : hyperparameter 집합
# verbose = 2 항상 출력, verbose = 1 최댓값일 때 출력, verbose = 0 출력 안함

bo = BayesianOptimization(f=xgb_cv, pbounds=test, verbose=2, random_state=RANDOM_SEED)    

# 메소드를 이용해 최대화 과정 수행
# init_points :  초기 Random Search 개수
# n_iter : 반복 횟수 (몇개의 입력값-함숫값 점들을 확인할지! 많을수록 정확한 값을 얻을 수 있다.)
# acq : Acquisition Function들 중 Expected Improvement(EI) 를 사용
bo.maximize(init_points=2, n_iter=30, acq='ei')

# ‘iter’는 반복 회차, ‘target’은 목적 함수의 값, 나머지는 입력값을 나타냅니다. 
# 현재 회차 이전까지 조사된 함숫값들과 비교하여, 현재 회차에 최댓값이 얻어진 경우, 
# bayesian-optimization 라이브러리는 이를 자동으로 다른 색 글자로 표시하는 것을 확인

# 찾은 파라미터 값 확인
print(bo.max)
