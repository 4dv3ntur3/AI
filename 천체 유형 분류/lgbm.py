from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score

random_seed = 66
data = np.load('./data/sdss_uv_s.npy', allow_pickle=True)

x = data[:, 2:]
y = data[:, 1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_seed
)


model = LGBMClassifier(
    bagging_fraction=0.7, 
    feature_fraction=0.7, 
    is_unbalance=True,
    lambda_l1=4.972, 
    lambda_l2=2.276, 
    learning_rate=0.03, 
    max_depth=18,
    min_child_weight=6.338458091153288,
    min_split_gain=0.08935815884666935, 
    num_class=19, 
    num_leaves=30,
    objective='multiclass')

model.fit(x_train, y_train)

kfold = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(model, x_train, y_train, cv=kfold)

y_pred = model.predict(x_test)

score = accuracy_score(y_pred, y_test)
print(model, "\n 's acc: ", score)


'''
tuning x 
acc:  0.6508490346592231

LGBMClassifier(bagging_fraction=0.7, booster='gdbt', feature_fraction=0.7,
               learning_rate=0.01, max_depth=9,
               min_child_weight=6.338458091153288,
               min_split_gain=0.08935815884666935, num_class=19, num_leaves=10,
               objective='multiclass')
 's acc:  0.6683724897262929

LGBMClassifier(bagging_fraction=0.7, feature_fraction=0.7, is_unbalance=True,
               lambda_l1=4.972, lambda_l2=2.276, learning_rate=0.3, max_depth=8,
               min_child_weight=6.338458091153288,
               min_split_gain=0.08935815884666935, num_class=19, num_leaves=10,
               objective='multiclass')
 's acc:  0.7812411155048978


LGBMClassifier(bagging_fraction=0.7, feature_fraction=0.7, is_unbalance=True,
               lambda_l1=4.972, lambda_l2=2.276, learning_rate=0.3, max_depth=8,
               min_child_weight=6.338458091153288,
               min_split_gain=0.08935815884666935, num_class=19, num_leaves=20,
               objective='multiclass')
 's acc:  0.7932335685301491

LGBMClassifier(bagging_fraction=0.7, feature_fraction=0.7, is_unbalance=True,
               lambda_l1=4.972, lambda_l2=2.276, learning_rate=0.3, max_depth=8,
               min_child_weight=6.338458091153288,
               min_split_gain=0.08935815884666935, num_class=19, num_leaves=30,
               objective='multiclass')
 's acc:  0.796231681786462


LGBMClassifier(bagging_fraction=0.7, feature_fraction=0.7, is_unbalance=True,
               lambda_l1=4.972, lambda_l2=2.276, learning_rate=0.3, max_depth=8,
               min_child_weight=6.338458091153288,
               min_split_gain=0.08935815884666935, num_class=19, num_leaves=40,
               objective='multiclass')
 's acc:  0.7949393915897754
 




'''