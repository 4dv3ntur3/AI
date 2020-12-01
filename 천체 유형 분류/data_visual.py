


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


data = pd.read_csv('./data/sdss_uv.csv', encoding='CP949', index_col=0)

# for colname in data.columns[2:]:
#     plt.figure(figsize = (15, 5))
#     plt.subplot(1,2,1); plt.title(f"TRAIN: {colname}"); sns.distplot(data[colname].values)
#     plt.show()


plt.figure(figsize=(15,4)); plt.xticks(rotation = 45)
print(sns.barplot(data=data.type.value_counts().reset_index(), x="index", y="type"))
plt.show()


