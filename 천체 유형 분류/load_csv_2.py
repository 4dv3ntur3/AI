#19개 CSV 합치기
#####import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100


#####data load

#vs code 좌측의 탐색기에서 연 폴더가 루트 폴더가 된다
#test에는 y값이 없으므로 column이 21임



galaxy = pd.read_csv('./data/sdss_csv/galaxy.csv', encoding='CP949')
qso = pd.read_csv('./data/sdss_csv/qso.csv', encoding='CP949')
redden_std = pd.read_csv('./data/sdss_csv/redden_std.csv', encoding='CP949')
rosat_d = pd.read_csv('./data/sdss_csv/rosat_d.csv', encoding='CP949')
serendipity_blue = pd.read_csv('./data/sdss_csv/serendipity_blue.csv', encoding='CP949')
serendipity_manual = pd.read_csv('./data/sdss_csv/serendipity_manual.csv', encoding='CP949')
serendipity_red = pd.read_csv('./data/sdss_csv/serendipity_red.csv', encoding='CP949')
serendipity_first = pd.read_csv('./data/sdss_csv/serendipity_first.csv', encoding='CP949')
serendipity_distant = pd.read_csv('./data/sdss_csv/serendipity_distant.csv', encoding='CP949')
sky = pd.read_csv('./data/sdss_csv/sky.csv', encoding='CP949')
spectrophoto_std = pd.read_csv('./data/sdss_csv/spectrophoto_std.csv', encoding='CP949')
star_brown_dwarf = pd.read_csv('./data/sdss_csv/star_brown_dwarf.csv', encoding='CP949')
star_bhb = pd.read_csv('./data/sdss_csv/star_bhb.csv', encoding='CP949')
star_carbon = pd.read_csv('./data/sdss_csv/star_carbon.csv', encoding='CP949')
star_caty_var = pd.read_csv('./data/sdss_csv/star_caty_var.csv', encoding='CP949')
star_pn = pd.read_csv('./data/sdss_csv/star_pn.csv', encoding='CP949')
star_red_dwarf = pd.read_csv('./data/sdss_csv/star_red_dwarf.csv', encoding='CP949')
star_sub_dwarf = pd.read_csv('./data/sdss_csv/star_sub_dwarf.csv', encoding='CP949')
star_white_dwarf = pd.read_csv('./data/sdss_csv/star_white_dwarf.csv', encoding='CP949')


sdss = pd.concat([galaxy, qso, rosat_d, redden_std, serendipity_blue, serendipity_manual, serendipity_red, serendipity_distant, serendipity_first, sky, spectrophoto_std, star_brown_dwarf,
 star_bhb, star_carbon, star_caty_var, star_pn, star_red_dwarf,star_sub_dwarf,star_white_dwarf], ignore_index=True)
print(sdss.info()) 
# print(sdss.describe())
# print(sdss)



#데이터 type의 분포 확인
fig = plt.figure(figsize=(18, 9))
plt.grid()
sdss['type'].value_counts()[:100].plot(kind='bar', alpha=0.7)
plt.title('type distribution')
plt.show()

print(sdss['type'].value_counts()[:100])



#값 제거 전
np.save('./data/sdss_only.npy', arr=sdss)



#중복되는 행 제거
# print(sdss.duplicated()) #3491 

print(sdss.duplicated().sum()) #3491 
sdss = sdss.drop_duplicates()
print("======중복행 제거======")
sdss.info() #196500
print("duplicated: ", sdss.duplicated().sum())


print(sdss.describe()) #196407.000000

print("null: ", sdss.isnull().sum())


#-9999.00000 있는 행 제거
#-9999 -> np.nan 후 한 번에 제거
sdss = sdss.replace(-9999.00000, np.nan)
sdss = sdss.dropna(axis=0, how='any')
sdss.info()
print(sdss.describe()) #count 196407.000000


# np.save('./data/sdss_del.npy', arr=sdss)


# print(sdss_nan.shape) #(196500, 22)
# print(sdss_nan.describe())
# sdss_nan.info()


np.save('./data/sdss_clean.npy', arr=sdss)

#자외선 지수 합치기
uv = pd.read_csv('./data/uv_nm.csv', index_col=0, encoding='CP949')
uv.info()
dataset = pd.merge(sdss, uv, on='mjd', how='left')
dataset.info() #193452

#자외선 지수가 없는 날의 데이터(결측치) 제거 
dataset = dataset.dropna(axis=0, how='any') #193452
dataset.info()
print(dataset.describe())

np.save('./data/sdss_uv.npy', arr=dataset)


#labeling: 'type' column
labeling = dataset.type.unique()

np.save('./data/sdss_label.npy', arr=labeling)
# dataset.to_csv('./data/sdss_uv.csv')







