#uvA, uvB csv 정리
#New Mexico state
#Daily Sum	  UVA	 (MJ/(m^2 day))
#Daily Sum	  UVB	 (KJ/(m^2 day))

#1997-01-01 ~ 2017.12.30
#2001-01-01 이전 데이터 삭제



import numpy as np
import pandas as pd


uv_ab_sum = pd.read_csv('./data/csv/NEWMEXICO-dsums_modified.csv', encoding='CP949')

# uv_ab_sum.info()


'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6420 entries, 0 to 6419
Data columns (total 6 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   date      6420 non-null   object
 1   LYYY      6420 non-null   int64
 2   LM        6420 non-null   int64
 3   LD        6420 non-null   int64
 4      uvaDS  6420 non-null   float64
 5      uvbDS  6420 non-null   float64
dtypes: float64(2), int64(3), object(1)
memory usage: 301.1+ KB
'''

#2000-01-01 데이터부터 사용 
uv_ab_sum = uv_ab_sum[(uv_ab_sum["LYYY"] > 1999)]



#date -> mjd 변환 
#date -> jd -> mjd
import datetime

def to_mjd(date):

    date = date.split('-')
    # 기준으로 스플릿
    y = int(date[0])
    m = int(date[1])
    d = int(date[2])

    #프리겔의 공식(그레고리력 -> 율리우스력)
    #1월과 2월은 각각 전년도의 13월, 14월로 계산한다
    if m==1:
        y=y-1
        m=13
    elif m==2:
        y=y-1
        m=14

    mjd = int(365.25*y) + int(y/400) - int(y/100) + int(30.59*(m-2)) + int(d-678912)
    return mjd

# print(len(uv_ab_sum.index)) #6420


# 자외선 dataset에 mjd column 추가
# list로 추가
mjd_add = []
for i in range(len(uv_ab_sum.index)):
    mjd_add.append(to_mjd(uv_ab_sum.iloc[i, 0]))
    
# print(len(mjd_add)) #6420
uv_ab_sum['mjd'] = mjd_add

print(uv_ab_sum.describe())
print(uv_ab_sum)
print(uv_ab_sum.columns.values)

uv_ab_sum = uv_ab_sum[["mjd", "uvaDS", "uvbDS"]]
uv_ab_sum.to_csv('./data/uv_nm.csv')

#sdss data load
sdss = pd.read_csv('./data/csv/sdss.csv', encoding='CP949')
sdss.info()

dataset = pd.merge(sdss, uv_ab_sum, on='mjd', how='left')

dataset.info()

'''
Int64Index: 732977 entries, 0 to 732976
Data columns (total 20 columns):
 #   Column             Non-Null Count   Dtype
---  ------             --------------   -----
 0   object_id          732977 non-null  float64
 1   right_ascension    732977 non-null  float64
 2   declination        732977 non-null  float64
 3   u_magnitude        732977 non-null  float64
 4   g_magnitude        732977 non-null  float64
 5   r_magnitude        732977 non-null  float64
 6   i_magnitude        732977 non-null  float64
 7   z_magnitude        732977 non-null  float64
 8   obs_run_number     732977 non-null  int64
 9   rerun_number       732977 non-null  int64
 10  camera_column      732977 non-null  int64
 11  field_number       732977 non-null  int64
 12  spectro_object_id  732977 non-null  float64
 13  class              732977 non-null  object
 14  redshift           732977 non-null  float64
 15  plate_id           732977 non-null  int64
 16  mjd                732977 non-null  int64
 17  fiber_id           732977 non-null  int64
 18  uvaDS              705083 non-null  float64
 19  uvbDS              705083 non-null  float64
dtypes: float64(12), int64(7), object(1)
'''
print(dataset.tail())

np.save('./data/npy/sdss_none_del.npy', arr=dataset)

#이상치 처리
#자외선 수치가 없는 날의 data out(listwise)
dataset = dataset.dropna(axis=0, how='any')
# print(dataset) #[705083 rows x 20 columns]
# dataset_m = dataset[["class", "u_magnitude", "g_magnitude", "r_magnitude", "i_magnitude", "z_magnitude", "redshift", "uvaDS", "uvbDS"]]
# dataset_m.info()

np.save('./data/npy/sdss_del.npy', arr=dataset) 

