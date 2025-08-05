# EDA_MALE_CRAMER_NO_SAMPLING

import pandas as pd
import numpy as np
from sklearn.preprocessing import  OrdinalEncoder
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.font_manager as fm
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

from sklearn.utils import resample
from sklearn.utils import shuffle

train = pd.read_csv('train.csv').drop(columns=['ID'])
test = pd.read_csv('test.csv').drop(columns=['ID'])

X = train.drop('임신 성공 여부', axis=1) # input
y = train['임신 성공 여부'] # target

# '단일 배아 이식 여부'별 데이터 개수 세기
total_counts1 = train['여성 주 불임 원인'].value_counts()
total_counts2 = train['여성 부 불임 원인'].value_counts()
total_counts3 = train['불임 원인 - 난관 질환'].value_counts()
total_counts4 = train['불임 원인 - 배란 장애'].value_counts()
total_counts5 = train['불임 원인 - 자궁경부 문제'].value_counts()
total_counts6 = train['불임 원인 - 자궁내막증'].value_counts()
total_counts7 = train['불명확 불임 원인'].value_counts()
total_counts8 = train['불임 원인 - 여성 요인'].value_counts()

# 출력
print("여성 주 불임 원인:")
print(total_counts1)
print("\n여성 부 불임 원인:")
print(total_counts2)
print("\n불임 원인 - 난관 질환:")
print(total_counts3)
print("\n불임 원인 - 배란 장애:")
print(total_counts4)
print("\n불임 원인 - 자궁경부 문제:")
print(total_counts5)
print("\n불임 원인 - 자궁내막증:")
print(total_counts6)
print("\n불임 원인 - 불명확 불임 원인:")
print(total_counts7)
print("\n불임 원인 - 여성 요인:")
print(total_counts8)
c1 = ((train['임신 성공 여부'] == 0) & (X['불임 원인 - 난관 질환'] == 1)).sum()
c2 = ((train['임신 성공 여부'] == 0) & (X['불임 원인 - 배란 장애'] == 1)).sum()
c3 = ((train['임신 성공 여부'] == 0) & (X['불임 원인 - 자궁경부 문제'] == 1)).sum()
c4 = ((train['임신 성공 여부'] == 0) & (X['불임 원인 - 자궁내막증'] == 1)).sum()

counta = ((X['불명확 불임 원인'] == 1) & (X['불임 원인 - 난관 질환'] == 1)).sum()
countb = ((X['불명확 불임 원인'] == 1) & (X['불임 원인 - 배란 장애'] == 1)).sum()
countc = ((X['불명확 불임 원인'] == 1) & (X['불임 원인 - 자궁경부 문제'] == 1)).sum()
countd = ((X['불명확 불임 원인'] == 1) & (X['불임 원인 - 자궁내막증'] == 1)).sum()
counte = ((X['불명확 불임 원인'] == 1) & (train['임신 성공 여부'] == 0)).sum()
print(counta)
print(countb)
print(countc)
print(countd)
print(counte)
# 불명확 불임 원인과는 별로 관련이 없구나..
# 불명확 불임 원인은 또 따로 빼야 할 듯 하고..

# 기존의 여성 요인은 과감하게 제거하고 (난관 질환 or 배란 장애 or 자궁경부 문제 or 자궁내막증 = 여성 요인)
# 즉 or gate featuring 진행

# 여성 요인 feature 생성
X['여성 요인'] = X[['불임 원인 - 난관 질환', '불임 원인 - 배란 장애', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증']].any(axis=1).astype(int)
X['불명확 여성 요인'] = (X['여성 요인'] & X['불명확 불임 원인']).astype(int)

print(X['여성 요인'].value_counts())
print(X['불명확 여성 요인'].value_counts())

# true 79000

countf = ((X['여성 요인'] == 1) & (train['임신 성공 여부'] == 0)).sum()
# 30.5%

count11 = ((X['여성 요인'] == 0) & (X['여성 주 불임 원인'] == 1) & (train['임신 성공 여부'] == 0)).sum()
count12 = ((X['여성 요인'] == 0) & (X['여성 부 불임 원인'] == 1) & (train['임신 성공 여부'] == 0)).sum()
# 하ㅂ치면 5600 3% 정도..
count13 = ((X['불명확 불임 원인'] == 1) & (train['임신 성공 여부'] == 0)).sum()
count14 = ((X['불명확 여성 요인'] == 1) & (train['임신 성공 여부'] == 0)).sum()
# 1.3%

X.loc[X['불명확 여성 요인'] == 1, '불명확 불임 원인'] = 0
print(X['불명확 불임 원인'].value_counts())
