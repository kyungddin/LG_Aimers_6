# 남성 요인에 대한 EDA 진행
# 즉 정자 농도, 정자 운동성, 정자 형태와 남성 요인의 연관성 파악이 목표!!
# 이게 일단 남성 요인 최종이다!!

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


#==============================================================================

train = pd.read_csv('train.csv').drop(columns=['ID'])
test = pd.read_csv('test.csv').drop(columns=['ID'])

X = train.drop('임신 성공 여부', axis=1) # input
y = train['임신 성공 여부'] # target

X = X.iloc[:, 10:27]
X = X.drop(['여성 주 불임 원인', '여성 부 불임 원인'
            , '불임 원인 - 난관 질환'
            , '불임 원인 - 배란 장애', '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제'
            , '불임 원인 - 자궁내막증'
            , '부부 주 불임 원인'
            , '부부 부 불임 원인', '불명확 불임 원인'], axis=1)

X = X.astype(int)

total_counts1 = train['불임 원인 - 정자 면역학적 요인'].value_counts()
print(total_counts1)

count1 = ((X['불임 원인 - 정자 형태'] == 1) & (y == 0)).sum()
# 143 중 111 = 77.62%
count2 = ((X['불임 원인 - 정자 운동성'] == 1) & (y == 0)).sum()
# 97 중 80 = 82.47%
count3 = ((X['불임 원인 - 정자 농도'] == 1) & (y == 0)).sum()
# 276 중 207 = 75%
count4 = ((X['불임 원인 - 정자 면역학적 요인'] == 1) & (y == 0)).sum()
# 1 of 1..

counta = ((X['불임 원인 - 남성 요인'] == 1) & (X['남성 주 불임 원인'] == 1)).sum() # 약 50%
countb = ((X['불임 원인 - 남성 요인'] == 1) & (X['남성 부 불임 원인'] == 1)).sum() # 약 40%
countc = ((X['불임 원인 - 남성 요인'] == 1) & (X['불임 원인 - 정자 형태'] == 1)).sum() # 100%
countd = ((X['불임 원인 - 남성 요인'] == 1) & (X['불임 원인 - 정자 운동성'] == 1)).sum() # 100%
counte = ((X['불임 원인 - 남성 요인'] == 1) & (X['불임 원인 - 정자 면역학적 요인'] == 1)).sum() # 100%
countf = ((X['불임 원인 - 남성 요인'] == 1) & (X['불임 원인 - 정자 농도'] == 1)).sum() # 100%

countx = ((X['불임 원인 - 남성 요인'] == 0) & (X['남성 주 불임 원인'] == 1)).sum() # 
county = ((X['불임 원인 - 남성 요인'] == 0) & (X['남성 부 불임 원인'] == 1)).sum() # 

countz = ((X['불임 원인 - 남성 요인'] == 1) & (train['임신 성공 여부'] == 0)).sum() # 

count11 = ((X['불임 원인 - 남성 요인'] == 0) & (X['남성 주 불임 원인'] == 1) & (train['임신 성공 여부'] == 0)).sum()
count12 = ((X['불임 원인 - 남성 요인'] == 0) & (X['남성 부 불임 원인'] == 1) & (train['임신 성공 여부'] == 0)).sum()


X = X.drop('불임 원인 - 정자 면역학적 요인', axis=1)

import prince

features = ["불임 원인 - 남성 요인", "불임 원인 - 정자 농도", 
            "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태"]

X = pd.DataFrame(X)  # 혹시 numpy 배열이면 DataFrame으로 변환
X_subset = X[features].copy()  # 선택한 feature만 사용

# prince 라이브러리로 MCA 모델 훈련 (n_components=1로 차원 축소)
mca = prince.MCA(n_components=1)

# MCA 모델을 훈련시키고 차원 축소된 데이터 얻기
mca_result = mca.fit_transform(X_subset)

# 기존 feature 삭제
X = X.drop(columns=features)

# 차원 축소된 데이터를 원본 데이터에 새로운 열로 추가
X["불임 원인 - 남성 요인"] = mca_result

# from scipy.stats import chi2_contingency
# # Cramér's V 함수 정의
# def cramers_v(x, y):
#     # 교차표 계산
#     contingency_table = pd.crosstab(x, y)
#     # 카이제곱 검정 수행
#     chi2, p, dof, expected = chi2_contingency(contingency_table)
#     # Cramér's V 계산
#     return np.sqrt(chi2 / (len(x) * (min(contingency_table.shape) - 1)))

# # 불임 원인 - 남성 요인과 다른 feature들 간의 Cramér's V 값 계산
# cramers_v_matrix = []

# # '불임 원인 - 남성 요인'과 다른 feature들의 Cramér's V 값 계산
# for column in X.columns:
#     if column != '불임 원인 - 남성 요인':  # '불임 원인 - 남성 요인'은 제외하고 계산
#         v = cramers_v(X['불임 원인 - 남성 요인'], X[column])
#         cramers_v_matrix.append(v)

# # 결과를 DataFrame 형태로 변환
# cramers_v_df = pd.DataFrame(cramers_v_matrix, columns=['Cramér\'s V'], index=X.columns[X.columns != '불임 원인 - 남성 요인'])

# # 히트맵 그리기
# plt.figure(figsize=(10, 8))
# sns.heatmap(cramers_v_df.T, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
# plt.title('Cramér\'s V Heatmap: 불임 원인 - 남성 요인 vs 다른 feature들')
# plt.show()

import scipy.stats as stats

# Cramér's V 계산 함수
def cramers_v(confusion_matrix):
    chi2, p, dof, expected = stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()  # 전체 샘플 수
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

# '불임 원인 - 남성 요인'과 다른 feature들 간의 Cramér's V 계산
cramers_v_values = {}
for col in ['남성 주 불임 원인', '남성 부 불임 원인']:  # 비교할 feature들
    contingency_table = pd.crosstab(X['불임 원인 - 남성 요인'], X[col])
    cramers_v_values[col] = cramers_v(contingency_table)

# 결과 출력
print(cramers_v_values)