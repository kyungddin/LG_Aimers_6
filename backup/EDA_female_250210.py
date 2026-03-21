# EDA by km at 250210
# 여성 요인 관련 feature 분석

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

#==============================================================================

# 언더샘플링 (다수 클래스 샘플 감소)
X_train_majority = X[y == 0]  # 실패(0) 샘플
X_train_minority = X[y == 1]  # 성공(1) 샘플

y_train_majority = y[y == 0]
y_train_minority = y[y == 1]

# 다수 클래스(실패) 샘플을 소수 클래스(성공) 샘플 크기만큼 줄이기
X_train_majority_undersampled, y_train_majority_undersampled = resample(X_train_majority, y_train_majority, 
                                                                         replace=False,   # 중복되지 않게 샘플링
                                                                         n_samples=len(X_train_minority),  # 소수 클래스 크기만큼 샘플링
                                                                         random_state=42)

# 언더샘플링된 다수 클래스와 원본 소수 클래스를 합치기
X_train_undersampled = pd.concat([X_train_majority_undersampled, X_train_minority])
y_train_undersampled = pd.concat([y_train_majority_undersampled, y_train_minority])

# 섞기 (랜덤하게 섞어줍니다)
X_train_undersampled, y_train_undersampled = shuffle(X_train_undersampled, y_train_undersampled, random_state=42)

#==============================================================================

X_train_undersampled = X_train_undersampled.iloc[:, 10:27]
X_train_undersampled = X_train_undersampled.drop(['남성 주 불임 원인', '남성 부 불임 원인', '부부 주 불임 원인'
                                                  , '부부 부 불임 원인', '불임 원인 - 정자 농도'
                                                  , '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태'
                                                  , '불임 원인 - 남성 요인'], axis=1)

X_train_undersampled = X_train_undersampled.astype(int)

for column in X_train_undersampled.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=X_train_undersampled[column], color="#FF1493")
    plt.title(f'Count plot for {column}')
    plt.show()
    
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency

#==============================================================================

# '여성 요인' 관련 데이터 (X_train_undersampled)

# 카이제곱 검정을 통해 각 feature 간의 관계 파악
def chi2_test(df):
    columns = df.columns
    chi2_matrix = np.zeros((len(columns), len(columns)))
    
    for i in range(len(columns)):
        for j in range(len(columns)):
            contingency_table = pd.crosstab(df[columns[i]], df[columns[j]])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            chi2_matrix[i][j] = p  # p-value 저장
    
    chi2_df = pd.DataFrame(chi2_matrix, columns=columns, index=columns)
    return chi2_df

# 여성 요인 데이터에서 상관관계 계산
# 범주형 데이터라면, 먼저 LabelEncoder를 사용하여 레이블을 숫자로 변환 후 계산
X_train_female_factors = X_train_undersampled
X_train_female_factors = X_train_female_factors.apply(LabelEncoder().fit_transform)

# 카이제곱 검정 결과 (p-value)
chi2_result = chi2_test(X_train_female_factors)

# 히트맵 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(chi2_result, annot=True, cmap='coolwarm', fmt='.7f', cbar=False)
plt.title('Chi-Square Test Results (p-values) for FeMale Factors')
plt.show()