# EDA by km at 250210
# 남성 요인 관련 feature 분석

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
X_train_undersampled = X_train_undersampled.drop(['여성 주 불임 원인', '여성 부 불임 원인', '부부 주 불임 원인'
                                                  , '부부 부 불임 원인', '불명확 불임 원인', '불임 원인 - 난관 질환'
                                                  , '불임 원인 - 배란 장애', '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제'
                                                  , '불임 원인 - 자궁내막증'], axis=1)

X_train_undersampled = X_train_undersampled.astype(int)

for column in X_train_undersampled.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=X_train_undersampled[column])
    plt.title(f'Count plot for {column}')
    plt.show()

#==============================================================================

from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    """Cramér's V 계산 함수"""
    confusion_matrix = pd.crosstab(x, y)  # 교차표 생성
    chi2 = chi2_contingency(confusion_matrix)[0]  # 카이제곱 통계량
    n = confusion_matrix.sum().sum()  # 전체 샘플 수
    k = min(confusion_matrix.shape)  # 행과 열 중 작은 값
    return np.sqrt(chi2 / (n * (k - 1)))

def cramers_v_matrix(df):
    """Cramér's V 행렬 생성"""
    columns = df.columns
    cramers_v_matrix = np.zeros((len(columns), len(columns)))
    
    for i in range(len(columns)):
        for j in range(len(columns)):
            cramers_v_matrix[i][j] = cramers_v(df[columns[i]], df[columns[j]])
    
    return pd.DataFrame(cramers_v_matrix, columns=columns, index=columns)

# 남성 요인 데이터에서 Cramér's V 계산
X_train_male_factors = X_train_undersampled[['남성 주 불임 원인', '남성 부 불임 원인',
                                             '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인',
                                             '불임 원인 - 정자 형태', '불임 원인 - 정자 운동성',
                                             '불임 원인 - 남성 요인']]
X_train_male_factors = X_train_male_factors.apply(LabelEncoder().fit_transform)

# Cramér's V 결과 행렬
cramers_v_result = cramers_v_matrix(X_train_male_factors)

# 히트맵 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cramers_v_result, annot=True, cmap='coolwarm', fmt='.3f', cbar=True)
plt.title("Cramér's V Test Results for Male Factors")
plt.show()