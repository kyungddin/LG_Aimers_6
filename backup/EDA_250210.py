# EDA by km at 250210
# 1번 feature 분석

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

b = train['시술 유형'].value_counts()
c = train['미세주입에서 생성된 배아 수'].value_counts()
a = ((train['총 생성 배아 수'] == 1) & (X['미세주입에서 생성된 배아 수'] == 1)).sum() # 3.47%
print(a)

print("dd")
countx = ((train['난자 출처'] == '기증 제공') & (train['시술 당시 나이'] == '만45-50세')).sum() # 3.47%
print(countx)
b = train['시술 당시 나이'].value_counts()
print(b)
county = ((train['시술 유형'] == 'IVF') & (train['시술 당시 나이'] == '만45-50세')).sum() # 3.47%
print(county)

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

# 1. 카이제곱 검정을 통한 '시술 시기 코드' 분석

from scipy.stats import chi2_contingency

# 교차 테이블 생성
crosstab = pd.crosstab(X_train_undersampled['시술 시기 코드'], y_train_undersampled)

# 카이제곱 검정
chi2, p, dof, expected = chi2_contingency(crosstab)
print(f"p-value: {p}")

# p-value가 0.05보다 작으면 두 변수 간에 유의미한 관계가 있다고 할 수 있습니다.
if p < 0.05:
    print("시술 시기 코드와 임신 성공 여부 사이에 유의미한 관계가 있습니다.")
else:
    print("시술 시기 코드와 임신 성공 여부 사이에 유의미한 관계가 없습니다.")
    
# 2. 시술 시기 코드에 대한 countplot 그리기
plt.figure(figsize=(10, 6))  # 그래프 크기 설정
sns.countplot(x='시술 시기 코드', data=X_train_undersampled)  # X는 train data
plt.title('시술 시기 코드의 빈도 분포')
plt.xlabel('시술 시기 코드')
plt.ylabel('빈도')
plt.xticks(rotation=45)  # x축 레이블 회전 (가독성 높이기)
plt.show()

# 3. 교차표 생성
cross_tab = pd.crosstab(X_train_undersampled['시술 시기 코드'], y_train_undersampled, margins=True, margins_name="Total")

# 교차표 출력
plt.figure(figsize=(10, 6))
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', cbar=True, linewidths=0.5)
plt.title("시술 시기 코드와 임신 성공 여부 교차표")
plt.show()