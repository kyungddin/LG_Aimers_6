import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prince

from sklearn.preprocessing import  OrdinalEncoder
from sklearn.ensemble import ExtraTreesClassifier

train = pd.read_csv('./train.csv').drop(columns=['ID'])
test = pd.read_csv('./test.csv').drop(columns=['ID'])

#=============================================================================

# '임신 시도 또는 마지막 임신 경과 연수' 열 제거
train = train.drop(columns=['임신 시도 또는 마지막 임신 경과 연수'])
test = test.drop(columns=['임신 시도 또는 마지막 임신 경과 연수'])

# 착상 전 유전 검사 및 진단 사용 여부 결측치 검사
train['착상 전 유전 검사 사용 여부'] = train['착상 전 유전 검사 사용 여부'].fillna(0)
train['착상 전 유전 진단 사용 여부'] = train['착상 전 유전 진단 사용 여부'].fillna(0)

test['착상 전 유전 검사 사용 여부'] = test['착상 전 유전 검사 사용 여부'].fillna(0)
test['착상 전 유전 진단 사용 여부'] = test['착상 전 유전 진단 사용 여부'].fillna(0)

# '남성 주 불임 원인'과 '남성 부 불임 원인' 열만 선택
male_infertility_data = train[['남성 주 불임 원인', '남성 부 불임 원인']]

# MCA 모델 훈련 (n_components=1로 차원 축소)
mca = prince.MCA(n_components=1)

# MCA 모델을 훈련시키고 차원 축소된 데이터 얻기
mca_result = mca.fit_transform(male_infertility_data)

# 차원 축소된 데이터를 원본 데이터에 새로운 열로 추가
train['남성 불임 원인'] = mca_result

# '남성 주 불임 원인'과 '남성 부 불임 원인' 열 삭제
train = train.drop(columns=['남성 주 불임 원인', '남성 부 불임 원인'])

# '남성 주 불임 원인'과 '남성 부 불임 원인' 열만 선택
test_male_infertility_data = test[['남성 주 불임 원인', '남성 부 불임 원인']]

# MCA 모델 훈련 (n_components=1로 차원 축소)
mca = prince.MCA(n_components=1)

# MCA 모델을 훈련시키고 차원 축소된 데이터 얻기
mca_result = mca.fit_transform(test_male_infertility_data)

# 차원 축소된 데이터를 원본 데이터에 새로운 열로 추가
test['남성 불임 원인'] = mca_result

# '남성 주 불임 원인'과 '남성 부 불임 원인' 열 삭제
test = test.drop(columns=['남성 주 불임 원인', '남성 부 불임 원인'])

# '여성 주 불임 원인'과 '여성 부 불임 원인' 열만 선택
female_infertility_data = train[['여성 주 불임 원인', '여성 부 불임 원인']]

# MCA 모델 훈련 (n_components=1로 차원 축소)
mca_f = prince.MCA(n_components=1)

# MCA 모델을 훈련시키고 차원 축소된 데이터 얻기
mca_f_result = mca_f.fit_transform(female_infertility_data)

# 차원 축소된 데이터를 원본 데이터에 새로운 열로 추가
train['여성 불임 원인'] = mca_f_result

# '남성 주 불임 원인'과 '남성 부 불임 원인' 열 삭제
train = train.drop(columns=['여성 주 불임 원인', '여성 부 불임 원인'])

# '여성 주 불임 원인'과 '여성 부 불임 원인' 열만 선택
test_female_infertility_data = test[['여성 주 불임 원인', '여성 부 불임 원인']]

# MCA 모델 훈련 (n_components=1로 차원 축소)
mca_f = prince.MCA(n_components=1)

# MCA 모델을 훈련시키고 차원 축소된 데이터 얻기
mca_f_result = mca_f.fit_transform(test_female_infertility_data)

# 차원 축소된 데이터를 원본 데이터에 새로운 열로 추가
test['여성 불임 원인'] = mca_f_result

# '남성 주 불임 원인'과 '남성 부 불임 원인' 열 삭제
test = test.drop(columns=['여성 주 불임 원인', '여성 부 불임 원인'])

# '부부 주 불임 원인'과 '부부 부 불임 원인' 열만 선택
couple_infertility_data = train[['부부 주 불임 원인', '부부 부 불임 원인']]

# MCA 모델 훈련 (n_components=1로 차원 축소)
mca_c = prince.MCA(n_components=1)

# MCA 모델을 훈련시키고 차원 축소된 데이터 얻기
mca_c_result = mca_c.fit_transform(couple_infertility_data)

# 차원 축소된 데이터를 원본 데이터에 새로운 열로 추가
train['부부 불임 원인'] = mca_c_result

# '남성 주 불임 원인'과 '남성 부 불임 원인' 열 삭제
train = train.drop(columns=['부부 주 불임 원인', '부부 부 불임 원인'])

# '부부 주 불임 원인'과 '부부 부 불임 원인' 열만 선택
test_couple_infertility_data = test[['부부 주 불임 원인', '부부 부 불임 원인']]

# MCA 모델 훈련 (n_components=1로 차원 축소)
mca_c = prince.MCA(n_components=1)

# MCA 모델을 훈련시키고 차원 축소된 데이터 얻기
mca_c_result = mca_c.fit_transform(test_couple_infertility_data)

# 차원 축소된 데이터를 원본 데이터에 새로운 열로 추가
test['부부 불임 원인'] = mca_c_result

# '남성 주 불임 원인'과 '남성 부 불임 원인' 열 삭제
test = test.drop(columns=['부부 주 불임 원인', '부부 부 불임 원인'])

# 사용할 feature 목록
features = [
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태"
]

# '불임 원인 - 남성 요인'이 0일 때만 max(axis=1)을 적용하여 변경
train["불임 원인 - 남성 요인"] = train.apply(
    lambda row: row[features].max() if row["불임 원인 - 남성 요인"] == 0 else row["불임 원인 - 남성 요인"],
    axis=1
)
test["불임 원인 - 남성 요인"] = test.apply(
    lambda row: row[features].max() if row["불임 원인 - 남성 요인"] == 0 else row["불임 원인 - 남성 요인"],
    axis=1
)


# '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태' 열 삭제
train = train.drop(columns=['불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태'])
test = test.drop(columns=['불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태'])

# 사용할 feature 목록
features_female = [
    "불임 원인 - 난관 질환",
    "불임 원인 - 배란 장애",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증"
]

# 단순 OR 연산을 이용한 컬럼 통합
train["불임 원인 - 여성 요인"] = train[features_female].max(axis=1)
test["불임 원인 - 여성 요인"] = test[features_female].max(axis=1)

# '남성 주 불임 원인'과 '남성 부 불임 원인' 열 삭제
train = train.drop(columns=['불임 원인 - 난관 질환', '불임 원인 - 배란 장애', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증'])
test = test.drop(columns=['불임 원인 - 난관 질환', '불임 원인 - 배란 장애', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증'])

train["배아 생성 주요 이유"] = train["배아 생성 주요 이유"].apply(lambda x: 1 if isinstance(x, str) and "현재 시술용" in x else 0)
test["배아 생성 주요 이유"] = test["배아 생성 주요 이유"].apply(lambda x: 1 if isinstance(x, str) and "현재 시술용" in x else 0)

# drop
train = train.drop(columns=['IVF 임신 횟수', 'IVF 출산 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수'])
test = test.drop(columns=['IVF 임신 횟수', 'IVF 출산 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수'])

# 1. 결측치가 매우 높은 두 feature drop
train.drop(['PGD 시술 여부'], axis=1, inplace=True)
train.drop(['PGS 시술 여부'], axis=1, inplace=True)

test.drop(['PGD 시술 여부'], axis=1, inplace=True)
test.drop(['PGS 시술 여부'], axis=1, inplace=True)

# 2. 동결/신선 배아 하나로 합치기
train.drop(['동결 배아 사용 여부'], axis=1, inplace=True)
test.drop(['동결 배아 사용 여부'], axis=1, inplace=True)

# 3. 대리모 여부 drop
train.drop(['대리모 여부'], axis=1, inplace=True)
test.drop(['대리모 여부'], axis=1, inplace=True)

# '난자 채취 경과일' 열 제거
train = train.drop(columns=['난자 채취 경과일'])
test = test.drop(columns=['난자 채취 경과일'])

# '난자 해동 경과일' 열 제거
train = train.drop(columns=['난자 해동 경과일'])
test = test.drop(columns=['난자 해동 경과일'])

# '난자 혼합 경과일' 열 제거
train = train.drop(columns=['난자 혼합 경과일'])
test = test.drop(columns=['난자 혼합 경과일'])

# '배아 이식 경과일'의 평균값 계산 (결측치는 제외한 값으로 평균 계산), 반올림 처리
mean_value = round(train['배아 이식 경과일'].mean())  # 반올림 처리


# 결측치를 평균값으로 대체
train['배아 이식 경과일'].fillna(mean_value, inplace=True)

# '배아 이식 경과일'의 평균값 계산 (결측치는 제외한 값으로 평균 계산), 반올림 처리
mean_value = round(test['배아 이식 경과일'].mean())  # 반올림 처리

# 결측치를 평균값으로 대체
test['배아 이식 경과일'].fillna(mean_value, inplace=True)

# '배아 해동 경과일'별 데이터 개수 세기
total_counts_5 = train['배아 해동 경과일'].value_counts()
test_total_counts_5 = test['배아 해동 경과일'].value_counts()

# 특정 컬럼의 결측치 개수 계산
missing_count = train['배아 해동 경과일'].isna().sum()
test_missing_count = test['배아 해동 경과일'].isna().sum()

# 원본 데이터의 비율 계산
value_ratios = total_counts_5 / total_counts_5.sum()
test_value_ratios = test_total_counts_5 / test_total_counts_5.sum()

# 각 값에 대해 채울 개수 계산
fill_counts = (value_ratios * missing_count).round().astype(int)
test_fill_counts = (test_value_ratios * test_missing_count).round().astype(int)

# 결측치 샘플링
filled_values = np.concatenate([
    np.full(count, value) for value, count in fill_counts.items()
])

test_filled_values = np.concatenate([
    np.full(count, value) for value, count in test_fill_counts.items()
])

# 배열을 섞어 랜덤 배치
np.random.shuffle(filled_values)
np.random.shuffle(test_filled_values)

# 결측치 채우기
train.loc[train['배아 해동 경과일'].isna(), '배아 해동 경과일'] = filled_values
test.loc[test['배아 해동 경과일'].isna(), '배아 해동 경과일'] = test_filled_values

train = train.drop(columns=['해동된 배아 수'])
train = train.drop(columns=['해동 난자 수'])
train = train.drop(columns=['기증자 정자와 혼합된 난자 수'])
train = train.drop(columns=['파트너 정자와 혼합된 난자 수'])

test = test.drop(columns=['해동된 배아 수'])
test = test.drop(columns=['해동 난자 수'])
test = test.drop(columns=['기증자 정자와 혼합된 난자 수'])
test = test.drop(columns=['파트너 정자와 혼합된 난자 수'])

#=============================================================================

X = train.drop('임신 성공 여부', axis=1)
y = train['임신 성공 여부']

categorical_columns = [
    "시술 시기 코드",
    "시술 당시 나이",
    "시술 유형",
    "특정 시술 유형",
    "배란 자극 여부",
    "배란 유도 유형",
    "단일 배아 이식 여부",
    "착상 전 유전 검사 사용 여부",
    "착상 전 유전 진단 사용 여부",
    "불임 원인 - 남성 요인",
    "불임 원인 - 여성 요인",
    "배아 생성 주요 이유",
    "총 시술 횟수",
    "DI 시술 횟수",
    "총 임신 횟수",
    "DI 임신 횟수",
    "총 출산 횟수",
    "총 생성 배아 수",
    "DI 출산 횟수",
    "난자 출처",
    "정자 출처",
    "난자 기증자 나이",
    "정자 기증자 나이",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "남성 불임 원인",
    "여성 불임 원인",
    "부부 불임 원인"
]

numeric_columns = [
    "총 생성 배아 수",
    "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수",
    "이식된 배아 수",
    "미세주입 배아 이식 수",
    "저장된 배아 수",
    "미세주입 후 저장된 배아 수",
    "수집된 신선 난자 수",
    "저장된 신선 난자 수",
    "혼합된 난자 수",
    "배아 이식 경과일",
    "배아 해동 경과일"
]

# 5. 범주형 데이터의 모든 자료형을 문자열로 변환
for col in categorical_columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)
    
    
# 6. 범주의 정수화 (by sklearn 전처리)   
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# test encoding 시 기존에 없던 범주를 발견하면 -1로 처리

X_train_encoded = X.copy()
X_train_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

X_test_encoded = test.copy()
X_test_encoded[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# 8. 결측치 처리
X_train_encoded[numeric_columns] = X_train_encoded[numeric_columns].fillna(0)
X_test_encoded[numeric_columns] = X_test_encoded[numeric_columns].fillna(0)

#=============================================================================

from lightgbm import LGBMClassifier
model = LGBMClassifier(random_state=42)

#=============================================================================

# 1. 라이브러리 import
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 2. Stratified K-Fold 설정 (데이터 불균형 고려)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

# 3. Cross Validation 수행 
# 사용법: X_train_encoded에 train data set, y에 target 넣기!!
for train_idx, val_idx in kf.split(X_train_encoded, y):
    X_train_part, X_val = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
    y_train_part, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # 모델 학습
    model.fit(X_train_part, y_train_part)
    
    # Validation 데이터 예측 확률
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # ROC-AUC 점수 계산
    roc_auc = roc_auc_score(y_val, val_pred_proba)
    auc_scores.append(roc_auc)

# 4. 평균 ROC-AUC 점수 출력
mean_auc = np.mean(auc_scores)
print(f"Mean ROC-AUC Score (5-Fold CV): {mean_auc:.10f}") # dacon과 같이 소수점 10자리 출력!