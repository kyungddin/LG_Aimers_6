# data_analysis_km_250203

import pandas as pd
import numpy as np
from sklearn.preprocessing import  OrdinalEncoder
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

train = pd.read_csv('train.csv').drop(columns=['ID'])
test = pd.read_csv('test.csv').drop(columns=['ID'])
X = train.drop('임신 성공 여부', axis=1) # input
y = train['임신 성공 여부'] # target
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
    "남성 주 불임 원인",
    "남성 부 불임 원인",
    "여성 주 불임 원인",
    "여성 부 불임 원인",
    "부부 주 불임 원인",
    "부부 부 불임 원인",
    "불명확 불임 원인",
    "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태",
    "배아 생성 주요 이유",
    "총 시술 횟수",
    "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수",
    "DI 시술 횟수",
    "총 임신 횟수",
    "IVF 임신 횟수",
    "DI 임신 횟수",
    "총 출산 횟수",
    "IVF 출산 횟수",
    "DI 출산 횟수",
    "난자 출처",
    "정자 출처",
    "난자 기증자 나이",
    "정자 기증자 나이",
    "동결 배아 사용 여부",
    "신선 배아 사용 여부",
    "기증 배아 사용 여부",
    "대리모 여부",
    "PGD 시술 여부",
    "PGS 시술 여부"
]
for col in categorical_columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)
numeric_columns = [
    "임신 시도 또는 마지막 임신 경과 연수",
    "총 생성 배아 수",
    "미세주입된 난자 수",
    "미세주입에서 생성된 배아 수",
    "이식된 배아 수",
    "미세주입 배아 이식 수",
    "저장된 배아 수",
    "미세주입 후 저장된 배아 수",
    "해동된 배아 수",
    "해동 난자 수",
    "수집된 신선 난자 수",
    "저장된 신선 난자 수",
    "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수",
    "기증자 정자와 혼합된 난자 수",
    "난자 채취 경과일",
    "난자 해동 경과일",
    "난자 혼합 경과일",
    "배아 이식 경과일",
    "배아 해동 경과일"
]

X = X.iloc[:, 52:63] # 본인 관련 데이터만 슬라이싱


#=====================================================================================================================


# 1. 난자 출처 관련 분석

# '난자 출처'별 데이터 개수 세기
egg_total_counts = train['난자 출처'].value_counts()
print("난자 출처 별 데이터 개수")
print(egg_total_counts)
print()

plt.figure(figsize=(5, 5))
sns.barplot(x=egg_total_counts.index, y=egg_total_counts.values, width=0.4, palette="coolwarm")
plt.title("난자 출처 별 데이터 개수", fontsize=14)
plt.xlabel("난자 출처", fontsize=12)
plt.ylabel("개수", fontsize=12)
plt.xticks(rotation=45)


# '난자 출처'로 '임신 성공 여부' 평균(성공률) 계산
egg_success_rate = train.groupby('난자 출처')['임신 성공 여부'].mean()
print("난자 출처 별 성공률")
print(egg_success_rate)
print()

plt.figure(figsize=(5, 5))
sns.barplot(x=egg_success_rate.index, y=egg_success_rate.values, width=0.4)
plt.title("난자 출처 별 성공률", fontsize=14)
plt.xlabel("난자 출처", fontsize=12)
plt.ylabel("성공률", fontsize=12)
plt.xticks(rotation=45)

correlation = train['난자 출처'].corr(train['임신 성공 여부'])
print(f"피어슨 상관 계수: {correlation:.4f}")
print()


# 난자의 경우 기증 받았을 때 성공률이 높은데, 이는 난자를 기증 받았다는 것이
# 원래 임신이 잘 안되는 케이스일 확률이 높으므로 그런 것일듯
# 난자는 기증 받았을 때 기증자의 나이가 더 중요! 이는 세 번째 데이터와 함께 연관 분석 필요



# 2. 정자 출처 관련 분석

# '정자 출처'별 데이터 개수 세기
sperm_total_counts = train['정자 출처'].value_counts()
print("정자 출처 별 데이터 개수")
print(sperm_total_counts)
print()

plt.figure(figsize=(5, 5))
sns.barplot(x=sperm_total_counts.index, y=sperm_total_counts.values, width=0.4, palette="coolwarm")
plt.title("정자 출처 별 데이터 개수", fontsize=14)
plt.xlabel("정자 출처", fontsize=12)
plt.ylabel("개수", fontsize=12)
plt.xticks(rotation=45)

# '정자 출처'로 '임신 성공 여부' 평균(성공률) 계산
sperm_success_rate = train.groupby('정자 출처')['임신 성공 여부'].mean()
print("정자 출처 별 성공률")
print(sperm_success_rate)
print()

plt.figure(figsize=(5, 5))
sns.barplot(x=sperm_success_rate.index, y=sperm_success_rate.values, width=0.4)
plt.title("정자 출처 별 성공률", fontsize=14)
plt.xlabel("정자 출처", fontsize=12)
plt.ylabel("성공률", fontsize=12)
plt.xticks(rotation=45)

correlation = train['정자 출처'].corr(train['임신 성공 여부'])
print(f"피어슨 상관 계수: {correlation:.4f}")
print()

# 또한 정자는 오히려 기증 제공에서 성공률이 떨어지는 모습을 보임



# 3. 난자 기증자 나이 관련 분석

# '난자 기증자 나이'별 데이터 개수 세기
egg_age_total_counts = train['난자 기증자 나이'].value_counts()
print("난자 기증자 나이 별 데이터 개수")
print(egg_age_total_counts)
print()

plt.figure(figsize=(5, 5))
sns.barplot(x=egg_age_total_counts.index, y=egg_age_total_counts.values, width=0.4, palette="coolwarm")
plt.title("난자 기증자 나이 별 데이터 개수", fontsize=14)
plt.xlabel("난자 기증자 나이", fontsize=12)
plt.ylabel("개수", fontsize=12)
plt.xticks(rotation=45)

# '난자 기증자 나이'로 '임신 성공 여부' 평균(성공률) 계산
egg_age_success_rate = train.groupby('난자 기증자 나이')['임신 성공 여부'].mean()
print("난자 기증자 나이 별 성공률")
print(egg_age_success_rate)
print()

plt.figure(figsize=(5, 5))
sns.barplot(x=egg_age_success_rate.index, y=egg_age_success_rate.values, width=0.4, palette="coolwarm")
plt.title("난자 기증자 나이 별 성공률", fontsize=14)
plt.xlabel("난자 기증자 나이", fontsize=12)
plt.ylabel("성공률", fontsize=12)
plt.xticks(rotation=45)

correlation = train['난자 기증자 나이'].corr(train['임신 성공 여부'])
print(f"피어슨 상관 계수: {correlation:.4f}")
print()

# 난자 기증의 경우 나이가 어릴 수록 성공률이 높아야 하는데 오히려 26-30세에서 가장 높은 것은
# 데이터 개수 차이로 인해 그런 것 같다
# 따라서 이러한 개수 차이를 적절히 반영할 필요..라기엔 평균인데
# 사실 26-30세면 오케이?
# 31-35세에 가중치를 덜 부여하는 것이 하나의 방법일 수 있다



# 4. 정자 기증자 나이 관련 분석

# '정자 기증자 나이'별 데이터 개수 세기
sperm_age_total_counts = train['정자 기증자 나이'].value_counts()
print("정자 기증자 나이 별 데이터 개수")
print(sperm_age_total_counts)
print()

plt.figure(figsize=(5, 5))
sns.barplot(x=sperm_age_total_counts.index, y=sperm_age_total_counts.values, width=0.4, palette="coolwarm")
plt.title("정자 기증자 나이 별 데이터 개수", fontsize=14)
plt.xlabel("정자 기증자 나이", fontsize=12)
plt.ylabel("개수", fontsize=12)
plt.xticks(rotation=45)

# '정자 기증자 나이'로 '임신 성공 여부' 평균(성공률) 계산
sperm_age_success_rate = train.groupby('정자 기증자 나이')['임신 성공 여부'].mean()
print("정자 기증자 나이 별 성공률")
print(sperm_age_success_rate)
print()

plt.figure(figsize=(5, 5))
sns.barplot(x=sperm_age_success_rate.index, y=sperm_age_success_rate.values, width=0.4, palette="coolwarm")
plt.title("정자 기증자 나이 별 성공률", fontsize=14)
plt.xlabel("정자 기증자 나이", fontsize=12)
plt.ylabel("성공률", fontsize=12)
plt.xticks(rotation=45)

correlation = train['정자 기증자 나이'].corr(train['임신 성공 여부'])
print(f"피어슨 상관 계수: {correlation:.4f}")
print()




# 5. 배아 종류 별 개수 분석

# 세 가지 feature에 대한 True 값 개수 계산
embryo_counts = train[['동결 배아 사용 여부', '신선 배아 사용 여부', '기증 배아 사용 여부']].sum()
print(train[['동결 배아 사용 여부', '신선 배아 사용 여부', '기증 배아 사용 여부']].sum())
print()

plt.figure(figsize=(5, 5))
ax = sns.barplot(x=embryo_counts.index, y=embryo_counts.values, palette="coolwarm")

# 막대 위에 값 표시
for i, v in enumerate(embryo_counts.values):
    ax.text(i, v + 0.5, str(v), ha='center', fontsize=12, color='black', fontweight='bold')

plt.title("배아 사용 여부 (True 개수)", fontsize=14, pad=20)
plt.xlabel("배아 종류", fontsize=12)
plt.ylabel("True 개수", fontsize=12)
plt.xticks(rotation=45)
plt.show()

# 신선 배아가 대부분임



# 6. 동결 배아 사용 여부 

freeze_egg_success_rate = train.groupby('동결 배아 사용 여부')['임신 성공 여부'].mean()
print("동결 배아 사용 여부 별 성공률")
print(freeze_egg_success_rate)
print()

# 동결 배아는 오히려 사용 시 성공률 감소



# 7. 신선 배아 사용 여부 분석

fresh_egg_success_rate = train.groupby('신선 배아 사용 여부')['임신 성공 여부'].mean()
print("신선 배아 사용 여부 별 성공률")
print(fresh_egg_success_rate)
print()

# 신선 배아의 경우 사용 시 성공률 증가



# 8. 기증 배아 사용 여부 분석

donate_egg_success_rate = train.groupby('기증 배아 사용 여부')['임신 성공 여부'].mean()
print("기증 배아 사용 여부 별 성공률")
print(donate_egg_success_rate)
print()

correlation = train['기증 배아 사용 여부'].corr(train['임신 성공 여부'])
print(f"피어슨 상관 계수: {correlation:.4f}")
print()

# 기증 배아의 경우 사용 시 성공률 대폭 증가



# 9. 대리모 여부 분석

dopa_total_counts = train['대리모 여부'].value_counts()
print("대리모 여부 별 데이터 개수")
print(dopa_total_counts)
print()

plt.figure(figsize=(5, 5))
sns.barplot(x=dopa_total_counts.index, y=dopa_total_counts.values, width=0.4, palette="coolwarm")
plt.title("대리모 여부 별 데이터 개수", fontsize=14)
plt.xlabel("대리모 여부", fontsize=12)
plt.ylabel("개수", fontsize=12)
plt.xticks(rotation=45)

dopa_success_rate = train.groupby('대리모 여부')['임신 성공 여부'].mean()
print("대리모 여부 별 성공률")
print(dopa_success_rate)
print()

plt.figure(figsize=(5, 5))
sns.barplot(x=dopa_success_rate.index, y=dopa_success_rate.values, width=0.4, palette="coolwarm")
plt.title("대리모 여부 별 성공률", fontsize=14)
plt.xlabel("대리모 여부", fontsize=12)
plt.ylabel("성공률", fontsize=12)

# 대리모 경력이 살짝 높으나 유의미하다고 판단되기엔 어려움.. 



# 10. PGD 시술 여부 분석

pgd_total_counts = train['PGD 시술 여부'].value_counts()
print("PGD 시술 여부 별 데이터 개수")
print(pgd_total_counts)
print()

pgd_success_rate = train.groupby('PGD 시술 여부')['임신 성공 여부'].mean()
print("PGD 시술 여부 별 성공률")
print(pgd_success_rate)
print()



# 11. PGS 시술 여부 분석

pgs_total_counts = train['PGS 시술 여부'].value_counts()
print("PGS 시술 여부 별 데이터 개수")
print(pgs_total_counts)
print()

pgs_success_rate = train.groupby('PGS 시술 여부')['임신 성공 여부'].mean()
print("PGS 시술 여부 별 성공률")
print(pgs_success_rate)
print()



# 12. etc

count = ((train['동결 배아 사용 여부'] == True) & (train['신선 배아 사용 여부'] == True)).sum()

print("Both feature1 and feature2 are True:", count)

#=====================================================================================================================