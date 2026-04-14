# LG Aimers 6th : 난임 환자 대상 임신 성공 확률 예측
Open-source Data 기반 맞춤형 임신 성공 여부 분류 모델 개발


## 1. Project Overview
- 본 프로젝트는 난임 환자의 임상 데이터를 분석하여 임신 성공 가능성을 예측하는 머신러닝 모델을 개발하는 것을 목표로 합니다. LG Aimers 6기 AI 해커톤 과제로 수행되었습니다.

- 주제: 환자별 특성에 따른 맞춤형 임신 성공 여부 분류 및 확률 예측

- 핵심 과제: 데이터 불균형 해소, 주요 피처 엔지니어링, 앙상블 모델 최적화

---


## 2. Repository Structure
```

team_meeting_docs/ : 프로젝트 진행 과정 및 회의록 기록

backup/ : 코드 버전 관리 및 백업 데이터

train.csv / test.csv : 모델 학습 및 평가용 데이터셋

main.ipynb : 데이터 전처리, 시각화 및 메인 모델링 과정

최종명세.xlsx : 분석 변수 정의 및 최종 모델 명세서

final_submission.csv : 최종 예측 결과 제출 파일

```

---


## 3. Tech Stack & Methodology
- Language: Python

- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

- Key Processes:
    - 데이터 정제 및 결측값 처리
    - 피처 중요도(Feature Importance) 분석을 통한 변수 선정
    - 임신 성공 확률 최적화를 위한 하이퍼파라미터 튜닝

---

## 4. Results
- ROC-AUC Score: 0.7412452582
    - Baseline Score: 0.6890035245

---

## 5. How to Run
리포지토리 복제:

```Bash
git clone https://github.com/kyungddin/LG_Aimers_6.git
```

---
