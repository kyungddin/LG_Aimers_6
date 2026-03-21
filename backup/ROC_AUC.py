# ROC-AUC 점수 계산 module
# EDA 완료 후 마지막에 붙여넣으시면 됩니다
# 3번의 학습 모델은 본인이 원하는 모델로 선택 가능합니다
# 4번의 for문의 경우 각자 코드에 맞게 변수명 설정을 해주세요


# 1. 라이브러리 import
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 2. Stratified K-Fold 설정 (데이터 불균형 고려)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

# 3. model 학습 (학습 모델은 자유롭게 선택 가능!)
from lightgbm import LGBMClassifier
model = LGBMClassifier(random_state=42)

# 4. Cross Validation 수행 
# 사용법: for문의 X_train_encoded에 train data set, y에 target 넣기!!
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

# 5. 평균 ROC-AUC 점수 출력
mean_auc = np.mean(auc_scores)
print(f"Mean ROC-AUC Score (5-Fold CV): {mean_auc:.10f}") # dacon과 같이 소수점 10자리 출력!