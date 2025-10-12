# %## =========================================
# Week 2 — Simple Nonlinear Node (Single Hidden Layer, 2 Hidden Nodes)
#  - Inputs: Height, Weight (normalized to 0~1)
#  - Hidden nodes: h1(x1) = -x1 + 1, h2(x2) = x2
#  - Output: Score = a1*h1 + a2*h2  (a1,a2 ≥ 0, a1+a2=1), Label via threshold T
#  - Format aligned with Week 1 script style
# =========================================

#%% Imports & Paths
import os, re
import numpy as np
import pandas as pd

# 기본 데이터 경로 설정 (동일 폴더 우선 → /mnt/data 폴백)
CANDIDATES = [
    os.path.join(os.getcwd(), "week1_health_dataset.csv"),
    "/mnt/data/week1_health_dataset.csv"
]
CSV_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
assert CSV_PATH is not None, "❌ 데이터 파일 week1_health_dataset.csv 을(를) 찾을 수 없습니다."

OUT_DIR  = os.path.join(os.path.dirname(CSV_PATH), "outputs_week2")
os.makedirs(OUT_DIR, exist_ok=True)

print("CSV_PATH:", CSV_PATH)
print("OUT_DIR :", OUT_DIR)

# %## Load Dataset & Normalize Headers (snake_case)
df = pd.read_csv(CSV_PATH)

def to_snake(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[\s/()\-\u200b]+', '_', name)
    name = re.sub(r'__+', '_', name)
    return name.strip('_').lower()

df.columns = [to_snake(c) for c in df.columns]
print("✅ Columns:", list(df.columns))
#%%
# 이번 주는 Height/Weight만 사용
needed = {'id','height_in','weight_lb'}
missing = needed - set(df.columns)
assert not missing, f"❌ 누락 컬럼: {missing}. CSV 헤더 확인 필요."

# %## Train/Test Split (Week 1과 동일 규칙: 1–10 train, 11–12 test)
train_mask = df['id'].between(1, 10)
test_mask  = df['id'].between(11, 12)

train = df.loc[train_mask].copy()
test  = df.loc[test_mask].copy()

print(f"Train size: {len(train)}  | Test size: {len(test)}")
#%%
# Min–Max Normalization (fit on train only)
def minmax_fit(series: pd.Series):
    lo, hi = float(series.min()), float(series.max())
    return lo, hi

def minmax_transform(series: pd.Series, lo: float, hi: float):
    return (series - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=series.index)

h_lo, h_hi = minmax_fit(train['height_in'])
w_lo, w_hi = minmax_fit(train['weight_lb'])

train['height_norm'] = minmax_transform(train['height_in'], h_lo, h_hi)
train['weight_norm'] = minmax_transform(train['weight_lb'], w_lo, w_hi)

# test는 train의 파라미터로 변환
test['height_norm'] = minmax_transform(test['height_in'], h_lo, h_hi)
test['weight_norm'] = minmax_transform(test['weight_lb'], w_lo, w_hi)

# 0~1 범위 체크(정보용)
for nm in ['height_norm','weight_norm']:
    lo, hi = float(train[nm].min()), float(train[nm].max())
    print(f"{nm}: train range ~ [{lo:.3f}, {hi:.3f}]")

#%% Define Hidden Nodes (Nonlinear Design)
# Node_11: h1(x1) = -x1 + 1 (키가 커질수록 감소 → 분모 효과)
# Node_21: h2(x2) =  x2     (몸무게가 클수록 증가 → 분자 효과)
def h1_height(x1: pd.Series) -> pd.Series:
    return -x1 + 1.0

def h2_weight(x2: pd.Series) -> pd.Series:
    return x2

# %## Output Score = a1*h1 + a2*h2  (a1+a2=1)
# 기본 가중치 (필요 시 변경)
a1, a2 = 0.5, 0.5
assert abs((a1 + a2) - 1.0) < 1e-9, "a1+a2=1 이어야 합니다."
T = 0.50  # 기본 임계값 (필요 시 변경)

print(f"Weights (a1,a2) = ({a1:.2f}, {a2:.2f}) | Threshold T = {T:.2f}")

#%% Compute Score & Predict (Train/Test)
def compute_score(df_part: pd.DataFrame, a1: float, a2: float) -> pd.Series:
    x1 = df_part['height_norm']
    x2 = df_part['weight_norm']
    z  = a1 * h1_height(x1) + a2 * h2_weight(x2)
    return z

def predict_label(score: pd.Series, T: float) -> pd.Series:
    return np.where(score < T, 'Healthy', 'Unhealthy')

train['score_week2'] = compute_score(train, a1, a2)
train['pred_label']  = predict_label(train['score_week2'], T)

test['score_week2'] = compute_score(test, a1, a2)
test['pred_label']  = predict_label(test['score_week2'], T)

print("✅ Score & Prediction 완료")

# %## Save Outputs
out_train_csv = os.path.join(OUT_DIR, "week2_train_predictions.csv")
out_test_csv  = os.path.join(OUT_DIR, "week2_test_predictions.csv")
train[['id','height_in','weight_lb','height_norm','weight_norm','score_week2','pred_label']].to_csv(out_train_csv, index=False)
test[['id','height_in','weight_lb','height_norm','weight_norm','score_week2','pred_label']].to_csv(out_test_csv, index=False)

print("💾 Saved:", out_train_csv)
print("💾 Saved:", out_test_csv)

#%% (Optional) Grid Search for (a1,a2,T)
#  - a1 ∈ {0.0, 0.1, …, 1.0}, a2 = 1 - a1
#  - T ∈ {0.30, 0.35, …, 0.70}
#  - 정확도 계산을 위해서는 true_label 이 필요 (Week 1 제출 후 공개 가정)
#  - 여기서는 true_label 컬럼이 있다면 동작하도록 구현 (없으면 스킵)
if 'true_label' in train.columns:
    grid_a1 = np.round(np.linspace(0.0, 1.0, 11), 2)
    grid_T  = np.round(np.linspace(0.30, 0.70, 9), 2)

    best = {'acc': -1, 'a1': None, 'a2': None, 'T': None}
    for a1_try in grid_a1:
        a2_try = 1.0 - a1_try
        sc = compute_score(train, a1_try, a2_try)
        for T_try in grid_T:
            pred = predict_label(sc, T_try)
            acc = np.mean(pred == train['true_label'])
            if acc > best['acc']:
                best = {'acc': acc, 'a1': a1_try, 'a2': a2_try, 'T': T_try}

    print(f"🔎 Best on Train: acc={best['acc']:.3f}, a1={best['a1']:.2f}, a2={best['a2']:.2f}, T={best['T']:.2f}")

    # 베스트 파라미터로 train/test 예측 저장
    train['score_best'] = compute_score(train, best['a1'], best['a2'])
    train['pred_best']  = predict_label(train['score_best'], best['T'])

    test['score_best'] = compute_score(test, best['a1'], best['a2'])
    test['pred_best']  = predict_label(test['score_best'], best['T'])

    out_train_best = os.path.join(OUT_DIR, "week2_train_best.csv")
    out_test_best  = os.path.join(OUT_DIR, "week2_test_best.csv")
    train[['id','score_best','pred_best']].to_csv(out_train_best, index=False)
    test[['id','score_best','pred_best']].to_csv(out_test_best, index=False)

    print("💾 Saved:", out_train_best)
    print("💾 Saved:", out_test_best)
else:
    print("ℹ️ 'true_label' 컬럼이 없어 Grid Search는 스킵합니다. (제출 후 공개 가정)")
