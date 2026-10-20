#%% =========================================
# Week 1 — Simple Linear Classifier (LOCAL, underscore headers)
# =========================================
import os, re
import numpy as np
import pandas as pd
from pathlib import Path
try:
    from IPython.display import display
except ImportError:
    display = print

# 데이터 경로
CSV_PATH = str(Path(os.environ.get("MY_PROJECTS_DATA_DIR", Path(__file__).resolve().parents[1] / "DataSets")) / "week1_health_dataset.csv")
OUT_DIR  = os.path.join(os.path.dirname(CSV_PATH), "outputs_week1")
os.makedirs(OUT_DIR, exist_ok=True)

print("CSV_PATH:", CSV_PATH)

#%% =========================================
# Task 1. Load Dataset + 표준화된 컬럼명(snake_case) 만들기
# =========================================
assert os.path.exists(CSV_PATH), f"❌ 파일 없음: {CSV_PATH}"
df = pd.read_csv(CSV_PATH)

def to_snake(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[\s/()\-\u200b]+', '_', name)  # 공백/괄호/하이픈 등 → _
    name = re.sub(r'__+', '_', name)               # 중복 _ 정리
    return name.strip('_').lower()

df.columns = [to_snake(c) for c in df.columns]
print("✅ Columns:", list(df.columns))

# 기대 컬럼 체크 (언더바 버전)
needed = {'id','height_in','weight_lb','waist_in','favorite_color'}
missing = needed - set(df.columns)
assert not missing, f"❌ 누락 컬럼: {missing}. CSV 헤더 확인 필요."

display(df.head())

#%% =========================================
# Task 2. Assign Weights & Color Encoding
#  - w1: height_in, w2: weight_lb, w3: waist_in, w4: colorcode  (합=1)
# =========================================
w1, w2, w3, w4 = 0.10, 0.10, 0.10, 0.70
assert abs((w1+w2+w3+w4) - 1.0) < 1e-9, "weights 합이 1이 되게 설정하세요."
print(f"Weights = {w1, w2, w3, w4}")

color_map = {
    'Blue': 0, 'Red': 1, 'Green': 2, 'Yellow': 3,
    'Black': 4, 'Pink': 5, 'Purple': 6, 'Orange': 7, 'White': 8
}
df['colorcode'] = df['favorite_color'].map(color_map)

# 미정의 색상 자동 코드 부여
if df['colorcode'].isna().any():
    next_code = (max(color_map.values()) if color_map else -1) + 1
    df['colorcode'] = df['colorcode'].fillna(next_code).astype(int)

display(df[['id','favorite_color','colorcode']])

#%% =========================================
# Task 3. Score (Before Normalization)
# =========================================
df['score_raw'] = (
    w1 * df['height_in'] +
    w2 * df['weight_lb'] +
    w3 * df['waist_in'] +
    w4 * df['colorcode']
)
print("✅ Raw score 계산 완료")
display(df[['id','score_raw']])

#%% =========================================
# Task 4. Normalization (Min–Max) & Recompute Score
# =========================================
def minmax(col: pd.Series) -> pd.Series:
    lo, hi = col.min(), col.max()
    return (col - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=col.index)

for c in ['height_in','weight_lb','waist_in','colorcode']:
    df[f'{c}_norm'] = minmax(df[c])

df['score_norm'] = (
    w1 * df['height_in_norm'] +
    w2 * df['weight_lb_norm'] +
    w3 * df['waist_in_norm'] +
    w4 * df['colorcode_norm']
)
print("⚖️ Normalized score 계산 완료")
display(df[['id','score_raw','score_norm']])

#%% =========================================
# Task 5. Classification (Threshold T)
# =========================================
T = 0.50
df['pred_label'] = np.where(df['score_norm'] < T, 'Healthy', 'Unhealthy')
print(f"✅ Classification 완료 (T={T})")
display(df[['id','score_norm','pred_label']])

# 저장
out_csv = os.path.join(OUT_DIR, "week1_predictions.csv")
df.to_csv(out_csv, index=False)
print("💾 Saved:", out_csv)

#%% =========================================
# (보조) Train/Test split & summary
# =========================================
train = df[df['id'].between(1,10)].copy()
test  = df[df['id'].between(11,12)].copy()

print("📘 TRAIN (1–10)")
display(train[['id','score_norm','pred_label']])

print("🧪 TEST (11–12)")
display(test[['id','score_norm','pred_label']])

#%% =========================================
# (보조) Reflection: feature 영향도(상관)
# =========================================
corrs = df[['score_norm',
            'height_in_norm','weight_lb_norm','waist_in_norm','colorcode_norm']].corr()['score_norm'].drop('score_norm')
print("📈 Corr(feature_norm, score_norm)")
display(corrs.to_frame('corr'))

#%% =========================================
# (선택) Challenge skeleton — Grid Search (true label 생기면 사용)
# =========================================
"""
best = {'acc': -1, 'w': None, 'T': None}
grid = np.arange(0, 1.01, 0.1)

for a in grid:
    for b in grid:
        for c in grid:
            d = 1 - (a+b+c)
            if d < -1e-9: 
                continue
            score_tr = (a*train['height_in_norm'] +
                        b*train['weight_lb_norm'] +
                        c*train['waist_in_norm'] +
                        d*train['colorcode_norm'])
            for TT in np.arange(0.0, 1.01, 0.05):
                pred = np.where(score_tr < TT, 'Healthy', 'Unhealthy')
                acc = (pred == train['true_label']).mean()
                if acc > best['acc']:
                    best = {'acc': float(acc), 'w': (float(a),float(b),float(c),float(d)), 'T': float(TT)}
print("🏆 Best:", best)
"""
