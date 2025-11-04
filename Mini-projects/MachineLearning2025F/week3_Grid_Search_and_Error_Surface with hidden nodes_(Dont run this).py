#%% =========================================
# Week 3.5 — Two-Node Hidden Layer (10 weights) with r-grid search
#   Inputs: x = [height_norm, weight_norm, waist_norm, color_norm]
#   Hidden node 1: h1 = act( w11*x1 + w12*x2 + w13*x3 + w14*x4 )
#   Hidden node 2: h2 = act( w21*x1 + w22*x2 + w23*x3 + w24*x4 )
#   Score: s = wr1*h1 + wr2*h2
#   Class rule: s < T  -> Healthy, else Unhealthy   (T*: boundary-midpoint)
# Notes:
#   - If USE_ACTIVATION=False (identity), model collapses to Week3 (1-layer linear).
#   - If USE_ACTIVATION=True, it's a genuine 2-layer nonlinear model.
# ============================================

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Config ----------
# (A) Data paths
CSV_PATH_W3 = r"C:\Users\joonc\My_github\My_projects\Mini-projects\DataSets\week3_health_labeled.csv"
CSV_PATH_W1 = r"C:\Users\joonc\My_github\My_projects\Mini-projects\DataSets\week1_health_dataset.csv"
OUT_DIR     = os.path.join(os.path.dirname(CSV_PATH_W1), "outputs_week3p5_two_hidden")
os.makedirs(OUT_DIR, exist_ok=True)

# (B) Search resolution
r_in  = 0.05  # grid for each hidden node's input-weights (simplex over 4 dims)
r_out = 0.05  # grid for (wr1, wr2) with wr1+wr2=1

# (C) Activation toggle
# USE_ACTIVATION = False  # False → Week3와 동일 효과(빠름), True → 진짜 2층(느림)
USE_ACTIVATION = True

print("OUT_DIR:", OUT_DIR, "| USE_ACTIVATION:", USE_ACTIVATION)

#%% =========================================
# Utilities: load/encode/normalize
# ============================================

def to_snake(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[\s/()\-\u200b]+', '_', name)
    name = re.sub(r'__+', '_', name)
    return name.strip('_').lower()

def load_dataset():
    if os.path.exists(CSV_PATH_W3):
        df = pd.read_csv(CSV_PATH_W3)
        print("✅ Loaded Week 3 labeled dataset.")
    else:
        print("ℹ️ Falling back to Week 1 and deriving labels from BMI.")
        assert os.path.exists(CSV_PATH_W1), f"Missing dataset: {CSV_PATH_W1}"
        df = pd.read_csv(CSV_PATH_W1)
        bmi = 703 * df["Weight_lb"] / (df["Height_in"] ** 2)
        df["true_label"] = np.where((bmi >= 18.5) & (bmi <= 24.9), "Healthy", "Unhealthy")
    return df

COLOR_MAP = {"Blue":0, "Red":1, "Green":2, "Yellow":3,
             "Black":4, "Pink":5, "Purple":6, "Orange":7, "White":8}

def encode_color(s: pd.Series) -> pd.Series:
    mapped = s.map(COLOR_MAP)
    unknown_mask = mapped.isna()
    if unknown_mask.any():
        print(f"⚠️ Unknown colors found ({unknown_mask.sum()}):",
              sorted(map(str, s[unknown_mask].unique()))[:8], "...")
    return mapped.fillna(9).astype(int)

def fit_minmax(col: pd.Series):
    return float(col.min()), float(col.max())

def transform_minmax(col: pd.Series, lo: float, hi: float) -> pd.Series:
    if hi > lo:
        z = (col - lo) / (hi - lo)
        return z.clip(0.0, 1.0)
    else:
        return pd.Series(0.0, index=col.index)

#%% =========================================
# Data prep (IDs 1–10 train, 11–12 test)
# ============================================

df = load_dataset()
df.columns = [to_snake(c) for c in df.columns]
required = {"id","height_in","weight_lb","waist_in","favorite_color","true_label"}
missing  = required - set(df.columns)
assert not missing, f"Missing columns: {missing}"

train = df[df["id"].between(1,10)].copy()
test  = df[df["id"].between(11,12)].copy()

train["colorcode"] = encode_color(train["favorite_color"])
test["colorcode"]  = encode_color(test["favorite_color"])

# fit minmax on train, apply to both
h_lo,h_hi = fit_minmax(train["height_in"])
w_lo,w_hi = fit_minmax(train["weight_lb"])
x_lo,x_hi = fit_minmax(train["waist_in"])
c_lo,c_hi = fit_minmax(train["colorcode"])

for d in (train, test):
    d["x1"] = transform_minmax(d["height_in"], h_lo, h_hi)   # height_norm
    d["x2"] = transform_minmax(d["weight_lb"], w_lo, w_hi)   # weight_norm
    d["x3"] = transform_minmax(d["waist_in"],  x_lo, x_hi)   # waist_norm
    d["x4"] = transform_minmax(d["colorcode"],  c_lo, c_hi)  # color_norm

X_train = train[["x1","x2","x3","x4"]].to_numpy()
X_test  = test[["x1","x2","x3","x4"]].to_numpy()
y_train = train["true_label"].to_numpy()
y_test  = test["true_label"].to_numpy()

print("✅ Prepared: train=", len(train), " test=", len(test))

#%% =========================================
# Activation & scoring helpers
# ============================================

def act(z):
    if not USE_ACTIVATION:
        return z  # identity
    # 선택 비선형: ReLU or sqrt; 교육용으로 sqrt가 부드럽고 수월
    # return np.maximum(0.0, z)  # ReLU
    return np.sqrt(np.clip(z, 0.0, None))  # sqrt ReLU

def forward_scores(X, w1_vec, w2_vec, wr1, wr2):
    """
    X: (N,4)
    w1_vec: [w11,w12,w13,w14] (sum=1, >=0)
    w2_vec: [w21,w22,w23,w24] (sum=1, >=0)
    wr1, wr2: wr1+wr2=1, >=0
    returns scores (N,)
    """
    h1_lin = X @ w1_vec  # (N,)
    h2_lin = X @ w2_vec
    h1 = act(h1_lin)
    h2 = act(h2_lin)
    return wr1*h1 + wr2*h2

def best_T_for_scores(score: np.ndarray, y_true: np.ndarray):
    order = np.argsort(score)
    s_sorted = score[order]
    y_sorted = y_true[order]
    eps = 1e-9
    mids = [s_sorted[0]-eps] + [0.5*(s_sorted[i]+s_sorted[i+1]) for i in range(len(s_sorted)-1)] + [s_sorted[-1]+eps]
    candidates = np.unique(np.asarray(mids))

    best_acc, best_T = -1.0, None
    for T in candidates:
        pred = np.where(s_sorted < T, "Healthy", "Unhealthy")
        acc  = float(np.mean(pred == y_sorted))
        if acc > best_acc or (acc == best_acc and (best_T is None or T < best_T)):
            best_acc, best_T = acc, T
    return best_T, best_acc

#%% =========================================
# Grid utilities over simplex (4 weights sum to 1) and pair (wr1+wr2=1)
# ============================================

def simplex_grid_4(r):
    """
    Yield 4D nonnegative tuples summing to 1 with step r.
    Count = C(M+3,3) where M=1/r.
    """
    M = int(round(1.0 / r))
    assert abs(M*r - 1.0) < 1e-9
    for k1 in range(M+1):
        for k2 in range(M - k1 + 1):
            for k3 in range(M - k1 - k2 + 1):
                k4 = M - k1 - k2 - k3
                yield np.array([k1, k2, k3, k4], dtype=float) * r

def pair_grid_2(r):
    M = int(round(1.0 / r))
    assert abs(M*r - 1.0) < 1e-9
    for k in range(M+1):
        yield (k*r, (M-k)*r)

#%% =========================================
# Core search
#  - Fast path (identity): collapse to Week3 by α = wr1*w1 + wr2*w2 ⇒ α also simplex,
#    so best is just Week3 search over α. (우린 10가중치 중 등가표현도 함께 리포트)
#  - Nonlinear path: 실제 10가중치 그리드(완전탐색은 비현실) → 교육용으로 거친 r_in, r_out로 실습
# ============================================

best = {"acc": -1.0, "w11":None,"w12":None,"w13":None,"w14":None,
        "w21":None,"w22":None,"w23":None,"w24":None, "wr1":None,"wr2":None, "T":None}

if not USE_ACTIVATION:
    # -------- Fast path: Equivalent to Week3 (collapse) --------
    # α = wr1*w1 + wr2*w2 이고 wr가 (1,0)일 수 있으므로 α는 단순히 w1의 어떤 점.
    # 따라서 4-simplex 그리드만 탐색하면 충분.
    for a in simplex_grid_4(r_in):
        scores = X_train @ a  # (N,)
        T_star, acc_star = best_T_for_scores(scores, y_train)
        if acc_star > best["acc"]:
            # 등가 10가중치로 리포트: wr1=1, wr2=0, w1=a, w2=[1,0,0,0] (dummy)
            best = {"acc": float(acc_star),
                    "w11":a[0],"w12":a[1],"w13":a[2],"w14":a[3],
                    "w21":1.0,"w22":0.0,"w23":0.0,"w24":0.0,
                    "wr1":1.0,"wr2":0.0, "T": float(T_star)}
else:
    # -------- Nonlinear path: genuine 2-layer (coarse search) --------
    for w1_vec in simplex_grid_4(r_in):
        for w2_vec in simplex_grid_4(r_in):
            for (wr1, wr2) in pair_grid_2(r_out):
                scores = forward_scores(X_train, w1_vec, w2_vec, wr1, wr2)
                T_star, acc_star = best_T_for_scores(scores, y_train)
                if acc_star > best["acc"]:
                    best = {"acc": float(acc_star),
                            "w11":w1_vec[0],"w12":w1_vec[1],"w13":w1_vec[2],"w14":w1_vec[3],
                            "w21":w2_vec[0],"w22":w2_vec[1],"w23":w2_vec[2],"w24":w2_vec[3],
                            "wr1":wr1,"wr2":wr2, "T": float(T_star)}

print("✅ Best TRAIN acc:", best["acc"])
print("✅ Weights:",
      (best["w11"],best["w12"],best["w13"],best["w14"]),
      (best["w21"],best["w22"],best["w23"],best["w24"]),
      (best["wr1"],best["wr2"]))
print("✅ Best T*:", best["T"])

#%% =========================================
# Save TRAIN details
# ============================================

# Forward with best weights
w1_star = np.array([best["w11"],best["w12"],best["w13"],best["w14"]], dtype=float)
w2_star = np.array([best["w21"],best["w22"],best["w23"],best["w24"]], dtype=float)
wr1, wr2 = float(best["wr1"]), float(best["wr2"])
T_best   = float(best["T"])

train_scores = forward_scores(X_train, w1_star, w2_star, wr1, wr2)
train_pred   = np.where(train_scores < T_best, "Healthy", "Unhealthy")
train_out = train.copy()
train_out["score_best"] = train_scores
train_out["pred_best"]  = train_pred
train_out["correct"]    = (train_out["pred_best"] == train_out["true_label"]).astype(int)

train_csv = os.path.join(OUT_DIR, f"week3p5_train_preds_USEACT_{int(USE_ACTIVATION)}.csv")
cols = ["id","height_in","weight_lb","waist_in","favorite_color",
        "x1","x2","x3","x4","score_best","pred_best","true_label","correct"]
train_out[cols].to_csv(train_csv, index=False)
print("💾 Saved TRAIN preds:", train_csv)

summary_path = os.path.join(OUT_DIR, f"week3p5_summary_USEACT_{int(USE_ACTIVATION)}.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=== Week3.5 Two-Node Hidden (10 weights) ===\n")
    f.write(f"USE_ACTIVATION={USE_ACTIVATION}\n")
    f.write(f"Best TRAIN acc: {best['acc']:.3f}\n")
    f.write(f"Hidden1 (w11..w14): {tuple(np.round(w1_star,4))}\n")
    f.write(f"Hidden2 (w21..w24): {tuple(np.round(w2_star,4))}\n")
    f.write(f"Output  (wr1,wr2):  ({wr1:.4f},{wr2:.4f})\n")
    f.write(f"Best T*: {T_best:.6f}\n")
print("💾 Saved summary:", summary_path)

#%% =========================================
# TEST evaluation
# ============================================

test_scores = forward_scores(X_test, w1_star, w2_star, wr1, wr2)
test_pred   = np.where(test_scores < T_best, "Healthy", "Unhealthy")

test_out = test.copy()
test_out["score_best"] = test_scores
test_out["pred_best"]  = test_pred

test_csv = os.path.join(OUT_DIR, f"week3p5_test_preds_USEACT_{int(USE_ACTIVATION)}.csv")
test_out[["id","height_in","weight_lb","waist_in","favorite_color",
          "x1","x2","x3","x4","score_best","pred_best","true_label"]].to_csv(test_csv, index=False)
print("💾 Saved TEST preds:", test_csv)

# 간단 지표
labels = ["Healthy","Unhealthy"]
def cm_simple(y_true, y_pred, labels):
    idx = {lab:i for i,lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t,p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm

cm = cm_simple(y_test, test_pred, labels)
acc_test = (np.trace(cm) / cm.sum()) if cm.sum()>0 else 0.0
print("\n📊 TEST Confusion Matrix (rows=true, cols=pred):")
print(pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels]))
print(f"\n🧾 TEST accuracy: {acc_test:.3f}")
