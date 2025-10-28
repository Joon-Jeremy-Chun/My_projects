#%% =========================================
# Week 3 — r-grid search with per-weight optimal T
#   - r is the grid step (e.g., 0.05)
#   - For each (w1..w4), compute train scores, find optimal T from N+1 boundary candidates
#   - Report best (w1,w2,w3,w4,T*, accuracy) + Top-K, Test eval, Heatmaps
# ============================================

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Config ----------------
r = 0.05  # grid resolution for weights (e.g., 0.05 -> k sums to 20)
CSV_PATH_W3 = r"C:\Users\joonc\My_github\My_projects\Mini-projects\DataSets\week3_health_labeled.csv"
CSV_PATH_W1 = r"C:\Users\joonc\My_github\My_projects\Mini-projects\DataSets\week1_health_dataset.csv"
OUT_DIR     = os.path.join(os.path.dirname(CSV_PATH_W1), "outputs_week3_rgrid")
os.makedirs(OUT_DIR, exist_ok=True)

print("Output dir:", OUT_DIR)

#%% =========================================
# Utilities: naming, dataset load, encoding, normalization
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
        print("ℹ️ Falling back to Week 1 and deriving labels from BMI (hidden).")
        assert os.path.exists(CSV_PATH_W1), f"Missing dataset: {CSV_PATH_W1}"
        df = pd.read_csv(CSV_PATH_W1)
        bmi = 703 * df["Weight_lb"] / (df["Height_in"] ** 2)
        df["true_label"] = np.where((bmi >= 18.5) & (bmi <= 24.9), "Healthy", "Unhealthy")
    return df

COLOR_MAP = {
    "Blue":0, "Red":1, "Green":2, "Yellow":3,
    "Black":4, "Pink":5, "Purple":6, "Orange":7, "White":8
}

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
        return z.clip(lower=0.0, upper=1.0)
    else:
        return pd.Series(0.0, index=col.index)

#%% =========================================
# Scoring / Prediction / Metrics helpers
# ============================================

def compute_score(dfp: pd.DataFrame, w):
    # w = (w1,w2,w3,w4) -> (height, weight, waist, color)
    w1,w2,w3,w4 = w
    return (w1*dfp["height_norm"] + w2*dfp["weight_norm"] +
            w3*dfp["waist_norm"]  + w4*dfp["color_norm"])

def predict_from_score(score: pd.Series, T: float):
    # rule: score < T -> Healthy, else Unhealthy
    return np.where(score < T, "Healthy", "Unhealthy")

def best_T_for_scores(score: np.ndarray, y_true: np.ndarray):
    """
    Given scores (N,) and string labels ('Healthy'/'Unhealthy'),
    find T* maximizing accuracy by testing only boundary midpoints:
      T0 = s(1)-eps, Ti=(s(i)+s(i+1))/2, TN = s(N)+eps
    Returns: (best_T, best_acc)
    """
    order = np.argsort(score)
    s_sorted = score[order]
    y_sorted = y_true[order]

    eps = 1e-9
    # build midpoint candidates (dedup)
    mids = [s_sorted[0] - eps]
    mids += [0.5*(s_sorted[i] + s_sorted[i+1]) for i in range(len(s_sorted)-1)]
    mids.append(s_sorted[-1] + eps)
    candidates = np.unique(np.asarray(mids))

    best_acc, best_T = -1.0, None
    for T in candidates:
        pred = np.where(s_sorted < T, "Healthy", "Unhealthy")
        acc  = float(np.mean(pred == y_sorted))
        # tie-break: prefer smaller T
        if acc > best_acc or (acc == best_acc and (best_T is None or T < best_T)):
            best_acc, best_T = acc, T
    return best_T, best_acc

def confusion_matrix_simple(y_true, y_pred, labels):
    idx = {lab:i for i,lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t,p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm

def classification_report_simple(y_true, y_pred, labels, digits=3):
    cm = confusion_matrix_simple(y_true, y_pred, labels)
    support = cm.sum(axis=1)
    precision = []
    recall = []
    f1 = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_i = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        precision.append(prec); recall.append(rec); f1.append(f1_i)
    acc = (np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0

    # Build text report
    lines = []
    lines.append("precision   recall  f1-score   support")
    for i,lab in enumerate(labels):
        lines.append(f"{lab:10s} {precision[i]:9.{digits}f} {recall[i]:7.{digits}f} {f1[i]:9.{digits}f} {support[i]:9d}")
    lines.append("")
    lines.append(f"accuracy: {acc:.{digits}f}  total={cm.sum()}")
    return "\n".join(lines), cm

#%% =========================================
# Load & prep (train/test split, normalization)
# ============================================

df = load_dataset()
df.columns = [to_snake(c) for c in df.columns]
required = {"id","height_in","weight_lb","waist_in","favorite_color","true_label"}
missing  = required - set(df.columns)
assert not missing, f"Missing columns: {missing}"

# Split: IDs 1~10 train, 11~12 test
train = df[df["id"].between(1,10)].copy()
test  = df[df["id"].between(11,12)].copy()

train["colorcode"] = encode_color(train["favorite_color"])
test["colorcode"]  = encode_color(test["favorite_color"])

# normalize (fit on train only)
h_lo,h_hi = fit_minmax(train["height_in"])
w_lo,w_hi = fit_minmax(train["weight_lb"])
x_lo,x_hi = fit_minmax(train["waist_in"])
c_lo,c_hi = fit_minmax(train["colorcode"])

for d in (train, test):
    d["height_norm"] = transform_minmax(d["height_in"], h_lo, h_hi)
    d["weight_norm"] = transform_minmax(d["weight_lb"], w_lo, w_hi)
    d["waist_norm"]  = transform_minmax(d["waist_in"],  x_lo, x_hi)
    d["color_norm"]  = transform_minmax(d["colorcode"],  c_lo, c_hi)

print("✅ Train/Test prepared:",
      f"train={len(train)}, test={len(test)}")

#%% =========================================
# r-grid over weights -> Best (w, T) + Top-K list
# ============================================

M = int(round(1.0 / r))
assert abs(M * r - 1.0) < 1e-9, "r must divide 1 (e.g., 0.1, 0.05, 0.02, ...)"

y_train = train["true_label"].to_numpy()

topK = []  # (acc, w_tuple, T)
for k1 in range(M+1):
    for k2 in range(M - k1 + 1):
        for k3 in range(M - k1 - k2 + 1):
            k4 = M - k1 - k2 - k3
            w  = np.array([k1, k2, k3, k4], dtype=float) * r  # sums to exactly 1
            score = compute_score(train, w)
            T_star, acc_star = best_T_for_scores(score.to_numpy(), y_train)
            topK.append((float(acc_star), tuple(np.round(w,4)), float(T_star)))

# 정렬 후 Top-15 저장
topK.sort(key=lambda x: (-x[0], x[1]))
topK15 = topK[:15]
top_csv = os.path.join(OUT_DIR, f"week3_top15_r{r:.2f}.csv")
pd.DataFrame([{"acc":a, "w1":w[0],"w2":w[1],"w3":w[2],"w4":w[3],"T":t} for a,w,t in topK15]).to_csv(top_csv, index=False)
print("💾 Saved Top-15 combos:", top_csv)

# 최상위 선택
best_acc, best_w, best_T = topK15[0]
best = {"acc": best_acc, "w": best_w, "T": best_T}

print("✅ Best train accuracy:", best["acc"])
print("✅ Best weights (w1,w2,w3,w4):", best["w"])
print("✅ Best threshold T*:", best["T"])

#%% =========================================
# Save summary + train predictions at best
# ============================================

w1,w2,w3,w4 = best["w"]
T_best      = best["T"]

train["score_best"] = compute_score(train, (w1,w2,w3,w4))
train["pred_best"]  = predict_from_score(train["score_best"], T_best)
train["correct"]    = (train["pred_best"] == train["true_label"]).astype(int)

summary_path = os.path.join(OUT_DIR, f"week3_best_summary_r{r:.2f}.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"r-grid step: {r}\n")
    f.write(f"Best train accuracy: {best['acc']:.3f}\n")
    f.write(f"Best weights (w1,w2,w3,w4): {best['w']}\n")
    f.write(f"Best threshold T*: {best['T']:.6f}\n")
print("💾 Saved summary:", summary_path)

out_csv = os.path.join(OUT_DIR, f"week3_train_best_r{r:.2f}.csv")
train[["id","height_in","weight_lb","waist_in","favorite_color",
       "score_best","pred_best","true_label","correct"]].to_csv(out_csv, index=False)
print("💾 Saved train predictions:", out_csv)

#%% =========================================
# Test evaluation (confusion matrix + simple report)
# ============================================

test["score_best"] = compute_score(test, (w1,w2,w3,w4))
test["pred_best"]  = predict_from_score(test["score_best"], T_best)

labels_order = ["Healthy","Unhealthy"]
report_text, cm = classification_report_simple(
    test["true_label"].to_numpy(),
    test["pred_best"].to_numpy(),
    labels_order,
    digits=3
)

print("\n📊 Test Confusion Matrix (rows=true, cols=pred):")
cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels_order],
                        columns=[f"pred_{l}" for l in labels_order])
print(cm_df)
print("\n🧾 Test Classification Report:")
print(report_text)

test_out_csv = os.path.join(OUT_DIR, f"week3_test_best_r{r:.2f}.csv")
test[["id","height_in","weight_lb","waist_in","favorite_color","score_best","pred_best","true_label"]].to_csv(test_out_csv, index=False)
print("💾 Saved test predictions:", test_out_csv)

with open(summary_path, "a", encoding="utf-8") as f:
    f.write("\n--- TEST EVAL ---\n")
    f.write(report_text + "\n")
    f.write("Confusion Matrix (rows=true, cols=pred):\n")
    f.write(cm_df.to_csv(index=True))
print("📝 Appended test metrics to summary.")

#%% =========================================
# Heatmaps: error surfaces for (w1,w2), (w2,w3), (w3,w4)
#   - At each (w_i, w_j), remaining mass is redistributed to the other two
#     in proportion to best weights from the grid search.
#   - Per-point optimal T* on train set.
# ============================================

# use best weights/T from grid
w_best = np.array([w1, w2, w3, w4], dtype=float)
T_best_report = T_best  # annotate

def redistrib_two(rem, fixed_pair_idx):
    """
    rem: remaining mass to split among the two 'other' weights
    fixed_pair_idx: indices of the pair we are varying (e.g., (0,1) for w1,w2)
    Returns: (other_idx, values) for remaining two weights, preserving w_best ratio.
    """
    all_idx = np.arange(4)
    other_idx = all_idx[~np.isin(all_idx, fixed_pair_idx)]
    other_vals = w_best[other_idx]
    s = other_vals.sum()
    ratios = other_vals / s if s > 0 else np.ones_like(other_vals) / len(other_vals)
    return other_idx, rem * ratios

def heatmap_for_pair(idx_i, idx_j, label_i, label_j, r_plot=0.02):
    grid = np.arange(0.0, 1.0 + 1e-9, r_plot)
    acc_map = np.full((len(grid), len(grid)), np.nan, dtype=float)

    for ai, wi in enumerate(grid):
        for aj, wj in enumerate(grid):
            rem = 1.0 - (wi + wj)
            if rem < -1e-12:  # invalid (sum>1)
                continue
            other_idx, others = redistrib_two(max(0.0, rem), (idx_i, idx_j))
            w_try = w_best.copy()
            w_try[idx_i] = wi
            w_try[idx_j] = wj
            w_try[other_idx] = others

            score = compute_score(train, w_try)
            T_star, acc_star = best_T_for_scores(score.to_numpy(), train["true_label"].to_numpy())
            acc_map[aj, ai] = acc_star  # note: rows->w_j, cols->w_i

    err_map = 1.0 - acc_map
    plt.figure(figsize=(8,6))
    im = plt.imshow(
        err_map,
        origin="lower",
        aspect="auto",
        extent=[grid.min(), grid.max(), grid.min(), grid.max()],
    )
    plt.colorbar(im, label="Error (1 - Accuracy)")
    plt.xlabel(f"{label_i} (varied)")
    plt.ylabel(f"{label_j} (varied)")
    plt.title(
        f"Error surface for {label_i} vs {label_j}\n"
        f"(others rescaled; per-point optimal T*)\n"
        f"best(w)={tuple(np.round(w_best,3))}, T*={T_best_report:.4f}"
    )
    # Mark best projection
    plt.scatter([w_best[idx_i]],[w_best[idx_j]], s=120, c="white", marker="x", linewidths=2)
    plt.text(w_best[idx_i], w_best[idx_j], " best(w)", color="white", va="bottom", ha="left")

    out_png = os.path.join(OUT_DIR, f"week3_heatmap_{label_i}_{label_j}_r{r_plot:.2f}.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
    print("🖼️ Saved:", out_png)

# Generate the requested pair plots
heatmap_for_pair(0,1,"w1","w2", r_plot=0.02)
heatmap_for_pair(1,2,"w2","w3", r_plot=0.02)
heatmap_for_pair(2,3,"w3","w4", r_plot=0.02)
print("✅ 2D heatmaps done.")
