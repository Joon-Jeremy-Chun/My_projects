# %% =========================================
# Week 3 — Grid Search & Error Surface (Linear Model, Accuracy on Train Only)
#  - Uses 4 inputs (Height, Weight, Waist, ColorCode)
#  - Finds (w1,w2,w3,w4,T) by exhaustive grid search on train (IDs 1–10)
#  - Generates 2D error-surface heatmaps: (w_k, T) with other weights rescaled to keep sum=1
# ============================================

# %% Imports & Paths
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Primary (labeled) and fallback (raw) dataset paths
CSV_PATH_W3 = r"C:\Users\joonc\My_github\My_projects\Mini-projects\DataSets\week3_health_labeled.csv"
CSV_PATH_W1 = r"C:\Users\joonc\My_github\My_projects\Mini-projects\DataSets\week1_health_dataset.csv"

OUT_DIR  = os.path.join(os.path.dirname(CSV_PATH_W1), "outputs_week3")
os.makedirs(OUT_DIR, exist_ok=True)

print("Primary CSV (Week 3 labeled):", CSV_PATH_W3)
print("Fallback  CSV (Week 1 raw)  :", CSV_PATH_W1)

# %% -----------------------------------------
# Helpers
# --------------------------------------------
def to_snake(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[\s/()\-\u200b]+', '_', name)  # spaces, slashes, etc. -> _
    name = re.sub(r'__+', '_', name)
    return name.strip('_').lower()

def load_dataset():
    """Load labeled dataset if present, otherwise derive hidden labels from BMI."""
    if os.path.exists(CSV_PATH_W3):
        df = pd.read_csv(CSV_PATH_W3)
        print("✅ Loaded Week 3 labeled dataset.")
    else:
        print("ℹ️ Week 3 labeled dataset not found. Falling back to Week 1 and deriving labels (hidden).")
        assert os.path.exists(CSV_PATH_W1), f"Missing dataset: {CSV_PATH_W1}"
        df = pd.read_csv(CSV_PATH_W1)
        # Hidden label from BMI (students won't see this step conceptually)
        bmi = 703 * df["Weight_lb"] / (df["Height_in"] ** 2)
        df["true_label"] = np.where((bmi >= 18.5) & (bmi <= 24.9), "Healthy", "Unhealthy")
    return df

def fit_minmax(col: pd.Series):
    lo, hi = float(col.min()), float(col.max())
    return lo, hi

def transform_minmax(col: pd.Series, lo: float, hi: float) -> pd.Series:
    return (col - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=col.index)

def compute_score(dfp: pd.DataFrame, w1, w2, w3, w4) -> pd.Series:
    return (w1*dfp["height_norm"] + w2*dfp["weight_norm"] +
            w3*dfp["waist_norm"]  + w4*dfp["color_norm"])

def predict_label(score: pd.Series, T: float) -> pd.Series:
    return np.where(score < T, "Healthy", "Unhealthy")

def accuracy_from_pred(dfp: pd.DataFrame, pred: pd.Series) -> float:
    return float(np.mean(pred == dfp["true_label"]))

# Encode color to numeric (fixed mapping for reproducibility)
COLOR_MAP = {
    "Blue":0, "Red":1, "Green":2, "Yellow":3,
    "Black":4, "Pink":5, "Purple":6, "Orange":7, "White":8
}
def encode_color(s: pd.Series) -> pd.Series:
    return s.map(COLOR_MAP).fillna(9).astype(int)

# %% -----------------------------------------
# Load & prep
# --------------------------------------------
df = load_dataset()
df.columns = [to_snake(c) for c in df.columns]
print("Columns:", list(df.columns))

required = {"id","height_in","weight_lb","waist_in","favorite_color","true_label"}
missing  = required - set(df.columns)
assert not missing, f"Missing columns: {missing}"

# Train/Test split (focus on train this week)
train = df[df["id"].between(1,10)].copy()
test  = df[df["id"].between(11,12)].copy()
print(f"Train size: {len(train)} | Test size: {len(test)}")

# Encode color
train["colorcode"] = encode_color(train["favorite_color"])
test["colorcode"]  = encode_color(test["favorite_color"])

# Fit normalization on train only
h_lo, h_hi = fit_minmax(train["height_in"])
w_lo, w_hi = fit_minmax(train["weight_lb"])
x_lo, x_hi = fit_minmax(train["waist_in"])
c_lo, c_hi = fit_minmax(train["colorcode"])

# Transform (train)
train["height_norm"] = transform_minmax(train["height_in"], h_lo, h_hi)
train["weight_norm"] = transform_minmax(train["weight_lb"], w_lo, w_hi)
train["waist_norm"]  = transform_minmax(train["waist_in"],  x_lo, x_hi)
train["color_norm"]  = transform_minmax(train["colorcode"],  c_lo, c_hi)

# (Optional) prepare test for next week
test["height_norm"] = transform_minmax(test["height_in"], h_lo, h_hi)
test["weight_norm"] = transform_minmax(test["weight_lb"], w_lo, w_hi)
test["waist_norm"]  = transform_minmax(test["waist_in"],  x_lo, x_hi)
test["color_norm"]  = transform_minmax(test["colorcode"],  c_lo, c_hi)

# %% -----------------------------------------
# Grid Search — maximize accuracy on train (IDs 1–10)
#   - w1..w4 in {0.0, 0.1, ..., 1.0}, sum=1
#   - T in {0.30, 0.35, ..., 0.70}
# --------------------------------------------
grid_w = np.round(np.linspace(0.0, 1.0, 11), 2)
grid_T = np.round(np.linspace(0.30, 0.70, 9), 2)

best = {"acc": -1.0, "w": None, "T": None}

for w1 in grid_w:
    w2_max = 1.0 - w1
    w2_grid = np.round(np.linspace(0.0, w2_max, int(w2_max*10)+1), 2)
    for w2 in w2_grid:
        rem = 1.0 - (w1 + w2)
        if rem < 0:
            continue
        w3_grid = np.round(np.linspace(0.0, rem, int(rem*10)+1), 2)
        for w3 in w3_grid:
            w4 = np.round(1.0 - (w1 + w2 + w3), 2)
            if w4 < -1e-9:
                continue

            score_tr = compute_score(train, w1, w2, w3, w4)
            for T in grid_T:
                pred_tr = predict_label(score_tr, T)
                acc     = accuracy_from_pred(train, pred_tr)
                if acc > best["acc"]:
                    best = {"acc": acc, "w": (w1, w2, w3, w4), "T": T}

print("✅ Best train accuracy:", best["acc"])
print("✅ Best weights (w1,w2,w3,w4):", best["w"])
print("✅ Best threshold T:", best["T"])

# Save summary
with open(os.path.join(OUT_DIR, "week3_best_summary.txt"), "w", encoding="utf-8") as f:
    f.write(f"Best train accuracy: {best['acc']:.3f}\n")
    f.write(f"Best weights (w1,w2,w3,w4): {best['w']}\n")
    f.write(f"Best threshold T: {best['T']:.2f}\n")

# Save train predictions at best params (for grading/inspection)
w1_best, w2_best, w3_best, w4_best = best["w"]
T_best = best["T"]

train["score_best"] = compute_score(train, w1_best, w2_best, w3_best, w4_best)
train["pred_best"]  = predict_label(train["score_best"], T_best)
train["correct"]    = (train["pred_best"] == train["true_label"]).astype(int)

out_train_csv = os.path.join(OUT_DIR, "week3_train_best.csv")
train[["id","height_in","weight_lb","waist_in","favorite_color",
       "score_best","pred_best","true_label","correct"]].to_csv(out_train_csv, index=False)
print("💾 Saved:", out_train_csv)

# %% -----------------------------------------
# 2D Error Surface Heatmaps: vary (w_k, T) with other weights rescaled
#   - For each k ∈ {1,2,3,4}, sweep w_k ∈ [0,1]
#   - Keep the proportions of the other three weights the same as in best['w'],
#     but rescale them to (1 - w_k) so the total still sums to 1.
# --------------------------------------------
T_grid = np.round(np.linspace(0.30, 0.70, 41), 2)
W_grid = np.round(np.linspace(0.00, 1.00, 41), 2)
w_best = np.array(best["w"], dtype=float)

def make_heatmap(vary_index: int, label: str):
    fixed = np.delete(w_best, vary_index)
    fixed_sum = fixed.sum()
    fixed_ratio = fixed / fixed_sum if fixed_sum > 0 else np.ones_like(fixed) / len(fixed)

    err_map = np.zeros((len(T_grid), len(W_grid)), dtype=float)

    for iT, T in enumerate(T_grid):
        for iw, w_var in enumerate(W_grid):
            rem = max(0.0, 1.0 - w_var)
            others = rem * fixed_ratio

            # reconstruct weight vector
            w_try = w_best.copy()
            w_try[vary_index] = w_var
            w_try[np.arange(4) != vary_index] = others

            sc   = compute_score(train, *w_try)
            pred = predict_label(sc, T)
            acc  = accuracy_from_pred(train, pred)
            err_map[iT, iw] = 1.0 - acc

    plt.figure(figsize=(8,5))
    im = plt.imshow(
        err_map,
        origin="lower",
        aspect="auto",
        extent=[W_grid.min(), W_grid.max(), T_grid.min(), T_grid.max()],
    )
    plt.colorbar(im, label="Error (1 - Accuracy)")
    plt.xlabel(f"{label} (varied)")
    plt.ylabel("Threshold T")
    plt.title(f"Week 3 — Error Surface for {label} vs T (others rescaled; best point marked)")

    # mark best point on this slice
    plt.scatter([w_best[vary_index]],[T_best], marker="x", s=120, color="white", linewidths=2)
    plt.text(w_best[vary_index], T_best, "  best", color="white", va="center", ha="left")

    out = os.path.join(OUT_DIR, f"week3_error_heatmap_{label}_T.png")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print("🖼️ Saved:", out)

for idx, label in enumerate(["w1","w2","w3","w4"]):
    make_heatmap(idx, label)

print("✅ All four heatmaps generated in:", OUT_DIR)

# %% -----------------------------------------
# Notes for students (short summary)
#  - We defined an error via Accuracy on the TRAIN set.
#  - We performed grid search over (w1..w4, T) and selected the best combination.
#  - Each heatmap is a 2D slice of the error surface: darker color = lower error.
#  - Next week: instead of checking every grid point, we'll learn Gradient Descent.
# --------------------------------------------
print("✅ Done. See outputs in:", OUT_DIR)
