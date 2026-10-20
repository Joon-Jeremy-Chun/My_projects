# %% [1] Setup & Reproducibility
# WHAT: Global config, random seed, paths. Keep experiments tidy and repeatable.
# WHY : So everyone can reproduce the same results with the same inputs.
# HOW : Change SEED or file paths if needed.

import os, sys, math, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

# -------------------- Random Seed --------------------
SEED = 42
rng = np.random.default_rng(SEED)

# -------------------- Paths (EDIT HERE IF NEEDED) --------------------
# Folder that contains your CSV file
BASE_DIR = Path(
    r"C:\Users\joonc\My_github\My_projects\Mini-projects\MachineLearning2025F\DataSets\Health_2020_Eng"
)  # <-- change this folder if your data is elsewhere

# CSV file name (inside BASE_DIR)
DATA_CSV = BASE_DIR / "National Health Insurance Service_Health Checkup Data_20201231_1_EN.csv"

# Where to save outputs (JSON, plots, etc.)
OUTPUT_DIR = BASE_DIR / "week6_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Numpy / Pandas Display Options --------------------
np.set_printoptions(suppress=True, linewidth=120)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 200)

# -------------------- Quick Sanity Print --------------------
print("[INFO] BASE_DIR :", BASE_DIR)
print("[INFO] DATA_CSV:", DATA_CSV)
print("[INFO] Exists?  :", DATA_CSV.exists())
print("[INFO] OUTPUT_DIR:", OUTPUT_DIR)

# %% [2] Load Data & Schema Map
# WHAT: Read CSV, select features and targets (SBP, DBP).
# WHY : Clean schema prevents downstream bugs.
# HOW : Update FEATURE_COLS / TARGET_COLS to match your dataset.

# [STUDENT ACTION] Change these to match your CSV column names.
# Minimal expected numeric columns: (examples) height_cm, weight_kg, waist_cm, glucose, chol_total
# Targets must be exactly 'sbp' and 'dbp' (lowercase) for this script unless you edit below.
FEATURE_COLS = [
    "height_cm", "weight_kg", "waist_cm",        # anthropometrics
    "glucose_fasting", "chol_total",             # labs (optional)
    # "age_years", "exercise_freq", "smoking_packyears",  # add if present
]
TARGET_COLS = ["sbp", "dbp"]  # systolic and diastolic blood pressure

# Load
if not Path(DATA_CSV).exists():
    raise FileNotFoundError(f"Could not find CSV at: {DATA_CSV} (edit DATA_CSV at top)")
df_raw = pd.read_csv(DATA_CSV)

# Sanity check for required columns
missing_feats = [c for c in FEATURE_COLS if c not in df_raw.columns]
missing_tgts  = [c for c in TARGET_COLS  if c not in df_raw.columns]
if missing_feats or missing_tgts:
    raise ValueError(
        f"Missing columns.\n"
        f"  Missing features: {missing_feats}\n"
        f"  Missing targets : {missing_tgts}\n"
        f"  Available      : {list(df_raw.columns)}"
    )

df = df_raw[FEATURE_COLS + TARGET_COLS].copy()


# %% [3] Missingness & Basic Stats Report (before any cleaning)
# WHAT: Print total rows/cols, NA counts, and min/max per numeric column.
# WHY : Show “nonsense” values to motivate cleaning.
# HOW : Run as-is; discuss min/max outliers in class.

def report_missing_minmax(frame: pd.DataFrame, title: str):
    print("\n" + "="*80)
    print(f"[REPORT] {title}")
    print("="*80)
    n_rows, n_cols = frame.shape
    print(f"Total rows: {n_rows:,} | Total cols: {n_cols}")
    na_counts = frame.isna().sum().sort_values(ascending=False)
    if na_counts.sum() > 0:
        print("\nPer-column NA counts (and %):")
        for c in na_counts.index:
            if na_counts[c] > 0:
                pct = 100.0 * na_counts[c] / n_rows
                print(f"  - {c:>20s}: {na_counts[c]:>6d} ({pct:5.2f}%)")
    else:
        print("\nNo missing values detected.")

    # Numeric columns only for min/max
    num_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        print("\nPer-numeric-column min/max (use to spot 'impossible' values):")
        for c in num_cols:
            col = frame[c].astype(float)
            print(f"  - {c:>20s}: min={col.min():>10.4f} | max={col.max():>10.4f}")
    print("="*80 + "\n")

report_missing_minmax(df, "Initial Dataset")


# %% [4] Drop Rows with Any Missing Values (Listwise Deletion)
# WHAT: Remove rows that contain NA in features or targets (simple educational choice).
# WHY : Keep the lesson focused on outlier handling; imputation is for later weeks.
# HOW : Change NA_STRATEGY below if you want to compare with imputation.

NA_STRATEGY = "drop_rows"  # options: "drop_rows" (this week focus)

if NA_STRATEGY == "drop_rows":
    n0 = len(df)
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS).reset_index(drop=True)
    print(f"[CLEAN] Dropped rows with any NA: {n0 - len(df):,} | Remaining: {len(df):,}")
else:
    raise NotImplementedError("Only 'drop_rows' is supported this week.")

report_missing_minmax(df, "After NA-Row Removal")


# %% [5] 3σ Outlier Cutoff — Count, Print, and Filter
# WHAT: Compute mean and std per numeric column; mark rows outside mean±3*std as outliers (per column).
# WHY : Educational rule to remove obvious typos/outliers.
# HOW TO CHANGE:
#   - SIGMA_K: tighten (2.5) or loosen (3.5) the cutoff
#   - APPLY_TO_TARGETS: include/exclude SBP/DBP from outlier filtering
# EXAMPLE:
#   SIGMA_K = 3.0
#   APPLY_TO_TARGETS = True

SIGMA_K = 3.0
APPLY_TO_TARGETS = True  # include SBP/DBP in the cutoff

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
consider_cols = [c for c in num_cols if (APPLY_TO_TARGETS or c not in TARGET_COLS)]

# 평균, 표준편차
mu_full = df[consider_cols].mean(axis=0)
sd_full = df[consider_cols].std(axis=0)

# std==0 인 컬럼은 outlier 필터에서 제외 (상수 컬럼)
constant_cols = sd_full[sd_full == 0].index.tolist()
if constant_cols:
    print("\n[3σ NOTE] Constant columns (std=0) will be skipped in outlier filtering:")
    for c in constant_cols:
        print(f"  - {c}")

varying_cols = [c for c in consider_cols if c not in constant_cols]
mu = mu_full[varying_cols]
sd = sd_full[varying_cols]

low = mu - SIGMA_K * sd
high = mu + SIGMA_K * sd

print("\n[3σ BOUNDS] (per column)")
for c in varying_cols:
    lo = low[c]; hi = high[c]
    print(f"  - {c:>20s}: μ={mu[c]:>9.4f} σ={sd[c]:>9.4f} | bounds [{lo:>9.4f}, {hi:>9.4f}]")

# Count rows outside bounds by column (for reporting only)
outside_by_col = {}
for c in varying_cols:
    col = df[c].astype(float)
    mask_out = (col < low[c]) | (col > high[c])
    outside_by_col[c] = int(mask_out.sum())

print("\n[3σ OUTLIERS] rows outside bounds by column:")
for c in varying_cols:
    print(f"  - {c:>20s}: {outside_by_col[c]:>6d}")

# Build a combined mask: keep rows inside all selected columns' bounds
keep_mask = np.ones(len(df), dtype=bool)
for c in varying_cols:
    col = df[c].astype(float)
    keep_mask &= (col >= low[c]) & (col <= high[c])

inside_count = int(keep_mask.sum())
removed_by_sigma = len(df) - inside_count
print(f"\n[3σ SUMMARY] kept: {inside_count:,} | removed: {removed_by_sigma:,} "
      f"({removed_by_sigma * 100.0 / len(df):.2f}% of current data)")

# Print min/max BEFORE filtering (last preview)
print("\n[MIN/MAX BEFORE 3σ FILTER] (last preview before removal)")
for c in num_cols:
    col = df[c].astype(float)
    print(f"  - {c:>20s}: min={col.min():>10.4f} | max={col.max():>10.4f}")

df = df.loc[keep_mask].reset_index(drop=True)
report_missing_minmax(df, "After 3σ Filtering")



# %% [6] Feature Engineering (Optional) & Numericization
# WHAT: Add derived features and ensure all model inputs are numeric.
# WHY : Models need numeric inputs; derived features may help.
# HOW : Toggle features here. If your dataset lacks a column, remove it from FEATURE_COLS instead.

def add_optional_features(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    # Example: BMI = weight_kg / (height_m^2), if both exist
    if {"height_cm","weight_kg"}.issubset(f.columns):
        h_m = f["height_cm"].astype(float) / 100.0
        f["BMI"] = f["weight_kg"].astype(float) / (h_m ** 2)
    # Example: WHtR = waist_cm / height_cm
    if {"waist_cm","height_cm"}.issubset(f.columns):
        f["WHtR"] = f["waist_cm"].astype(float) / f["height_cm"].astype(float)
    return f

df = add_optional_features(df)

# Refresh feature list to include any new columns (keep targets separate)
ALL_FEATURES = [c for c in df.columns if c not in TARGET_COLS]
X_df = df[ALL_FEATURES].select_dtypes(include=[np.number]).copy()
Y_df = df[TARGET_COLS].copy()

print(f"[FEATS] Using {X_df.shape[1]} numeric features: {list(X_df.columns)}")


# %% [7] Train/Validation Split (Hold-out)
# WHAT: Keep a clean validation set for model selection.
# WHY : Prevents overfitting to the same data we train on.
# HOW TO CHANGE:
#   - VAL_RATIO: 0.2 ~ 0.3 typical
#   - RANDOM_SEED: change to see stability across splits
# EXAMPLE:
#   VAL_RATIO = 0.25

VAL_RATIO = 0.25
RANDOM_SEED = 0

n = len(X_df)
perm = np.random.default_rng(RANDOM_SEED).permutation(n)
n_val = int(n * VAL_RATIO)
val_idx = perm[:n_val]
trn_idx = perm[n_val:]

X_tr_raw = X_df.iloc[trn_idx].to_numpy().astype(float)
Y_tr = Y_df.iloc[trn_idx].to_numpy().astype(float)
X_val_raw = X_df.iloc[val_idx].to_numpy().astype(float)
Y_val = Y_df.iloc[val_idx].to_numpy().astype(float)

print(f"[SPLIT] Train: {len(X_tr_raw):,} | Val: {len(X_val_raw):,}")


# %% [8] Scaling (Standardize with Train μ, σ only)
# WHAT: Standardize features for fair comparison; avoid leakage.
# WHY : Many optimizers and models behave better when inputs are standardized.
# HOW TO CHANGE:
#   - SCALER: 'standard' only provided this week; add 'minmax' if you want
#   - SCALE_TARGETS: leave False (report predictions in original units)

SCALER = "standard"
SCALE_TARGETS = False

def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd

def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd

X_tr, mu_X, sd_X = standardize_fit(X_tr_raw)
X_val = standardize_apply(X_val_raw, mu_X, sd_X)

if SCALE_TARGETS:
    Y_tr, mu_Y, sd_Y = standardize_fit(Y_tr)
    Y_val = standardize_apply(Y_val, mu_Y, sd_Y)
else:
    mu_Y, sd_Y = None, None

print("[SCALE] Standardization complete (features). Targets scaled? ", SCALE_TARGETS)


# %% [9] Models — Linear (closed form or GD) and MLP (1 hidden)
# WHAT: Compare pure linear combination vs. hidden nonlinearity.
# WHY : Shows how hidden nodes + activation increase model capacity.
# HOW TO CHANGE:
#   - MODE: 'linear' or 'mlp'
#   - For LINEAR (closed-form ridge): set REG='none'|'ridge' and ALPHA
#   - For MLP: set HIDDEN, ACT, INIT, WEIGHT_DECAY, etc.

def linear_closed_form_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 0.0) -> Dict[str, np.ndarray]:
    # Solve (X'X + alpha I)W = X'Y; returns W and b via augmented trick
    n, d = X.shape
    # add bias column
    Xa = np.concatenate([X, np.ones((n,1))], axis=1)
    I = np.eye(d+1); I[-1,-1] = 0.0  # don't regularize bias
    Wb = np.linalg.pinv(Xa.T @ Xa + alpha * I) @ (Xa.T @ Y)
    W = Wb[:-1, :]
    b = Wb[-1:, :]
    return {"W": W, "b": b}

def linear_predict(X: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    return X @ params["W"] + params["b"]

def he_init(n_in, n_out, rng):
    std = np.sqrt(2.0 / n_in)
    return rng.normal(0, std, size=(n_in, n_out))

def xavier_init(n_in, n_out, rng):
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=(n_in, n_out))

def act_forward(Z, kind: str):
    if kind == "relu": return np.maximum(0, Z)
    if kind == "tanh": return np.tanh(Z)
    if kind == "leaky_relu": return np.where(Z>0, Z, 0.01*Z)
    raise ValueError("Unknown activation")

def act_backward(Z, kind: str):
    if kind == "relu": return (Z > 0).astype(Z.dtype)
    if kind == "tanh": return 1 - np.tanh(Z)**2
    if kind == "leaky_relu":
        G = np.ones_like(Z); G[Z<0] = 0.01; return G
    raise ValueError("Unknown activation")

def mlp_init(d_in, h, d_out, act="relu", seed=SEED):
    rr = np.random.default_rng(seed)
    W1 = he_init(d_in, h, rr) if act in ("relu","leaky_relu") else xavier_init(d_in, h, rr)
    b1 = np.zeros((1, h))
    W2 = xavier_init(h, d_out, rr)
    b2 = np.zeros((1, d_out))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def mlp_forward(X, P, act):
    Z1 = X @ P["W1"] + P["b1"]
    A1 = act_forward(Z1, act)
    Z2 = A1 @ P["W2"] + P["b2"]  # regression output
    cache = {"X": X, "Z1": Z1, "A1": A1}
    return Z2, cache

def mlp_backward(Y, Pred, P, cache, act, weight_decay=0.0):
    n = Y.shape[0]
    X, Z1, A1 = cache["X"], cache["Z1"], cache["A1"]
    dZ2 = (2.0 / n) * (Pred - Y)                # dL/dZ2 for MSE
    dW2 = A1.T @ dZ2 + weight_decay * P["W2"]
    db2 = dZ2.sum(axis=0, keepdims=True)

    dA1 = dZ2 @ P["W2"].T
    dZ1 = dA1 * act_backward(Z1, act)
    dW1 = X.T @ dZ1 + weight_decay * P["W1"]
    db1 = dZ1.sum(axis=0, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def mse(Y,P): return float(np.mean((Y-P)**2))
def rmse(Y,P): return float(np.sqrt(mse(Y,P)))
def mae(Y,P): return float(np.mean(np.abs(Y-P)))
def r2(Y,P):
    ss_res = np.sum((Y-P)**2)
    ss_tot = np.sum((Y - Y.mean(axis=0, keepdims=True))**2)
    return float(1.0 - ss_res/(ss_tot + 1e-8))


# %% [10] Training Loop (Mini-batch GD) + Step LR
# WHAT: Train either linear-by-GD (optional) or MLP with GD/step LR.
# WHY : Show gradient descent dynamics and schedules.
# HOW TO CHANGE:
#   - OPTIMIZER: we keep simple GD this week
#   - LR, EPOCHS, BATCH, STEP_SCHEDULE: students should tweak these
# EXAMPLE:
#   LR = 1e-3, EPOCHS = 250, BATCH = 64

def train_mlp_gd(
    X_tr, Y_tr, X_val, Y_val,
    hidden=32, act="relu",
    lr=1e-3, epochs=250, batch=64,
    weight_decay=0.0,
    seed=SEED,
    verbose=True
):
    n, d = X_tr.shape
    d_out = Y_tr.shape[1]
    rr = np.random.default_rng(seed)
    P = mlp_init(d, hidden, d_out, act=act, seed=seed)

    def step_lr(e):
        # Step schedule: (0–50%) lr, (50–80%) 0.3*lr, (80–100%) 0.1*lr
        if e < epochs*0.5: return lr
        if e < epochs*0.8: return lr*0.3
        return lr*0.1

    idx = np.arange(n)
    for e in range(epochs):
        rr.shuffle(idx)
        curr_lr = step_lr(e)
        for s in range(0, n, batch):
            batch_idx = idx[s:s+batch]
            xb, yb = X_tr[batch_idx], Y_tr[batch_idx]
            pred, cache = mlp_forward(xb, P, act)
            grads = mlp_backward(yb, pred, P, cache, act, weight_decay=weight_decay)
            P["W1"] -= curr_lr * grads["dW1"]; P["b1"] -= curr_lr * grads["db1"]
            P["W2"] -= curr_lr * grads["dW2"]; P["b2"] -= curr_lr * grads["db2"]
        # Progress log
        if verbose and (e % max(1, epochs//5) == 0 or e == epochs-1):
            tr_pred, _ = mlp_forward(X_tr, P, act)
            va_pred, _ = mlp_forward(X_val, P, act)
            print(f"Epoch {e:3d} | TR RMSE:{rmse(Y_tr,tr_pred):.3f} R2:{r2(Y_tr,tr_pred):.3f} "
                  f"| VAL RMSE:{rmse(Y_val,va_pred):.3f} R2:{r2(Y_val,va_pred):.3f}")
    return P

def predict_mlp(X, P, act="relu"):
    out, _ = mlp_forward(X, P, act)
    return out


# %% [11] Staged Search (Coarse -> Refine)
# WHAT: Don’t brute-force the whole space. Scan coarsely, then refine near the best.
# WHY : Educational + practical. Also mention non-convex local minima risk.
# HOW TO CHANGE:
#   - MODES: ('linear','mlp') if you want to compare; here we show both.
#   - COARSE grids: LR_GRID, H_GRID, ACTS, SEEDS
#   - REFINE around the best from stage 1

# STUDENT HYPER-PARAMETERS (Edit here)
# WHAT : Main knobs for the experiment. Change only these at first!
# WHY  : To observe how each affects RMSE/R2 and convergence.
# HOW  : Pick one, modify, record effect in worksheet.
# EXAMPLE:
#   LR_GRID = [1e-2, 3e-3, 1e-3]   # try faster or slower learning
#   H_GRID  = [16, 32, 64]         # more hidden units -> more capacity (and overfitting risk)
#   ACTS    = ["relu", "tanh"]     # nonlinearity choice affects loss landscape

MODES    = ("linear", "mlp")  # include both for class comparison
LR_GRID  = [1e-2, 3e-3, 1e-3]
H_GRID   = [16, 32, 64]
ACTS     = ["relu", "tanh"]
SEEDS    = [0, 1, 2]
EPOCHS   = 250
BATCH    = 64
WEIGHT_DECAY = 0.0

# Linear model options (closed-form ridge)
LINEAR_REG   = "ridge"  # 'none' or 'ridge'
LINEAR_ALPHA = 0.1

def eval_metrics(Y_true, Y_pred, name="VAL"):
    return {
        f"{name}_RMSE": rmse(Y_true, Y_pred),
        f"{name}_MAE" : mae(Y_true, Y_pred),
        f"{name}_R2"  : r2(Y_true,  Y_pred),
    }

best = {"rmse": np.inf, "cfg": None, "params": None, "mode": None, "act": None}

# ----- Stage 1: Coarse scan -----
print("\n[STAGE 1] Coarse scan")
for mode in MODES:
    if mode == "linear":
        # closed-form ridge (no epochs)
        params = linear_closed_form_ridge(X_tr, Y_tr, alpha=(LINEAR_ALPHA if LINEAR_REG=="ridge" else 0.0))
        pred_val = linear_predict(X_val, params)
        metrics = eval_metrics(Y_val, pred_val, name="VAL")
        print(f"(linear, reg={LINEAR_REG}, alpha={LINEAR_ALPHA}) -> VAL_RMSE={metrics['VAL_RMSE']:.3f}, VAL_R2={metrics['VAL_R2']:.3f}")
        if metrics["VAL_RMSE"] < best["rmse"]:
            best.update({"rmse": metrics["VAL_RMSE"], "cfg": ("linear", None, None, None), "params": params, "mode": "linear", "act": None})
    else:
        for act in ACTS:
            for h in H_GRID:
                for lr in LR_GRID:
                    for sd in SEEDS:
                        P = train_mlp_gd(
                            X_tr, Y_tr, X_val, Y_val,
                            hidden=h, act=act, lr=lr, epochs=EPOCHS, batch=BATCH,
                            weight_decay=WEIGHT_DECAY, seed=sd, verbose=False
                        )
                        pred_val = predict_mlp(X_val, P, act=act)
                        metrics = eval_metrics(Y_val, pred_val, name="VAL")
                        print(f"(mlp, act={act}, h={h}, lr={lr:g}, seed={sd}) -> VAL_RMSE={metrics['VAL_RMSE']:.3f}, VAL_R2={metrics['VAL_R2']:.3f}")
                        if metrics["VAL_RMSE"] < best["rmse"]:
                            best.update({"rmse": metrics["VAL_RMSE"], "cfg": ("mlp", h, act, lr), "params": P, "mode": "mlp", "act": act})

print("\n[STAGE 1] Best so far:", best)

# ----- Stage 2: Refine near the best (only for MLP best) -----
if best["mode"] == "mlp":
    mode_b, h_b, act_b, lr_b = best["cfg"]
    print("\n[STAGE 2] Refine near best (MLP)")
    lr_refine = [lr_b*1.5, lr_b, lr_b*0.5]
    h_refine  = list({max(1, h_b//2), h_b, h_b*2})
    for h in h_refine:
        for lr in lr_refine:
            for sd in SEEDS:
                P = train_mlp_gd(
                    X_tr, Y_tr, X_val, Y_val,
                    hidden=h, act=act_b, lr=lr, epochs=EPOCHS, batch=BATCH,
                    weight_decay=WEIGHT_DECAY, seed=sd, verbose=False
                )
                pred_val = predict_mlp(X_val, P, act=act_b)
                metrics = eval_metrics(Y_val, pred_val, name="VAL")
                print(f"(refine, act={act_b}, h={h}, lr={lr:g}, seed={sd}) -> VAL_RMSE={metrics['VAL_RMSE']:.3f}, VAL_R2={metrics['VAL_R2']:.3f}")
                if metrics["VAL_RMSE"] < best["rmse"]:
                    best.update({"rmse": metrics["VAL_RMSE"], "cfg": ("mlp", h, act_b, lr), "params": P, "mode": "mlp", "act": act_b})

print("\n[FINAL BEST] ", best)


# %% [12] Final Evaluation & Reporting
# WHAT: Print metrics and short takeaways. If targets were scaled, inverse-transform.
# WHY : Communicate results clearly; link back to data cleaning decisions.
# HOW : Extend with residual plots or per-target metrics if desired.

def inverse_targets(Y_hat):
    if not SCALE_TARGETS:
        return Y_hat
    # inverse standardization
    return Y_hat * sd_Y + mu_Y

if best["mode"] == "linear":
    val_pred = linear_predict(X_val, best["params"])
else:
    val_pred = predict_mlp(X_val, best["params"], act=best["act"])

val_pred = inverse_targets(val_pred)
Y_val_orig = Y_val if not SCALE_TARGETS else (Y_val * sd_Y + mu_Y)

metrics = {
    "VAL_RMSE": rmse(Y_val_orig, val_pred),
    "VAL_MAE" : mae(Y_val_orig,  val_pred),
    "VAL_R2"  : r2(Y_val_orig,   val_pred),
}
print("\n[RESULTS] Validation metrics (SBP & DBP jointly):")
print(json.dumps(metrics, indent=2))

# Per-target metrics (SBP = col 0, DBP = col 1)
print("\n[RESULTS] Per-target metrics:")
for j, name in enumerate(TARGET_COLS):
    m = {
        f"{name}_RMSE": rmse(Y_val_orig[:, [j]], val_pred[:, [j]]),
        f"{name}_MAE" : mae(Y_val_orig[:, [j]],  val_pred[:, [j]]),
        f"{name}_R2"  : r2(Y_val_orig[:,  [j]],  val_pred[:,  [j]]),
    }
    print(f"  - {name}: {m}")

# Save a concise preprocessing summary for your worksheet
summary = {
    "initial_rows"        : int(len(df_raw)),
    "initial_cols"        : int(df_raw.shape[1]),
    "after_na_rows"       : int(len(df_raw.dropna(subset=FEATURE_COLS+TARGET_COLS))),
    "after_sigma_rows"    : int(len(df)),
    "sigma_k"             : float(SIGMA_K),
    "apply_to_targets"    : bool(APPLY_TO_TARGETS),
    "kept_ratio_3sigma(%)": round(100.0 * len(df) / len(df_raw.dropna(subset=FEATURE_COLS+TARGET_COLS)), 2),
    "features_used"       : list(X_df.columns),
    "best_mode"           : best["mode"],
    "best_cfg"            : best["cfg"]
}
with open(OUTPUT_DIR / "preprocess_and_model_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n[WRITE] Saved summary ->", (OUTPUT_DIR / "preprocess_and_model_summary.json").resolve())
