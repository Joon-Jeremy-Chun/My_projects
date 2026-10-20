# code_02_bands_and_plots.py
# Load saved gold price data, normalize, compute MA + asymmetric sigma bands,
# and save plots to ./figures/

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS (Edit only this block to change the entire pipeline)
# ============================================================

CSV_PATH = Path("./data/gold_prices.csv")   # CSV file saved by code_01
FIG_DIR  = Path("./figures")                # Directory to save figures

START_DATE = "2018-01-01"   # Analysis start date
END_DATE   = "2025-12-31"   # Analysis end date (inclusive)

NORMALIZE_MODE = "zscore"   # "zscore" | "minmax" | "none"

MA_WINDOW = 20              # Moving average window size
UPPER_K = 2.0               # Upper band sigma multiplier
LOWER_K = 1.2               # Lower band sigma multiplier (asymmetric allowed)

# Zoom plot (optional): specify dates, or set to None to disable
ZOOM_START = "2020-03-01"
ZOOM_END   = "2020-06-01"

# Output figure filenames
FULL_FIG_NAME = "gold_bands_full.png"
ZOOM_FIG_NAME = "gold_bands_zoom.png"


# ============================================================


def ensure_figures_dir(fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"Date", "Price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}. Expected at least: {sorted(required)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Price"]).sort_values("Date").reset_index(drop=True)

    # ✅ Filter by date range (inclusive)
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    df = df[(df["Date"] >= s) & (df["Date"] <= e)].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No data after filtering dates: {start_date} ~ {end_date}")

    return df


def add_features(
    df: pd.DataFrame,
    ma_window: int,
    upper_k: float,
    lower_k: float,
    normalize_mode: str,
) -> pd.DataFrame:
    out = df.copy()

    x = out["Price"].astype(float).values

    if normalize_mode == "zscore":
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=0)
        if sd == 0:
            raise ValueError("Standard deviation is 0; cannot z-score normalize.")
        out["Price_Norm"] = (out["Price"] - mu) / sd

    elif normalize_mode == "minmax":
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        if xmax == xmin:
            raise ValueError("All prices identical; cannot min-max normalize.")
        out["Price_Norm"] = (out["Price"] - xmin) / (xmax - xmin)

    elif normalize_mode == "none":
        out["Price_Norm"] = out["Price"]

    else:
        raise ValueError("normalize_mode must be one of: zscore, minmax, none")

    # Rolling stats
    out["MA"] = out["Price_Norm"].rolling(window=ma_window, min_periods=ma_window).mean()
    out["Sigma"] = out["Price_Norm"].rolling(window=ma_window, min_periods=ma_window).std(ddof=0)

    # Asymmetric bands
    out["Upper"] = out["MA"] + upper_k * out["Sigma"]
    out["Lower"] = out["MA"] - lower_k * out["Sigma"]

    return out


def plot_series(df: pd.DataFrame, fig_path: Path, title: str) -> Path:
    plt.figure()
    plt.plot(df["Date"], df["Price_Norm"], label="Normalized Price")
    plt.plot(df["Date"], df["MA"], label="MA (center)")
    plt.plot(df["Date"], df["Upper"], label="Upper band")
    plt.plot(df["Date"], df["Lower"], label="Lower band")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    #plt.close()
    return fig_path


def main():
    ensure_figures_dir(FIG_DIR)

    try:
        df = load_data(CSV_PATH, START_DATE, END_DATE)
        df_feat = add_features(df, MA_WINDOW, UPPER_K, LOWER_K, NORMALIZE_MODE)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Drop early rows where rolling stats are NaN
    df_plot = df_feat.dropna(subset=["MA", "Sigma", "Upper", "Lower"]).copy()
    if df_plot.empty:
        print("[ERROR] Not enough data after rolling window. Try smaller MA_WINDOW or more data.",
              file=sys.stderr)
        sys.exit(1)

    title_full = (
        f"Gold Bands | {START_DATE}~{END_DATE} | normalize={NORMALIZE_MODE} | "
        f"MA={MA_WINDOW} | upper_k={UPPER_K} | lower_k={LOWER_K}"
    )

    full_path = plot_series(df_plot, FIG_DIR / FULL_FIG_NAME, title_full)
    print(f"[OK] Saved full plot: {full_path.resolve()}")

    # Optional zoom plot
    if ZOOM_START and ZOOM_END:
        z0 = pd.to_datetime(ZOOM_START)
        z1 = pd.to_datetime(ZOOM_END)
        zoom_df = df_plot[(df_plot["Date"] >= z0) & (df_plot["Date"] <= z1)].copy()
        if zoom_df.empty:
            print(f"[WARN] Zoom range empty: {ZOOM_START} ~ {ZOOM_END}", file=sys.stderr)
        else:
            title_zoom = title_full + f" | ZOOM {ZOOM_START}~{ZOOM_END}"
            zoom_path = plot_series(zoom_df, FIG_DIR / ZOOM_FIG_NAME, title_zoom)
            print(f"[OK] Saved zoom plot: {zoom_path.resolve()}")

    # Save feature table
    out_features = CSV_PATH.with_name(CSV_PATH.stem + "_features.csv")
    df_feat.to_csv(out_features, index=False)
    print(f"[OK] Saved features CSV: {out_features.resolve()}")


if __name__ == "__main__":
    main()
