# -*- coding: utf-8 -*-
"""
Week 5 exploratory health-data plots, completed 2026-10-20.

Accept an explicit CSV produced by week5_data_sets.py. No clinical model or
validation is implied by the descriptive plots. Importing this file is read-only.
"""

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd


def prepare_data(frame):
    """Normalize field names and coerce available measurement columns."""
    frame = frame.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate columns after normalizing field names.")
    columns = [c for c in ["height_cm", "weight_kg", "waist_cm", "bmi", "systolic_bp", "diastolic_bp"] if c in frame]
    if not columns:
        raise ValueError("No recognized measurement columns in the input.")
    result = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if "bmi" not in result and {"height_cm", "weight_kg"} <= set(result):
        valid_height = result["height_cm"].where(result["height_cm"] > 0)
        result["bmi"] = result["weight_kg"] / (valid_height/100)**2
    result = result.dropna(how="all")
    if result.empty:
        raise ValueError("No usable numeric measurements in the input.")
    return result


def make_plots(frame, output_dir):
    import matplotlib.pyplot as plt
    data = prepare_data(frame)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    axes = data.hist(bins=20, figsize=(10, 7))
    fig = np.asarray(axes).flat[0].figure
    fig.suptitle("Observed measurements: descriptive exploration")
    fig.tight_layout()
    paths.append(output_dir/"week5_distributions.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), labels=corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), labels=corr.columns)
    ax.set_title("Pairwise measurement correlations")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    paths.append(output_dir/"week5_correlations.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/week5"))
    args = parser.parse_args()
    for path in make_plots(pd.read_csv(args.input), args.output_dir):
        print(path)


if __name__ == "__main__":
    main()

