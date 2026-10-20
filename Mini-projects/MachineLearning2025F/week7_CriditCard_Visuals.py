#%% [1] Load Dataset (Excel → DataFrame)
import pandas as pd
import numpy as np

# Path to your dataset (update if needed)
FILE_PATH = r"C:\Users\joonc\My_github\My_projects\Mini-projects\MachineLearning2025F\DataSets\Credit_Card\default of credit card clients.xls"

# Read Excel (first sheet)
df = pd.read_excel(FILE_PATH, header=1)  # header=1 skips the first row title

print("Loaded shape:", df.shape)
print(df.head())
print("\nColumns:\n", df.columns.tolist())

#%% [0.5] Create folder for saving figures (Credit Card visuals)
import os

FIG_DIR = r"C:\Users\joonc\My_github\My_projects\Mini-projects\MachineLearning2025F\CreditCard_Visuals"

if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)
    print("Created folder:", FIG_DIR)
else:
    print("Folder already exists:", FIG_DIR)


#%% [2] Basic Visualizations (Credit Card Visuals)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Selected columns
cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "PAY_0"]

# Split datasets by gender
df_male = df[df["SEX"] == 1]
df_female = df[df["SEX"] == 2]

# 1. Histogram (overall)
df[cols].hist(figsize=(12, 8), bins=30)
plt.suptitle("Credit Card Dataset - Histograms", fontsize=16)
plt.savefig(os.path.join(FIG_DIR, "creditcard_histograms.png"))
plt.show()

# 2. Boxplots (overall)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[cols])
plt.title("Credit Card Dataset - Boxplots (Outlier Detection)")
plt.xticks(rotation=45)
plt.savefig(os.path.join(FIG_DIR, "creditcard_boxplots.png"))
plt.show()

# 3A. Correlation Heatmap — Overall
plt.figure(figsize=(10, 8))
corr_all = df[cols].corr()
sns.heatmap(corr_all, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (All Data)")
plt.savefig(os.path.join(FIG_DIR, "creditcard_corr_all.png"))
plt.show()

# 3B. Correlation Heatmap — Male only
plt.figure(figsize=(10, 8))
corr_male = df_male[cols].corr()
sns.heatmap(corr_male, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap — Male (SEX=1)")
plt.savefig(os.path.join(FIG_DIR, "creditcard_corr_male.png"))
plt.show()

# 3C. Correlation Heatmap — Female only
plt.figure(figsize=(10, 8))
corr_female = df_female[cols].corr()
sns.heatmap(corr_female, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap — Female (SEX=2)")
plt.savefig(os.path.join(FIG_DIR, "creditcard_corr_female.png"))
plt.show()
#%%
# 4.1 Scatter Plot (overall) — LIMIT_BAL vs BILL_AMT1
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="LIMIT_BAL", y="BILL_AMT1", alpha=0.3)
plt.title("LIMIT_BAL vs BILL_AMT1 (Overall)")
plt.savefig(os.path.join(FIG_DIR, "creditcard_scatter_LIMIT_BAL_vs_BILL_AMT1.png"))
plt.show()

# 4.2 Scatter Plot (overall) — LIMIT_BAL vs BILL_AMT2
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="LIMIT_BAL", y="BILL_AMT2", alpha=0.3)
plt.title("LIMIT_BAL vs BILL_AMT2 (Overall)")
plt.savefig(os.path.join(FIG_DIR, "creditcard_scatter_LIMIT_BAL_vs_BILL_AMT2.png"))
plt.show()

# 4.3 Scatter Plot (overall) — BILL_AMT1 vs BILL_AMT2
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="BILL_AMT1", y="BILL_AMT2", alpha=0.3)
plt.title("BILL_AMT1 vs BILL_AMT2 (Overall)")
plt.savefig(os.path.join(FIG_DIR, "creditcard_scatter_BILL_AMT1_vs_BILL_AMT2.png"))
plt.show()

#%% [2.5] Correlation by Education Level (Credit Card Visuals)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Same columns as before
cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "PAY_0"]

# Map EDUCATION code to label
edu_labels = {
    1: "Graduate School",
    2: "University",
    3: "High School",
    4: "Others"
}

for edu_code, edu_name in edu_labels.items():
    df_sub = df[df["EDUCATION"] == edu_code]

    # Just in case there are too few rows
    if df_sub.shape[0] < 10:
        print(f"Skipping EDUCATION={edu_code} ({edu_name}) — too few rows:", df_sub.shape[0])
        continue

    corr_edu = df_sub[cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_edu, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title(f"Correlation Heatmap — EDUCATION={edu_code} ({edu_name})\n(n={df_sub.shape[0]})")

    filename = f"creditcard_corr_education_{edu_code}.png"
    plt.savefig(os.path.join(FIG_DIR, filename))
    plt.show()

    print(f"Saved correlation heatmap for EDUCATION={edu_code} ({edu_name}) to {filename}")


#%% [3] Outlier Detection using μ ± kσ rule
import numpy as np
import pandas as pd

# Choose columns for outlier detection
outlier_cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "PAY_0"]

# Choose k (standard deviations)
k = 3  # You can change this!

results = []

for col in outlier_cols:
    mu = df[col].mean()
    sigma = df[col].std()
    lower = mu - k * sigma
    upper = mu + k * sigma
    
    # count rows outside bounds
    outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    ratio = outliers / len(df) * 100
    
    results.append([col, round(mu, 2), round(sigma, 2), round(lower, 2), round(upper, 2), outliers, round(ratio, 3)])

outlier_df = pd.DataFrame(results, columns=[
    "Column", "Mean", "Std", "Lower (μ-kσ)", "Upper (μ+kσ)", "# Outliers", "Outlier %"
])

print(outlier_df)

#%% [4] Correlation Table (sorted)
import pandas as pd
import numpy as np

corr_matrix = df.corr()

# Flatten into table format
pairs = []
for col1 in corr_matrix.columns:
    for col2 in corr_matrix.columns:
        if col1 < col2:  # avoid duplicates
            pairs.append([col1, col2, corr_matrix.loc[col1, col2]])

corr_df = pd.DataFrame(pairs, columns=["Var1", "Var2", "Correlation"])

# Sort by absolute correlation
corr_sorted = corr_df.reindex(corr_df.Correlation.abs().sort_values(ascending=False).index)

print("Top 10 strongest correlations:")
print(corr_sorted.head(10))

#%% [5] Save Additional Visualizations (Examples)

# Example: Histogram of BILL_AMT1
plt.figure(figsize=(8, 6))
sns.histplot(df["BILL_AMT1"], bins=40, kde=True)
plt.title("Distribution of BILL_AMT1")
plt.savefig(os.path.join(FIG_DIR, "creditcard_dist_BILL_AMT1.png"))
plt.show()

# Example: Scatter - AGE vs PAY_AMT1
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="AGE", y="PAY_AMT1", alpha=0.3)
plt.title("AGE vs PAY_AMT1")
plt.savefig(os.path.join(FIG_DIR, "creditcard_scatter_AGE_vs_PAY_AMT1.png"))
plt.show()


