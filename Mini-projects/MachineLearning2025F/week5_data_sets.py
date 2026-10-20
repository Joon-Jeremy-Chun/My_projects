# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:14:49 2025

@author: joonc
"""

# %% ------------------ CONFIG ------------------
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import glob

# ▶▶ 여기를 본인 경로로 바꾸세요 (폴더 구조는 스샷 기준)
BASE_DIR = Path(__file__).resolve().parents[1] / "DataSets"
KOR_DIR  = BASE_DIR / "Health_2020_Kor"
OUT_DIR  = BASE_DIR / "Health_2020_Eng"

FILE_PATTERN = "National Health Insurance Service_Health Checkup Data_20201231_*.csv"

# %% ------------------ COLUMN MAP ------------------
COL_MAP = {
    "기준년도": "Year",
    "시도코드": "Region_Code",
    "성별코드": "Gender_Code",
    "연령대 코드(5세단위)": "Age_Group_5yr",
    "신장(5Cm단위)": "Height_cm",
    "체중(5Kg 단위)": "Weight_kg",
    "허리둘레": "Waist_cm",
    "시력(좌)": "Vision_Left",
    "시력(우)": "Vision_Right",
    "청력(좌)": "Hearing_Left",
    "청력(우)": "Hearing_Right",
    "수축기 혈압": "Systolic_BP",
    "이완기 혈압": "Diastolic_BP",
    "식전혈당(공복혈당)": "Fasting_Blood_Sugar",
    "총 콜레스테롤": "Total_Cholesterol",
    "트리글리세라이드": "Triglyceride",
    "HDL 콜레스테롤": "HDL_Cholesterol",
    "LDL 콜레스테롤": "LDL_Cholesterol",
    "혈색소": "Hemoglobin",
    "요단백": "Urine_Protein",
    "혈청크레아티닌": "Serum_Creatinine",
    "(혈청지오티)AST": "AST",
    "(혈청지오티)ALT": "ALT",
    "감마 지티피": "Gamma_GTP",
    "흡연상태": "Smoking_Status",
    "음주여부": "Drinking_Status",
    "구강검진 수검여부": "Dental_Checkup",
    "치아우식증유무": "Tooth_Decay",
    "치석": "Tartar",
    "데이터 공개일자": "Data_Release_Date",
}

# %% ------------------ HELPERS ------------------
def read_korean_csv(path: Path) -> pd.DataFrame:
    """EUCKR/CP949 순서로 시도해서 읽기."""
    try:
        return pd.read_csv(path, encoding="euc-kr")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")

def convert_and_save_one(src_path: Path, out_dir: Path) -> Path:
    df = read_korean_csv(src_path)
    # 컬럼명 변환
    df = df.rename(columns=COL_MAP)
    # 출력 파일명: ..._EN.csv
    out_name = src_path.stem + "_EN.csv"
    out_path = out_dir / out_name
    # UTF-8 with BOM (엑셀 호환)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Converted → {out_path.name}  ({len(df):,} rows)")
    return out_path

def make_week5_subset(df_all: pd.DataFrame, out_dir: Path) -> Path:
    """Week5 미니프로젝트용 서브셋 생성 (BMI, 고혈압 라벨 포함)."""
    df = df_all.copy()

    # BMI 생성 (신장 cm → m)
    df["Height_m"] = df["Height_cm"] / 100.0
    df["BMI"] = (df["Weight_kg"] / (df["Height_m"] ** 2)).round(2)

    # Hypertension 라벨(예시): SBP>=140 또는 DBP>=90
    df["Hypertension"] = ((df["Systolic_BP"] >= 140) | (df["Diastolic_BP"] >= 90)).astype(int)

    keep_cols = [
        "Year","Region_Code","Gender_Code","Age_Group_5yr",
        "Height_cm","Weight_kg","Waist_cm","BMI",
        "Systolic_BP","Diastolic_BP","Fasting_Blood_Sugar",
        "Smoking_Status","Drinking_Status",
        "Total_Cholesterol","Triglyceride","HDL_Cholesterol","LDL_Cholesterol",
        "Hypertension"
    ]
    # 존재하는 컬럼만 사용
    keep_cols = [c for c in keep_cols if c in df.columns]
    subset = df[keep_cols].dropna(how="any")  # 간단히 결측 제거

    out_path = out_dir / "Week5_health_subset.csv"
    subset.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Week5 subset saved → {out_path.name}  ({len(subset):,} rows)")
    return out_path

# %% ------------------ MAIN ------------------
def main(argv=None):
    parser = ArgumentParser(description="Convert explicitly selected NHIS CSVs and build the Week 5 subset.")
    parser.add_argument("--input-dir", type=Path, default=KOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    src_files = sorted(glob.glob(str(args.input_dir / FILE_PATTERN)))
    if not src_files:
        raise FileNotFoundError(f"No files matched: {args.input_dir / FILE_PATTERN}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(src_files)} file(s). Converting...")
    out_paths = [convert_and_save_one(Path(p), args.output_dir) for p in src_files]

    # 합본 생성
    print("Concatenating all converted files...")
    dfs = [pd.read_csv(p, encoding="utf-8-sig") for p in out_paths]
    all_df = pd.concat(dfs, ignore_index=True)
    merged_path = args.output_dir / "NHIS_Health_Checkup_2020_All_English.csv"
    all_df.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Merged → {merged_path.name}  ({len(all_df):,} rows)")

    # Week5 서브셋
    make_week5_subset(all_df, args.output_dir)

if __name__ == "__main__":
    main()
