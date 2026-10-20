# Auto-generated from: Data_overview_천준.ipynb
# Date: 2025-11-12T23:02:58
# Purpose: Translated (heuristic) English comments + #%% segmentation for teaching.
# Note: Original Korean comments are preserved as [KO], and English hints as [EN].
#       This is a best-effort offline translation for classroom use.

#%% -------- Markdown: Markdown cell 1
# ## 여러 정리되지 않은 생각들
# - 변수들의 n차식 그리고 서로 곱(나누는)는 비선형으로 한번해보기
# - 혈압 처방을 받았다면 혈압을 통제하고 있을건데 그러면 몸무게나 다른 팩트들도 정상화 되나?
# - 그냥 혈압만 약에대해 통제된다 했을때 국민건강보험공단 데이터에서 고혈압 환자들을 어떻게 구분할까? : 일단, 고협압 환자들이 거의 없을꺼같은 젊은층에서 모델링
# - 근데또 혈압은 나이에 따라 (몸무게도 마찬가지) 달라지는데... 

#%% -------- Code Cell 1
import warnings
warnings.filterwarnings('ignore')

#%% -------- Code Cell 2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

# [KO] 상관계수
# [EN] correlation계수
from scipy import stats


#%% -------- Markdown: I. &
# ## I.데이터 불러오기&전처리

#%% -------- Code Cell 3
from argparse import ArgumentParser
from pathlib import Path
parser = ArgumentParser(description="Inspect an explicitly supplied historical 2017 NHIS CSV.")
parser.add_argument("--input", type=Path, required=True, help="Path to NHIS_OPEN_GJ_2017_v1.1.csv")
args = parser.parse_args()
if not args.input.is_file():
    parser.error(f"Input file not found: {args.input}")

#%% -------- Code Cell 4
print(f"Input size: {args.input.stat().st_size:,} bytes")

#%% -------- Code Cell 5
DF = pd.read_csv(args.input, encoding='cp949')

DF.info()

#%% -------- Code Cell 6
# [KO] null 값 확인
# [EN] null 값 check
DF.isnull().sum()

#%% -------- Code Cell 7
# [KO] 열 삭제하기 (시력)(청력)(구강건강관련) ,기준년도,가입자일년번호, 데이터공개일자, *일단=(혈청지오티)AST 등4개
# [EN] 열 삭제하기 (시력)(청력)(구강건강관련) ,기준년도,가입자일년번호, data공개일자, *일단=(혈청지오티)AST 등4개

DF = DF.drop(columns=['기준년도','가입자일련번호','시력(좌)','시력(우)','청력(좌)','청력(우)','구강검진수검여부','치아우식증유무','결손치유무','치아마모증유무','제3대구치(사랑니)이상','치석','데이터공개일자','(혈청지오티)AST','(혈청지오티)ALT','감마지티피'])

#%% -------- Code Cell 8
# [KO] 결측치 제거
# [EN] missing values remove
DF.dropna(inplace = True)

#%% -------- Code Cell 9
# [KO] 변수명 정의
# [EN] 변수명 정의
sex = DF['성별코드']
age = DF['연령대코드(5세단위)']
city = DF['시도코드']
ht = DF['신장(5Cm단위)']
wt = DF['체중(5Kg단위)']
waist = DF['허리둘레']
sbp = DF['수축기혈압']
dbp = DF['이완기혈압']
fbs = DF['식전혈당(공복혈당)']
tchol = DF['총콜레스테롤']
tg = DF['트리글리세라이드']
hdl = DF['HDL콜레스테롤']
ldl = DF['LDL콜레스테롤']
hgb = DF['혈색소']
upro = DF['요단백']
cr = DF['혈청크레아티닌']
smoking = DF['흡연상태']
alcohol = DF['음주여부']

#%% -------- Code Cell 10
# [KO] 데이터 프레임 변수명 전환
# [EN] data 프레임 변수명 전환
DF = DF.rename(columns={'성별코드':'sex', '연령대코드(5세단위)':'age','시도코드':'city','신장(5Cm단위)':'ht','체중(5Kg단위)':'wt','허리둘레':'waist','수축기혈압':'sbp','이완기혈압':'dbp','식전혈당(공복혈당)':'fbs','총콜레스테롤':'tchol','트리글리세라이드':'tg','HDL콜레스테롤':'hdl','LDL콜레스테롤':'ldl','혈색소':'hgb','요단백':'upro','혈청크레아티닌':'cr','흡연상태':'smoking','음주여부':'alcohol'})

#%% -------- Code Cell 11
# [KO] 음주여부 Y=1,N=0
# [EN] 음주여부 Y=1,N=0
DF = DF.replace({'Y':1.0,'N':0.0})

#%% -------- Code Cell 12
# [KO] 정수 실수로 전환
# [EN] 정수 실수로 전환
DF['sex'] = pd.to_numeric(DF['sex'], downcast='float')
DF['age'] = pd.to_numeric(DF['age'], downcast='float')
DF['city'] = pd.to_numeric(DF['city'], downcast='float')
DF['ht'] = pd.to_numeric(DF['ht'], downcast='float')
DF['wt'] = pd.to_numeric(DF['wt'], downcast='float')
DF.info()

#%% -------- Code Cell 13
# [KO] 시간이 너무 많이 걸림
# [EN] 시간이 너무 많이 걸림
# [KO] sns.pairplot(DF)
# [EN] sns.pairplot(DF)
# [KO] plt.show()
# [EN] plt.show()

#%% -------- Code Cell 14
# [KO] 전체 상관계수 분포
# [EN] 전체 correlation계수 distribution
plt.figure(figsize=(12,10))
corr = DF.corr()
sns.heatmap(corr, cmap='viridis')
plt.show()

#%% -------- Code Cell 15
# [KO] 연령대별로 groupby
# [EN] 연령대별로 groupby
plt.figure(figsize=(16,112))
corr_age = DF.groupby('age').corr()
sns.heatmap(corr_age, cmap='viridis')
plt.show()

#%% -------- Markdown: II.
# ## II.데이터 처리후 모델링

#%% -------- Markdown: - IQR 3    outlier
# - IQR 3배 기준 밖의 데이터를 outlier로 처리한다
# - q25% -(q75-q25)*3
# - q75% +(q75-q25)*3

#%% -------- Code Cell 16
# [KO] IQR3 배기준으로 잘라야되는 기준
# [EN] IQR3 배기준으로 잘라야되는 기준
columns = ['ht','wt','waist','sbp','dbp','fbs','tchol','tg','hdl','ldl','hgb','cr']

for i in columns:
    
    print('=======', i, '=======')
    
    q25, q75 = np.quantile(DF[i], 0.25), np.quantile(DF[i], 0.75)
    IQR = q75 - q25
    
    lower = q25 - IQR*3
    upper = q75 + IQR*3
    
    print('lower :', lower)
    print('upper :', upper, '\n')

#%% -------- Code Cell 17
columns = ['ht','wt','waist','sbp','dbp','fbs','tchol','tg','hdl','ldl','hgb','cr']

DF_1 = DF.copy()
for i in columns:
    
    
    q25, q75 = np.quantile(DF[i], 0.25), np.quantile(DF[i], 0.75)
    IQR = q75 - q25
    
    lower = q25 - IQR*3
    upper = q75 + IQR*3
    
    DF_1 = DF_1.loc[DF_1[i].between(lower, upper)].copy()
    print('완료 :', i)


    lowerN = (DF[i]>upper).sum()
    upperN = (DF[i]<lower).sum()
    print('outlier :' , lowerN + upperN)

    
print('outlier가 중복될 수 있음')
DF_1.info()

#%% -------- Markdown: III.
# ## III.모델링
# - 

#%% -------- Code Cell 18


#%% -------- Code Cell 19


#%% -------- Discovered Paths Summary
# The following paths were detected in the original notebook (best-effort scan):
#   - /content/drive
#   - /content/drive/MyDrive/Colab
#   - /datasets/health_care/NHIS_OPEN_GJ_2017_v1.1.csv
