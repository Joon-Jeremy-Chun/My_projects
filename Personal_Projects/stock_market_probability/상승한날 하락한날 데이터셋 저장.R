# 필요한 패키지 로드
library(quantmod)
library(dplyr)

# 주식 데이터 불러오기
symbol <- "^GSPC"  # S&P 500 지수의 심볼
start_date <- "1980-01-01"  # 시작일
end_date <- "2023-06-14"  # 종료일

# Yahoo Finance에서 주식 데이터 불러오기
getSymbols(symbol, from = start_date, to = end_date)

# 전날 대비 수익률 계산
returns <- dailyReturn(Cl(GSPC))

# 새로운 데이터셋 생성
new_dataset <- data.frame(Date = index(returns),
                          Returns = as.vector(returns))

# Direction 열 추가
new_dataset <- new_dataset %>%
  mutate(Direction = ifelse(Returns >= 0, "상승", "하락"))


#데이터 명명
up_data <- new_dataset[new_dataset$Direction == "상승", ]
down_data <- new_dataset[new_dataset$Direction == "하락", ]
head(up_data)
head(down_data)

