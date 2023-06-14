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

# 2일 연속으로 상승한 날짜 수 계산
num_2_up_days <- new_dataset %>%
  filter(Direction == "상승" & lag(Direction) == "상승") %>%
  nrow()

# 2일 연속으로 하락한 날짜 수 계산
num_2_down_days <- new_dataset %>%
  filter(Direction == "하락" & lag(Direction) == "하락") %>%
  nrow()

# 결과 출력
num_2_up_days
num_2_down_days

# 데이터의 총 일 수 계산
total_days <- nrow(GSPC)

# 결과 출력
total_days
