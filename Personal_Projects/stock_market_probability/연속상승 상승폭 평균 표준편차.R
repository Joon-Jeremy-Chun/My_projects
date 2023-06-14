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


# 2일 연속 상승한 날의 상승률을 저장할 데이터셋 생성
d2_up_data <- new_dataset %>% 
  filter(Direction == "상승" & lag(Direction) == "상승") %>% 
  select(Date, Returns)

# 3일 연속 상승한 날의 상승률을 저장할 데이터셋 생성
d3_up_data <- new_dataset %>% 
  filter(Direction == "상승" & lag(Direction) == "상승" & lag(lag(Direction)) == "상승") %>% 
  select(Date, Returns)

# 4일 연속 상승한 날의 상승률을 저장할 데이터셋 생성
d4_up_data <- new_dataset %>% 
  filter(Direction == "상승" & lag(Direction) == "상승" & lag(lag(Direction)) == "상승" & lag(lag(lag(Direction))) == "상승") %>% 
  select(Date, Returns)

# 5일 연속 상승한 날의 상승률을 저장할 데이터셋 생성
d5_up_data <- new_dataset %>% 
  filter(Direction == "상승" & lag(Direction) == "상승" & lag(lag(Direction)) == "상승" & lag(lag(lag(Direction))) == "상승" & lag(lag(lag(lag(Direction)))) == "상승") %>% 
  select(Date, Returns)




# 평균 상승률 계산
mean_up_2 <- mean(d2_up_data$Returns)
mean_up_3 <- mean(d3_up_data$Returns)
mean_up_4 <- mean(d4_up_data$Returns)
mean_up_5 <- mean(d5_up_data$Returns)

# 표준 편차 계산
sd_up_2 <- sd(d2_up_data$Returns)
sd_up_3 <- sd(d3_up_data$Returns)
sd_up_4 <- sd(d4_up_data$Returns)
sd_up_5 <- sd(d5_up_data$Returns)

# 결과 출력
mean_up_2
mean_up_3
mean_up_4
mean_up_5

sd_up_2
sd_up_3
sd_up_4
sd_up_5

# > # 결과 출력
#   > mean_up_2
# [1] 0.006957741
# > mean_up_3
# [1] 0.006380299
# > mean_up_4
# [1] 0.005946095
# > mean_up_5
# [1] 0.005168117
# > sd_up_2
# [1] 0.007039878
# > sd_up_3
# [1] 0.006594506
# > sd_up_4
# [1] 0.005825113
# > sd_up_5
# [1] 0.004765803