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

# 하락한 날의 수와 연속적인 하락일 수 계산
num_down_days <- sum(new_dataset$Direction == "하락")
num_down_2_days <- sum(new_dataset$Direction == "하락" & lead(new_dataset$Direction) == "하락")
num_down_3_days <- sum(new_dataset$Direction == "하락" & lead(new_dataset$Direction, 2) == "하락")
num_down_4_days <- sum(new_dataset$Direction == "하락" & lead(new_dataset$Direction, 3) == "하락")
num_down_5_days <- sum(new_dataset$Direction == "하락" & lead(new_dataset$Direction, 4) == "하락")

# 2일 연속 하락한 날의 하락률을 저장할 데이터셋 생성
d2_down_data <- new_dataset %>% 
  filter(Direction == "하락" & lead(Direction) == "하락") %>% 
  select(Date, Returns)

# 3일 연속 하락한 날의 하락률을 저장할 데이터셋 생성
d3_down_data <- new_dataset %>% 
  filter(Direction == "하락" & lead(Direction) == "하락" & lead(Direction, 2) == "하락") %>% 
  select(Date, Returns)

# 4일 연속 하락한 날의 하락률을 저장할 데이터셋 생성
d4_down_data <- new_dataset %>% 
  filter(Direction == "하락" & lead(Direction) == "하락" & lead(Direction, 2) == "하락" & lead(Direction, 3) == "하락") %>% 
  select(Date, Returns)

# 5일 연속 하락한 날의 하락률을 저장할 데이터셋 생성
d5_down_data <- new_dataset %>% 
  filter(Direction == "하락" & lead(Direction) == "하락" & lead(Direction, 2) == "하락" & lead(Direction, 3) == "하락" & lead(Direction, 4 == "하락") %>%
  select(Date, Returns)

# 평균 상승률 계산
mean_down_2 <- mean(d2_down_data$Returns)
mean_down_3 <- mean(d3_down_data$Returns)
mean_down_4 <- mean(d4_down_data$Returns)
mean_down_5 <- mean(d5_down_data$Returns)

# 표준 편차 계산
sd_down_2 <- sd(d2_down_data$Returns)
sd_down_3 <- sd(d3_down_data$Returns)
sd_down_4 <- sd(d4_down_data$Returns)
sd_down_5 <- sd(d5_down_data$Returns)

# 결과 출력
mean_down_2
mean_down_3
mean_down_4
mean_down_5

sd_down_2
sd_down_3
sd_down_4
sd_down_5         

# > # 결과 출력
#   > mean_down_2
# [1] -0.007449442
# > mean_down_3
# [1] -0.007221588
# > mean_down_4
# [1] -0.007444483
# > mean_down_5
# [1] 0.0004044179
# > sd_down_2
# [1] 0.00775593
# > sd_down_3
# [1] 0.007470992
# > sd_down_4
# [1] 0.007868371
# > sd_down_5         
# [1] 0.01133856