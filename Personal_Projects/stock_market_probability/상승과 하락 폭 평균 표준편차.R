# 상승한 날 수
num_up_days <- sum(new_dataset$Direction == "상승")

# 오른폭의 평균
mean_up_percent <- mean(new_dataset$Returns[new_dataset$Direction == "상승"])

# 오른폭의 표준편차
sd_up_percent <- sd(new_dataset$Returns[new_dataset$Direction == "상승"])

# 하락한 날 수
num_down_days <- sum(new_dataset$Direction == "하락")

# 하락폭의 평균
mean_down_percent <- mean(new_dataset$Returns[new_dataset$Direction == "하락"])

# 하락폭의 표준편차
sd_down_percent <- sd(new_dataset$Returns[new_dataset$Direction == "하락"])

# 결과 출력
summary(up_data$Returns)
summary(down_data$Returns)

num_up_days
mean_up_percent
sd_up_percent

num_down_days
mean_down_percent
sd_down_percent

# > num_up_days
# [1] 5850
# > mean_up_percent
# [1] 0.00746903
# > sd_up_percent
# [1] 0.007927782

# > num_down_days
# [1] 5105
# > mean_down_percent
# [1] -0.007691171
# > sd_down_percent
# [1] 0.009007688


