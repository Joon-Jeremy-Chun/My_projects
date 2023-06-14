# 필요한 패키지 로드
library(dplyr)

# 앞면과 뒷면의 선택지 생성
choices <- c("앞면", "뒷면")

# 10000개의 샘플 뽑기
sample_results <- sample(choices, size = 10000, replace = TRUE, prob = c(0.5, 0.5))

# 연속해서 나타나는 경우의 수 계산
count_2 <- 0
count_3 <- 0
count_4 <- 0

for (i in 1:(length(sample_results) - 1)) {
  if (sample_results[i] == sample_results[i+1]) {
    if (sample_results[i] == "앞면") {
      count_2 <- count_2 + 1
      if (i < (length(sample_results) - 2) && sample_results[i+2] == "앞면") {
        count_3 <- count_3 + 1
        if (i < (length(sample_results) - 3) && sample_results[i+3] == "앞면") {
          count_4 <- count_4 + 1
        }
      }
    } else {
      count_2 <- count_2 + 1
      if (i < (length(sample_results) - 2) && sample_results[i+2] == "뒷면") {
        count_3 <- count_3 + 1
        if (i < (length(sample_results) - 3) && sample_results[i+3] == "뒷면") {
          count_4 <- count_4 + 1
        }
      }
    }
  }
}

# 결과 출력
probability_2 <- count_2 / length(sample_results)
probability_3 <- count_3 / length(sample_results)
probability_4 <- count_4 / length(sample_results)

print(paste("앞면 또는 뒷면이 연속해서 2번 나올 확률:", probability_2))
print(paste("앞면 또는 뒷면이 연속해서 3번 나올 확률:", probability_3))
print(paste("앞면 또는 뒷면이 연속해서 4번 나올 확률:", probability_4))

# 데이터프레임으로 변환
df <- data.frame(x = c(2, 3, 4), y = c(count_2, count_3, count_4))

# 그래프 그리기
ggplot(data = df, aes(x = x, y = y)) +
  geom_point() +
  geom_line() +
  labs(title = "연속해서 나올 확률", x = "연속 횟수", y = "확률") +
  theme_minimal()
