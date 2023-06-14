# 필요한 패키지 로드
library(dplyr)

# 앞면과 뒷면의 선택지 생성
choices <- c("앞면", "뒷면")

# 10000개의 샘플 뽑기
sample_results <- sample(choices, size = 10000, replace = TRUE, prob = c(0.5, 0.5))

# 연속해서 나타나는 경우의 수 계산
count <- 0
consecutive_count <- 0

for (i in 1:length(sample_results)) {
  if (i == 1) {
    consecutive_count <- 1
  } else {
    if (sample_results[i] == sample_results[i-1]) {
      consecutive_count <- consecutive_count + 1
    } else {
      consecutive_count <- 1
    }
  }
  
  if (consecutive_count >= 2) {
    count <- count + 1
  }
}

# 결과 출력
probability <- count / length(sample_results)
print(paste("앞면 또는 뒷면이 연속해서 나타날 확률:", probability))
