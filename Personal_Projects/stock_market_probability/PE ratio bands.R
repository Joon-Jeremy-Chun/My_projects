library(quantmod)
library(rvest)

# Define the start and end dates
start_date <- as.Date("2010-01-01")
end_date <- Sys.Date() - 1

# Specify the ticker symbol for Samsung on the Korea Stock Exchange
ticker_symbol <- "005930.KS"

# Use the getSymbols() function to fetch the data
getSymbols(ticker_symbol, src = "yahoo", from = start_date, to = end_date)

# Extract the adjusted closing prices
prices <- Ad(get(ticker_symbol))

# Fetch earnings data from Yahoo Finance
earnings_url <- paste0("https://finance.yahoo.com/quote/", ticker_symbol, "/financials?p=", ticker_symbol)
earnings_tables <- read_html(earnings_url) %>%
  html_table(fill = TRUE)
earnings_url

# Extract the quarterly earnings data from the table
earnings_data <- earnings_tables[[2]]
earnings_data <- earnings_data[1:5, c(1, 5:9)]
colnames(earnings_data) <- c("Quarter", "EPS_1Q", "EPS_2Q", "EPS_3Q", "EPS_4Q")

# Convert earnings data to numeric
earnings_data[, -1] <- apply(earnings_data[, -1], 2, function(x) as.numeric(gsub(",", "", x)))

# Combine the quarterly earnings data with the prices
pe_data <- merge(prices, earnings_data, by.x = "Date", by.y = "Quarter")

# Calculate the PE ratio for each quarter
pe_data$PE_1Q <- pe_data$Prices / pe_data$EPS_1Q
pe_data$PE_2Q <- pe_data$Prices / pe_data$EPS_2Q
pe_data$PE_3Q <- pe_data$Prices / pe_data$EPS_3Q
pe_data$PE_4Q <- pe_data$Prices / pe_data$EPS_4Q

# Define the threshold values for each PE ratio band
low_threshold <- 15
high_threshold <- 25

# Categorize the PE ratios into bands for each quarter
pe_data$PE_Band_1Q <- cut(pe_data$PE_1Q, breaks = c(-Inf, low_threshold, high_threshold, Inf), labels = c("Low", "Moderate", "High"))
pe_data$PE_Band_2Q <- cut(pe_data$PE_2Q, breaks = c(-Inf, low_threshold, high_threshold, Inf), labels = c("Low", "Moderate", "High"))
pe_data$PE_Band_3Q <- cut(pe_data$PE_3Q, breaks = c(-Inf, low_threshold, high_threshold, Inf), labels = c("Low", "Moderate", "High"))
pe_data$PE_Band_4Q <- cut(pe_data$PE_4Q, breaks = c(-Inf, low_threshold, high_threshold, Inf), labels = c("Low", "Moderate", "High"))

# Display the PE ratio bands for Samsung by quarter
print(pe_data[, c("Date", "Prices", "PE_1Q", "PE_2Q", "PE_3Q", "PE_4Q", "PE_Band_1Q", "PE_Band_2Q", "PE_Band_3Q", "PE_Band_4Q")])
