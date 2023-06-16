library(quantmod)

# Define the start and end dates
start_date <- as.Date("1980-01-01")
end_date <- Sys.Date() - 1

# Specify the ticker symbol for the desired stock/index
ticker_symbol <- "^GSPC"  # S&P 500 (^GSPC) as an example

# Use the getSymbols() function to fetch the data
getSymbols(ticker_symbol, src = "yahoo", from = start_date, to = end_date)

# Access the S&P 500 data using the ticker symbol
sp500_data <- GSPC

# Extract the adjusted closing prices
prices <- Ad(sp500_data)

# Define the number of days for the moving average and standard deviation
n_days <- 20

# Calculate the moving average
moving_avg <- SMA(prices, n = n_days)

# Calculate the standard deviation
std_dev <- runSD(prices, n = n_days)

# Print the moving average and standard deviation
cat("Moving Average:", moving_avg, "\n")
cat("Standard Deviation:", std_dev, "\n")
