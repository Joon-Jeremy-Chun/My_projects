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
tail(moving_avg, 1)


##

# Specify the ticker symbols for the desired indices
sp500_ticker <- "^GSPC"         # S&P 500
nasdaq_ticker <- "^IXIC"        # NASDAQ
djia_ticker <- "^DJI"           # Dow Jones Industrial Average
smallcap2000_ticker <- "^RUT"   # Small Cap 2000

# Use the getSymbols() function to fetch the data for each index
getSymbols(c(sp500_ticker, nasdaq_ticker, djia_ticker, smallcap2000_ticker), src = "yahoo", from = start_date, to = end_date)

# Access the data for each index using the respective ticker symbols
sp500_data <- GSPC
nasdaq_data <- IXIC
djia_data <- DJI
smallcap2000_data <- RUT

# Create separate dataframes for each index
sp500_df <- data.frame(Date = index(sp500_data), SP500 = Ad(sp500_data))
nasdaq_df <- data.frame(Date = index(nasdaq_data), NASDAQ = Ad(nasdaq_data))
djia_df <- data.frame(Date = index(djia_data), DJIA = Ad(djia_data))
smallcap2000_df <- data.frame(Date = index(smallcap2000_data), SmallCap2000 = Ad(smallcap2000_data))

# Print the first few rows of each dataframe
head(sp500_df)
head(nasdaq_df)
head(djia_df)
head(smallcap2000_df)
