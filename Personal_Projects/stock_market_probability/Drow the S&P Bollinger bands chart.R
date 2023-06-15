library(quantmod)
library(dygraphs)

# Define the start and end dates
start_date <- as.Date("1980-01-01")
end_date <- Sys.Date() - 1

# Specify the ticker symbol for S&P 500 (^GSPC)
ticker_symbol <- "^GSPC"

# Use the getSymbols() function to fetch the data
getSymbols(ticker_symbol, src = "yahoo", from = start_date, to = end_date)

# Access the S&P 500 data using the ticker symbol
sp500_data <- GSPC

# Extract the adjusted closing prices
prices <- Ad(sp500_data)

# Define n-days and sd-value
n_days = 1000
sd_value = 1.96

# Calculate the 20-day simple moving average
sma <- SMA(prices, n = n_days)

# Calculate the standard deviation of prices
sd_series <- runSD(prices, n = n_days)

# Calculate the upper and lower Bollinger Bands
upper_band <- sma + sd_value * sd_series
lower_band <- sma - sd_value * sd_series

# Combine the prices, upper band, and lower band into a data frame
bollinger_data <- cbind(prices, sma, upper_band, lower_band)
colnames(bollinger_data) <- c("Prices", "Moving Average", "Upper Band", "Lower Band")

# Create a dygraph with zooming functionality
dygraph(bollinger_data, main = "S&P 500 with Bollinger Bands") %>%
  dyRangeSelector() %>%
  dySeries("Prices", label = "Prices") %>%
  dySeries("Moving Average", label = "Moving Average") %>%
  dySeries("Upper Band", label = "Upper Band") %>%
  dySeries("Lower Band", label= "Lower Band") %>%
  dyOptions(drawGrid = FALSE, pointSize = 0) %>%
  dyCSS(textConnection(".dygraph-legend {display: none}"))
