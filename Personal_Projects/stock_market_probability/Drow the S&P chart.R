library(quantmod)

# Define the start and end dates
start_date <- as.Date("1980-01-01")
end_date <- Sys.Date() - 1

# Specify the ticker symbol for S&P 500 (^GSPC)
ticker_symbol <- "^GSPC"

# Use the getSymbols() function to fetch the data
getSymbols(ticker_symbol, src = "yahoo", from = start_date, to = end_date)

# Access the S&P 500 data using the ticker symbol
sp500_data <- GSPC

# Plot the chart using chartSeries()
chartSeries(sp500_data, theme = "white", name = "S&P 500")

## zoom in out chart

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

# Create a dygraph with zooming functionality
dygraph(prices, main = "S&P 500") %>%
  dyRangeSelector() %>%
  dyOptions(drawGrid = FALSE, pointSize = 0) %>%
  dyCSS(textConnection(".dygraph-legend {display: none}"))
