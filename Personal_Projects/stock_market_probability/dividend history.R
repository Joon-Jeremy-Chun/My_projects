library(quantmod)

# Define the ticker symbol for the stock of interest
ticker_symbol <- "TMF"  # Replace with your desired stock symbol

# Fetch dividend history using quantmod and Yahoo Finance
dividend_data <- getDividends(ticker_symbol, src = "yahoo")

# Print dividend history
print(dividend_data)

# Fetch future dividend plans using Yahoo Finance API
future_dividends_url <- paste0("https://query1.finance.yahoo.com/v7/finance/quote?&symbols=", ticker_symbol)
future_dividends_json <- jsonlite::fromJSON(future_dividends_url)

# Extract future dividend information from the JSON response
future_dividends <- future_dividends_json$quoteSummary$result[[1]]$dividendDate$fmt

# Print future dividend information
print(future_dividends)
