#Download the daily adjusted closing price of IBM stock from 1990–2012 using
 #fetchyahooquotes. Take the natural logarithm of this price. Then, using
 #Stata’s D notation, generate a new variable containing the percentage returns of
#IBM’s share price. (The first difference of the logs is equal to the percentage
 #change.) What is the average daily rate of return for IBM during this period?
 #On which date did IBM have its highest percentage returns? On which date did
 #it have its lowest percentage returns?
 
import yfinance as yf
import numpy as np

import pandas as pd
 
IBM = yf.download("IBM", start="1990-01-01", end="2012-12-31")



log_prices = np.log(IBM['Close'])
returns = log_prices.diff().dropna() * 100  # Convierto en porcentaje
if isinstance(returns, pd.DataFrame):
    returns = returns['IBM']
average_return = returns.mean()
max_return = returns.max()
min_return = returns.min()
highest_return_dates = returns[returns == max_return].index
lowest_return_dates = returns[returns == min_return].index


print("Daily Returns of IBM (1990-2012):", returns.head())
print("Average Daily Rate of Return for IBM (1990-2012):", average_return)
print("Highest Percentage Return:", returns.max())
print("Highest Percentage Return Date:", highest_return_dates.strftime('%Y-%m-%d'))
print("Lowest Percentage Return:", returns.min())
print("Lowest Percentage Return Date:", lowest_return_dates.strftime('%Y-%m-%d'))
