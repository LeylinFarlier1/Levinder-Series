# Given the time series: X = [10, 15, 23, 20, 19], where X1 = 10, X2 =
# 15,. ..calculate, by hand, the first and second lags, and the first and second
#differences of X.

import pandas as pd

X = [10, 15, 23, 20, 19]
X_series = pd.Series(X, index=[1, 2, 3, 4, 5])
# Calculo de rezagos (lags)
X_lag1 = X_series.shift(1)
X_lag2 = X_series.shift(2)
# Cálculo de diferencias
X_diff1 = X_series.diff(1)
X_diff2 = X_diff1.diff(1)
# Mostrar resultados
print("X (Serie Original):")    
print(X_series.tolist())
print("\nPrimer rezago (X_lag1):")  
print(X_lag1.tolist())
print("\nSegundo rezago (X_lag2):")
print(X_lag2.tolist())
print("\nPrimera diferencia (X_diff1):")
print(X_diff1.tolist())