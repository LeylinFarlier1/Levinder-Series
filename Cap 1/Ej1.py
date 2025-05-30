 #Given the time series: X = [2, 4, 6, 8, 10], where X1 = 2, X2 = 4,...,and
 #X5 = 10, calculate, by hand, the first and second lags of X. Also calculate the
 #first and second differences of X.

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

# Dada la serie temporal X = [2, 4, 6, 8, 10]
X = [2, 4, 6, 8, 10]

# Convierto la lista a una Serie de pandas
X_series = pd.Series(X, index=[1, 2, 3, 4, 5])

# Calculo de rezagos (lags)
X_lag1 = X_series.shift(1)  
X_lag2 = X_series.shift(2)  

# Cálculo de diferencias
X_diff1 = X_series.diff(1)  
# Cálculo de la segunda diferencia
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

print("\nSegunda diferencia (X_diff2):")
print(X_diff2.tolist())
