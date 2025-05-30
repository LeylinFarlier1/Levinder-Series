import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Simulamos una serie temporal X
np.random.seed(0)
n = 100
X = pd.Series(np.random.normal(0, 1, n))

# Empirical ACFs are not the result of a model. They are a description of data.
# They can be calculated much like any other correlation.

# To calculate an empirical ACF in Python, create a new variable that is the lag of X
# Let us call it LagX. Treat this new variable like any other variable and 
# calculate the correlation between X and LagX.

LagX = X.shift(1)  # Esto es como "gen LagX = L.X" en Stata
correlation = X.corr(LagX)  # Esto es como "correlate X LagX" en Stata
print(f"Correlación entre X y su primer rezago: {correlation:.3f}")

# In fact, Python is quite smart. There is no need to create the new variable.
# Rather, we may estimate the correlation between X and its lag more directly by:

autocorr_lag1 = X.autocorr(lag=1)
print(f"Autocorrelación en el rezago 1: {autocorr_lag1:.3f}")

# which only calculates the autocorrelation at a lag of one.
# To calculate deeper lags:
for lag in range(1, 6):
    print(f"Lag {lag}: autocorrelación = {X.autocorr(lag=lag):.3f}")

# Alternatively, we can use plot_acf from statsmodels
# which provides the empirical ACF (and PACF), as well as a text-based picture of the two.

plot_acf(X, lags=20)
plt.title("Función de Autocorrelación (ACF)")
plt.show()

#como la variable x es ruido blanco, la ACF debería ser cero para todos los rezagos excepto el rezago cero. 