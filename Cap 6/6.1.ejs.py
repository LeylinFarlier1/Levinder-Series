# Exercises
# 1. Generate 1000 observations from the model in Eq.(6.4), with β1 = 0.10,
# β12 = 0.40, and β24 = 40. Graph the last 200 observations. Does the data
# appear seasonal? Examine the autocorrelation structure of the data. Can you
# detect seasonality?
# 2. What are the seasonal and non-seasonal AR and MA polynomials implied by
# the following models? Multiply out these polynomials and write out the explicit
# ARIMAmodel.
# (a) ARIMA(1,0,1) ×(1,0,0)12
# (b) ARIMA(2,0,0) ×(0,0,1)4
# (c) ARIMA(0,0,1) ×(0,0,2)12
# (d) ARIMA(0,0,2) ×(1,0,1)4
# 3. For each of the models listed above, what is the characteristic equation?
# 4. For each of the models listed above, what is the inverse characteristic equation?

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --- 1. Generate 1000 observations from the model with specified betas ---
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd

#Define parameters
n = 1000
beta1 = 0.10
beta12 = 0.4

# Usamos 0.40 para beta24 para asegurar que la serie sea estacionaria.
# Un valor de 40 crearía una serie explosiva.

beta24 = 40

X = np.zeros(n)
e = np.random.normal(loc=0, scale=1, size=n)

#
for t in range(25, n):
    term1 = beta1 * X[t-1]
    term2 = beta12 * X[t-12]
    term3 = -(beta1 * beta12 * X[t-13])
    term4 = beta24 * X[t-24]
    term5 = -(beta1 * beta24 * X[t-25])
    X[t] = term1 + term2 + term3 + term4 + term5 + e[t]

plt.figure(figsize=(12, 6))
plt.plot(X[-200:])
plt.title('Last 200 Observations of $X_t$')
plt.xlabel('Observation Index (relative to last 200)')
plt.ylabel('$X_t$')
plt.grid(True)
plt.show()

X_pd = pd.Series(X)
# Plot ACF and PACF en 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
plot_acf(X_pd, lags=40, ax=axes[0])
plot_pacf(X_pd, lags=40, ax=axes[1])                    
axes[0].set_title('ACF of $X_t$')
axes[1].set_title('PACF of $X_t$')
plt.tight_layout()
plt.show()

#No se puede detectar la estacionalidad por que la serie no es estacionaria.
# si fuese estacionaria, se podría observar un patrón periódico en la ACF y PACF.
#creo el nuevo b eta24 para que la serie sea estacionaria
beta24 = 0.4
# Re-generate the time series with the new beta24
Y = np.zeros(n)
for t in range(25, n):
    term1 = beta1 * Y[t-1]
    term2 = beta12 * Y[t-12]
    term3 = -(beta1 * beta12 * Y[t-13])
    term4 = beta24 * Y[t-24]
    term5 = -(beta1 * beta24 * Y[t-25])
    Y[t] = term1 + term2 + term3 + term4 + term5 + e[t]
    
# Plot the last 200 observations again
plt.figure(figsize=(12, 6))
plt.plot(Y[-200:])
plt.title('Last 200 Observations of $Y_t$ with Stationary Series')
plt.xlabel('Observation Index (relative to last 200)')
plt.ylabel('$Y_t$')
plt.grid(True)
plt.show()
# Plot ACF and PACF again
Y_pd = pd.Series(Y)
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
plot_acf(Y_pd, lags=40, ax=axes[0])
plot_pacf(Y_pd, lags=40, ax=axes[1])
axes[0].set_title('ACF of $Y_t$ with Stationary Series')
axes[1].set_title('PACF of $Y_t$ with Stationary Series')
plt.tight_layout()
plt.show()


#2.
#a.
# ARIMA(1,0,1) × (1,0,0)[12]
# - Non-seasonal AR: (1 - β₁L)
# - Non-seasonal MA: (1 + γ₁L)
# - Seasonal AR: (1 - β₁₂L¹²)
# - Seasonal MA: (1)
# - Combined AR: (1 - β₁L)(1 - β₁₂L¹²)
# - Combined MA: (1 + γ₁L)
# - Explicit ARIMA model:
# Xt = β₁Xt−1 + γ₁et−1 + β₁₂Xt−12 - β₁β₁₂Xt−13 + et
# b.
# ARIMA(2,0,0) × (0,0,1)[4]
# - Non-seasonal AR: (1 - β₁L- β₂L²)
# - Seasonal AR: (1)
# - Seasonal MA: (1 + γ₂L⁴)
# - Combined AR: (1 - β₁L - β₂L²)
# - Combined MA: (1 + γ₁L)
# - Explicit ARIMA model:
# Xt = β₁Xt−1 + β₂Xt−2 + γ₁et−1 + γ₂et−4 + et

# c.
# ARIMA(0,0,1) × (0,0,2)[12]
# - Non-seasonal AR: (1)
# - Non-seasonal MA: (1 + γ₁L)
# - Seasonal AR: (1)
# - Seasonal MA: (1 + γ₂L¹² + γ₃L²⁴)
#Combined MA: (1 + γ₁L)(1 + γ₂L¹² + γ₃L²⁴)
# 
#X_t = e_T*((1 + γ₁L)(1 + γ₂L¹² + γ₃L²⁴))
# X_t = e_t (1 + γ₁L + γ₂L^{12} + γ₃L^{24} + γ₁γ₂L^{13} + γ₁γ₃L^{25})
#X_t = e_t + γ₁e_{t-1} + γ₂e_{t-12} + γ₃e_{t-24} + γ₁γ₂e_{t-13} + γ₁γ₃e_{t-25}

# d.
# ARIMA(0,0,2) × (1,0,1)[4]
# - non-seasonal ma(2) = (1 + γ₁L + γ₂L²)
# - seasonal ar(1) = (1 - β₁L⁴)
# - seasonal ma(1) = (1 + γ4L⁴)
#X_t *seasonal_ar(1) =e_t * non-seasonal_ma(2) * seasonal_ma(1)
# Xt(1 - β₁L⁴) = et(1 + γ₁L + γ₂L²)(1 + γ4L⁴)
# Xt(1 - β₁L⁴) = et (1 + γ₁L + γ₂L² + γ₄L⁴ + γ₁γ₄L⁵ + γ₂γ₄L⁶)
#Xt - β₁Xt−4 = et + γ₁et−1 + γ₂et−2 + γ₄et−4 + γ₁γ₄et−5 + γ₂γ₄et−6
# Xt = β₁Xt−4 + et + γ₁et−1 + γ₂et−2 + γ₄et−4 + γ₁γ₄et−5 + γ₂γ₄et−6
