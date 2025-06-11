# ------------------------------------------------------------
# Notas: 6.1.4 Multiplicative Seasonality
# ------------------------------------------------------------

# En muchos contextos, la estacionalidad no es puramente aditiva.
# Por ejemplo: si las ventas de noviembre son altas, es probable que las de diciembre también lo sean.
# Esto sugiere dependencia estacional persistente → la estacionalidad se modela mejor de forma **multiplicativa**.

# Box y Jenkins (1976) propusieron una forma multiplicativa de incluir términos estacionales.

# Modelo ARIMA(1,0,0):
# Xt = β₁ * Xt−1 + eₜ   →   (1 − β₁L)Xt = eₜ

# Queremos incluir un componente estacional multiplicativo:
# φ(L⁴) = (1 − β₄L⁴)

# Modelo multiplicativo:
# (1 − β₁L)(1 − β₄L⁴)Xt = eₜ
# Al expandir:
# Xt − β₁Xt−1 − β₄Xt−4 + β₁β₄Xt−5 = eₜ

# El modelo resultante tiene lags en 1, 4 y 5 → se logra parquedad con pocos parámetros.

# Notación: ARIMA(1,0,0) × (1,0,0)[4]
# El símbolo × indica multiplicación entre los polinomios estacional y no estacional.
# El subíndice [4] indica periodicidad trimestral.

# Ejemplo más complejo:
# ARIMA(1,0,1) × (2,0,0)[4]:
# - Polinomio AR: (1 − β₁L)(1 − β₄L⁴ − β₈L⁸)
# - Polinomio MA: (1 + γ₁L)

# Expandido:
# Xt − β₁Xt−1 − β₄Xt−4 + β₁β₄Xt−5 − β₈Xt−8 + β₁β₈Xt−9 = ut + γ₁ut−1

# Ejemplo anual:
# ARIMA(1,0,0) × (2,0,0)[12]
# - (1 − β₁L)(1 − β₁₂L¹² − β₂₄L²⁴)Xt = eₜ

# Modelo explícito:
# Xt = β₁Xt−1 + β₁₂Xt−12 − β₁β₁₂Xt−13 + β₂₄Xt−24 − β₁β₂₄Xt−25 + eₜ


#Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

#Consider an ARIMA(1,0,0) model such as
 #Xt =β1Xt−1 +et
 #X(1−β1L) = e
 #X (L) =e
 
 #Create a time series AR (1,0,0)
np.random.seed(42)  # For reproducibility
e_t = np.random.normal(0, 1, 100)  # Generate white noise
X = np.zeros(100)  # Initialize the time series
for t in range(1, 100):
    X[t] = 0.5 * X[t-1] + e_t[t]  # AR(1) process

# Nowsuppose that you want to include a seasonal lag polynomial such as
#φ L4 = 1−β4L4
# but you want to include it multiplicatively, so that
# X (L)φ(L4) = e
#X(1−β1L) 1−β4L4 =e

#creo la variable para el proceso estacional multiplicativo
#seasonal_order=(1, 0, 0, 4)
zt = np.zeros(100)  # Initialize the seasonal component
for t in range(4, 100):
    zt[t] = 0.05 * zt[t-4] + e_t[t]  # Seasonal component with lag 4


# Create a seasonal ARIMA(1,0,0) × (1,0,0)[4] model
model = ARIMA(X + zt, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4))
# Fit the model
results = model.fit()
print("Summary of the Seasonal ARIMA(1,0,0) × (1,0,0)[4] model:")
print(results.summary())
# Plot the time series and the fitted values
plt.figure(figsize=(12, 6))
plt.plot(X + zt, label='Time Series with Seasonal Component', color='blue')
plt.plot(results.fittedvalues, label='Fitted Values', color='red')
plt.title('Seasonal ARIMA(1,0,0) × (1,0,0)[4] Model')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()