# Estacionalidad MA Multiplicativa
# Anteriormente, consideramos la estacionalidad MA aditiva.
# Ahora, podemos tener estacionalidad MA multiplicativa.
# Un polinomio MA(q) no estacional se podría expresar como:
# Xt = ut + γ1 * ut-1 + γ2 * ut-2 + ... + γq * ut-q
# Esto también se puede escribir en notación de operador de retardo (lag operator, L) como:
# Xt = ut * (1 + γ1*L + γ2*L^2 + ... + γq*L^q)  -- Nota: el texto original usa 'ut(L)' que es una simplificación
#                                                     donde L aquí representa el polinomio MA.
#                                                     Comúnmente, MA es (1 + theta_1*L + ...), no 'ut(L)'.
#                                                     Aclaremos que el texto original usa 'u_t(L)' de forma no estándar
#                                                     para representar la aplicación del polinomio MA a 'u_t'.
# Siendo más precisos, un polinomio MA(q) es:
# θ(L) = (1 + θ_1 L + θ_2 L^2 + ... + θ_q L^q)
# Entonces, X_t = θ(L) * e_t

# Ahora, introducimos un polinomio MA(Q) estacional multiplicativo:
# θ_s(L^s) = (1 + γ_s L^s + γ_{2s} L^{2s} + ... + γ_{Qs} L^{Qs})
# Donde 's' es el período estacional (e.g., 12 para datos mensuales).

# Al multiplicar estos polinomios, se produce el modelo con estacionalidad MA multiplicativa:
# X_t = θ(L) * θ_s(L^s) * e_t
# (En el texto original se usa 'u_t' en lugar de 'e_t' para el ruido blanco)

# Estimación de este modelo en Stata:
# Para Stata, el comando podría ser `arima X, ma(q) sma(Q,s)`.
# Por ejemplo, para un MA(1) no estacional y un MA(1) estacional con período 12:
# . arima X, ma(1) sma(1,12)
# La opción `sma(P,D,Q,s)` se refiere a los parámetros estacionales:
# P: número de rezagos AR estacionales multiplicativos (ninguno en este caso).
# D: número de diferenciaciones estacionales necesarias para inducir estacionariedad (ninguna en este caso).
# Q: número de rezagos MA estacionales multiplicativos (hay uno en este ejemplo: Q=1).
# s: longitud estacional (12 meses).

# Otro comando equivalente en Stata es:
# . arima X, mma(lags,s)
# Por ejemplo, `mma(1,12)` indica un modelo de media móvil multiplicativo con un rezago estacional
# en una longitud estacional de 12 meses. Esto es similar a `sma(0,0,1,12)`.

# Ejemplo: ¿Cómo se vería un modelo ARIMA(0,0,1)×(0,0,2)12?
# No tenemos términos AR (p=0, P=0).
# Tenemos un término MA no estacional (q=1).
# Tenemos dos rezagos MA estacionales en un período de doce (Q=2, s=12).

# Los polinomios serían:
# MA no estacional: θ(L) = (1 + β_1 L)  -- Nota: el texto usa (1-Lβ1) lo que implica que β1 es negativo en la expansión.
#                                             Adoptaremos la convención del texto para este ejemplo.
#                                             Si MA = (1 - b1 L), entonces X_t = e_t - b1 e_{t-1}.
# MA estacional: θ_s(L^12) = (1 + β_{12} L^{12} + β_{24} L^{24}) -- Nota: el texto usa (1-L12β12 -L24β24)
#                                                                 lo que nuevamente implica coeficientes negativos.

# Siguiendo la notación del texto original (con coeficientes negativos para MA, lo cual es una convención menos común pero posible):
# Polinomio MA no estacional: (1 - β_1 L)
# Polinomio MA estacional: (1 - β_{12} L^{12} - β_{24} L^{24})

# Combinando los polinomios MA multiplicativamente:
# X_t = e_t * (1 - β_1 L) * (1 - β_{12} L^{12} - β_{24} L^{24})
# X_t = e_t * [ 1 * (1 - β_{12} L^{12} - β_{24} L^{24}) - β_1 L * (1 - β_{12} L^{12} - β_{24} L^{24}) ]
# X_t = e_t * [ 1 - β_{12} L^{12} - β_{24} L^{24} - β_1 L + β_1 β_{12} L^{13} + β_1 β_{24} L^{25} ]
# Reordenando por el lag del operador L:
# X_t = e_t * [ 1 - β_1 L - β_{12} L^{12} + β_1 β_{12} L^{13} - β_{24} L^{24} + β_1 β_{24} L^{25} ]

# O, explícitamente (aplicando el operador a e_t):
# X_t = e_t - β_1 e_{t-1} - β_{12} e_{t-12} + β_1 β_{12} e_{t-13} - β_{24} e_{t-24} + β_1 β_{24} e_{t-25}

#Xt =ut (L)θ Ls 
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
# Set random seed for reproducibility
np.random.seed(42)
# Define parameters
n = 1000  # Number of observations
beta1 = 0.30
beta12 = 0.2  # Coefficient for lag 12
beta24 = 0.1  # Coefficient for lag 24

# Initialize the time series
X = np.zeros(n)  # Initialize the time series
e = np.random.normal(loc=0, scale=1, size=n)  # Generate white noise
# Generate the time series with AR and MA components
for t in range(25, n):
    term1 = beta1 * e[t-1]  # AR(1) component
    term2 = beta12 * e[t-12]  # MA(12) component
    term3 = (beta1 * beta12 * e[t-13])  # Interaction term
    term4 = beta24 * e[t-24]  # MA(24) component
    term5 = (beta1 * beta24 * e[t-25])  # Interaction term
    X[t] = term1 + term2 + term3 + term4 + term5 + e[t]  # Combine all components
    
# Convert to pandas Series for easier handling
X_pd = pd.Series(X)
# Plot the last 200 observations of the time series
plt.figure(figsize=(12, 6))
plt.plot(X_pd[-200:])
plt.title('Last 200 Observations of $X_t$')
plt.xlabel('Observation Index (relative to last 200)')
plt.ylabel('$X_t$')
plt.grid(True)
plt.show()
# Estimate the ARIMA model
model = ARIMA(X_pd, order=(1, 0, 0), seasonal_order=(0, 0, 2, 12))
results = model.fit()
# Print the summary of the model
print("Summary of the ARIMA(1,0,0) × (0,0,2)[12] model:")
print(results.summary())

#plot the ACF and PACF
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(X_pd, lags=40, ax=axes[0])
plot_pacf(X_pd, lags=40, ax=axes[1])
axes[0].set_title('ACF of $X_t$')
axes[1].set_title('PACF of $X_t$')
plt.tight_layout()
plt.show()
