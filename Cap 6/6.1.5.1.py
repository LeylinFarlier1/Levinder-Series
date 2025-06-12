# 6.1.5 Estacionalidad MA (Media Móvil)

# Las ventas minoristas de esta temporada (Xt) podrían depender directamente de las ventas del año pasado (Xt-12)
# a través de un término autorregresivo (AR). Sin embargo, también podrían estar relacionadas
# mediante los términos de error del modelo. Esto significa que podríamos tener estacionalidad
# a través de un término de media móvil (MA). Estos términos pueden incorporarse de forma aditiva o multiplicativa.

# Estacionalidad MA Aditiva
# Un ejemplo de estacionalidad MA aditiva es la ecuación:
# Xt = et + γ12 * ut-12
# En esta expresión, 'et' representa el término de error actual (ruido blanco).
# 'ut-12' es un término de error adicional proveniente de 12 períodos anteriores,
# el cual se suma directamente a las ventas actuales, de ahí el nombre de "estacionalidad aditiva".
# 'γ12' es el coeficiente que cuantifica el impacto de este término de error estacional pasado.
# Este tipo de modelo, al igual que otras estructuras ARIMA, puede estimarse
# utilizando software estadístico como Stata.

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
#import arima
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
# Set random seed for reproducibility
np.random.seed(42)

n= 1000  # Number of observations
gamma1= 0.5
gamma12 = 0.3  # Coefficient for lag 12

X = np.zeros(n)  # Initialize the time series
e = np.random.normal(loc=0, scale=1, size=n)  # Generate white noise
# Generate the time series with MA(1) and MA(12) components
for t in range(12, n):
    term1 = gamma1*e[t-1] + e[t]  # Current error term
    term2 = gamma12 * e[t-12]  # Error term from 12 periods ago
    X[t] = term1 + term2  # Combine current and lagged error terms
    
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

#estimate the ARIMA model
model = ARIMA(X_pd, order=(0, 0, 1), seasonal_order=(0, 0, 1, 12))
Results = model.fit()
# Print the summary of the model
print("Summary of the ARIMA(0,0,1) × (0,0,1)[12] model:")
print(Results.summary())
