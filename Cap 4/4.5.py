# =============================================================================
# AR(1) como MA(∞)
# -----------------------------------------------------------------------------
# Un proceso AR(1):       Xt = β * Xt−1 + et
#
# puede representarse como un proceso MA(∞) **si es estacionario**.
#
# Al realizar sustituciones sucesivas:
#     Xt = et + β*et−1 + β²*et−2 + β³*et−3 + ...
#
# Obtenemos una representación MA(∞):
#     Xt = et + γ1*et−1 + γ2*et−2 + γ3*et−3 + ...
# donde:
#     γ1 = β,   γ2 = β²,   γ3 = β³,   etc.
#
# Pero esta expansión solo es válida si la suma infinita es **convergente**.
# Esto ocurre **solo si |β| < 1**, es decir, cuando el proceso es **estacionario**.
#
# Si |β| ≥ 1, entonces los términos β^k no tienden a cero y la serie diverge,
# por lo tanto, el proceso **no puede representarse como MA(∞)**.
# =============================================================================

import pandas as pd
import numpy as np
#import ARIMA
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Configuración de la semilla para reproducibilidad
np.random.seed(42)
# Número de observaciones
n = 2000
# Coeficiente AR(1) dentro del rango estacionario
beta = 0.5  # Coeficiente AR(1) dentro del rango estacionario
# Generar serie AR(1) estacionaria
serie_ar1 = np.zeros(n)
for t in range(1, n):
    et = np.random.normal(0, 1)  # Ruido blanco
    serie_ar1[t] = beta * serie_ar1[t - 1] + et
# Convertir la serie en un DataFrame
df_ar1 = pd.DataFrame({'serie_ar1': serie_ar1})

#Ajustar un modelo ARIMA(1, 0, 0) (equivalente a AR(1))
modelo_ar1 = ARIMA(df_ar1['serie_ar1'], order=(1, 0, 0)).fit()
# Imprimir el resumen del modelo
print(modelo_ar1.summary())
# Graficar la serie y la predicción del modelo AR(1)
plt.figure(figsize=(10, 5))
plt.plot(df_ar1['serie_ar1'], label='Serie AR(1)', color='blue')
plt.axhline(0, color='red', linestyle='--', label='Media = 0')
plt.title('Serie AR(1) Estacionaria')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()

#ajusto el modelo ma(∞) con ARIMA(0, 0, 20)
modelo_ma_infinito = ARIMA(df_ar1['serie_ar1'], order=(0, 0, 30)).fit()


# Gráfico del forecast del modelo AR(1)
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Predicción AR(1)
axs[0].axhline(0, color='red', linestyle='--', label='Media = 0')
axs[0].plot(modelo_ar1.fittedvalues, label='Predicción AR(1)', color='orange')
axs[0].set_title('Predicción del Modelo AR(1)')
axs[0].set_ylabel('Valor')
axs[0].legend()

# Predicción MA(∞)
axs[1].axhline(0, color='red', linestyle='--', label='Media = 0')
axs[1].plot(modelo_ma_infinito.fittedvalues, label='Predicción MA(∞)', color='green')
axs[1].set_title('Predicción del Modelo MA(∞)')
axs[1].set_xlabel('Tiempo')
axs[1].set_ylabel('Valor')
axs[1].legend()

plt.tight_layout()
plt.show()

#h


