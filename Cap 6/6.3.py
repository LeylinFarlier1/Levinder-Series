# Los conceptos de invertibilidad y estabilidad se generalizan bien de los rezagos no estacionales
# a los rezagos estacionales.

# Estabilidad de un proceso ARIMA estacional:
# Un proceso ARIMA estacional es estable (estacionario) si las raíces de sus dos polinomios
# característicos, φ(L) (polinomio AR no estacional) y Φ(L^s) (polinomio AR estacional),
# se encuentran fuera del círculo unitario.
# De manera equivalente, las raíces de los polinomios inversos (1/φ(L) y 1/Φ(L^s))
# deben estar dentro del círculo unitario.

# La inclusión de rezagos estacionales multiplicativos no altera esta condición;
# de hecho, ayuda en el análisis. Si un valor particular 'z' hace que Φ(L^s) sea igual a cero,
# entonces 'z' también será una raíz de la combinación multiplicativa φ(L)Φ(L^s),
# ya que cualquier cosa multiplicada por cero es cero.

# Comando equivalente en Stata:
# Para verificar la estabilidad y la invertibilidad en Stata, se usaría el comando:
# . estat aroots
# Este comando proporciona un gráfico del círculo unitario y traza las raíces de los polinomios
# característicos inversos (tanto AR como MA).

#Por ejemplo, si tenemos un modelo ARIMA(1,0,0) × (1,0,0)[4]

# podemos verificar la estabilidad y la invertibilidad de los polinomios AR y ar estacionales 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

np.random.seed(42) # Para reproducibilidad
n_observations = 1000

e = np.random.normal(0, 1, n_observations)  # Ruido blanco
X = np.zeros(n_observations)  # Inicializar la serie temporal
for t in range(1, n_observations):
    X[t] = 0.4 * X[t-1] + e[t] + 0.2 * X[t-4] + e[t-1]*0.3 # AR(1) y AR estacional (4)
X_pd = pd.Series(X)
# Ajustar el modelo SARIMAX
model = SARIMAX(X_pd, order=(1, 0, 1), seasonal_order=(1, 0, 0, 4))
results = model.fit(disp=False)
# Resumen del modelo
print("Resumen del modelo SARIMAX(1,0,0) × (1,0,0)[4]:")
print(results.summary())

# --- Encontrar las raíces del modelo ajustado ---

print("\n" + "="*50)
print("ANÁLISIS DE RAÍCES")
print("="*50)

# Obtener las raíces del polinomio autorregresivo (AR)
ar_roots = results.arroots
print("\nRaíces del Polinomio Autorregresivo (AR):")
print(ar_roots)

# El modelo es estacionario si el módulo de todas las raíces AR es > 1
print("\nMódulo de las raíces AR (deben ser > 1 para estacionariedad):")
print(np.abs(ar_roots))

# Obtener las raíces del polinomio de media móvil (MA)
# En este caso, estará vacío porque no hay términos MA (q=0, Q=0)
ma_roots = results.maroots
print("\nRaíces del Polinomio de Media Móvil (MA):")
print(ma_roots)

print("="*50)