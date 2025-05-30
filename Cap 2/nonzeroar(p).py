import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

#primero, generamos los datos de una seria AR(1) con media distinta de 0
n = 1000
np.random.seed(42)  
e_t = np.random.normal(0, 1, n)  # Ruido blanco
b0 = 500   # Media de la serie AR(1)
b1 = 0.5  # Coeficiente AR(1)
(1 - b1)  # arranque desde la media teórica

X = np.zeros(n)  # Inicializar la serie temporal
for i in range(1, n):
    X[i] = b0 + b1 * X[i-1] + e_t[i]  # AR(1) con media distinta de 0
    X[0] = b0 / (1 - b1)  # Ajustar el primer valor para que la media sea b0
# Graficar las últimas 100 observaciones de X
plt.figure(figsize=(10, 4))
plt.plot(X[-100:], label='Últimas 100 observaciones de X')
plt.axhline(np.mean(X), color='red', linestyle='--', label='Media')
plt.title('Últimas 100 observaciones de X')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()
# Estimar el modelo AR(1) con media distinta de 0
model_ar1 = ARIMA(X, order=(1, 0, 0), trend='c')  # 'c' = constante
results_ar1 = model_ar1.fit()
print("Resumen del modelo AR(1) con media distinta de 0:")
print(results_ar1.summary())


#OPCION 2 , transformar la serie para que tenga media 0
X_centered = X - np.mean(X)  # Centrar la serie para que tenga media 0
# Estimar el modelo AR(1) con media 0
model_ar1_centered = ARIMA(X_centered, order=(1, 0, 0), trend='n')  # 'n' = no constante
results_ar1_centered = model_ar1_centered.fit()
print("Resumen del modelo AR(1) con media 0:")
print(results_ar1_centered.summary())

#plotear la serie centrada
plt.figure(figsize=(10, 4))
plt.plot(X_centered[-100:], label='Últimas 100 observaciones de X centradas')
plt.axhline(0, color='red', linestyle='--', label='Media (0)')
plt.title('Últimas 100 observaciones de X centradas')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()
