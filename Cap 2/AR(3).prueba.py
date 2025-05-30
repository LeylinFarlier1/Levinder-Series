#Suppose that X follows an AR(3) process,
 #Xt =β1Xt−1 +β2Xt−2 +β3Xt−3 +et
 #and it has been estimated
 #to be, say,
 #Xt =0.60Xt−1 +0.20Xt−2 +0.10Xt−3 +et

import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from matplotlib import pyplot as plt
# Calcular y graficar la función de respuesta al impulso (IRF) para el modelo ARIMA

#generar datos aleatorios para X
n = 1000
np.random.seed(32)
e_t = np.random.normal(0, 1, n)
b1 = 0.3  # Coeficiente AR(1)
b2 = 0.2  # Coeficiente AR(2)
b1 = 0.1  # Coeficiente AR(3)
x = np.zeros(n)  # Inicializar la serie temporal X
for i in range(3, n):
    x[i] = b1 * x[i-1] + b2 * x[i-2] + b1 * x[i-3] + e_t[i]  # AR(3) con ruido normal usando e_t
# Graficar las últimas 100 observaciones de X
plt.figure(figsize=(10, 4))
plt.plot(x[-100:], label='Últimas 100 observaciones de X')
plt.title('Últimas 100 observaciones de X')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()
# Estimar el modelo AR(2)
model_ar3 = AutoReg(x, lags=3, trend='n')  # 'n' = no constante
results_ar3 = model_ar3.fit()
print("Resumen del modelo AR(3):")
print(results_ar3.summary())
# Estimar el modelo AR(3) usando ARIMA
model_arima = ARIMA(x, order=(3, 0, 0), trend='n')  # 'n' = no constante
results_arima = model_arima.fit()
print("Resumen del modelo AR(3) con ARIMA:")
print(results_arima.summary())


#impulse response function (IRF)
impulse_response = results_arima.impulse_response(steps=30)
plt.figure(figsize=(8, 4))
plt.stem(range(31), impulse_response, use_line_collection=True)
plt.title('Función de Respuesta al Impulso (IRF) del AR(3)')
plt.xlabel('Retraso')
plt.ylabel('Respuesta')
plt.tight_layout()
plt.show()
