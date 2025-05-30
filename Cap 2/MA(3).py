#2.5
# MA(q)Models
 #Moving Average models can be functions of lags deeper than 1. The general form
 #of the Moving Average model with lags of one through q, an MA(q) model, is:
 #q
 #Xt =ut +β1ut−1 +β2ut−2 +...+βqut−q =
 #where β0 is implicitly equal to one.
 #2.5.1 Estimation
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt

u_t = np.random.normal(0, 1, 1000)  # Generar ruido blanco
b1 = 0.5  # Coeficiente MA(1)
b2 = 0.3  # Coeficiente MA(2)
b3 = 0.1  # Coeficiente MA(3)
et = np.zeros_like(u_t)
for i in range(3, len(u_t)):
    et[i] = u_t[i] + b1 * u_t[i-1] + b2 * u_t[i-2] + b3 * u_t[i-3]  # MA(3) con coeficientes beta

#estimar el modelo MA(3)
model_ma3 = ARIMA(et, order=(0, 0, 3), trend='n')  # MA(3) model
results_ma3 = model_ma3.fit()
print("Resumen del modelo MA(3):")
print(results_ma3.summary())

# Graficar la serie temporal
plt.figure(figsize=(10, 4))
plt.plot(et[-100:], label='Últimos 100 valores MA(3)')
plt.title('Últimos 100 valores de la serie MA(3)')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()

# Calcular los residuos
residuals = results_ma3.resid
# Graficar los residuos
plt.figure(figsize=(10, 4))
plt.plot(residuals, label='Residuos del modelo MA(3)')
plt.title('Residuos del modelo MA(3)')
plt.xlabel('Tiempo')
plt.ylabel('Residuos')
plt.legend()
plt.tight_layout()
plt.show()

# Predecir el siguiente valor usando los residuos
next_value = residuals[-1] + b1 * residuals[-2] + b2 * residuals[-3] + b3 * residuals[-4]
print(f"Valor predicho para X1: {next_value:.4f}")
# Predecir los próximos 10 valores usando el modelo MA(3) estimado
predictions = []
last_resid = residuals[-1]
second_last_resid = residuals[-2]
third_last_resid = residuals[-3]
for i in range(100):
    next_value = b1 * last_resid + b2 * second_last_resid + b3 * third_last_resid  # Para pasos mayores a 1, la esperanza de nuevos errores es 0
    predictions.append(next_value)
    # Actualizar los residuos para la siguiente iteración
    third_last_resid = second_last_resid
    second_last_resid = last_resid
    last_resid = next_value
print("Próximos 10 valores predichos:")
print(predictions)
plt.figure(figsize=(10, 4))
plt.plot(range(100), predictions, marker='o', label='Predicciones MA(3)')
plt.title('Próximos 10 valores predichos usando MA(3)')
plt.xlabel('Índice de predicción')
plt.ylabel('Valor predicho')
plt.legend()
plt.tight_layout()
plt.show()

