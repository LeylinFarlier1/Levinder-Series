 #Let us work concretely with a simple MA(1) model:
 #Xt =ut +βut−1.
 #Andlet us suppose that we have 100 observations of data on X, extending back from
 #t =−99 through t = 0. Now we find ourselves at t = 0 and we wish to forecast
 #next period’s value, X1. First, we estimate the parameter β, and let’s suppose that
 #ˆ
 #β = 0.50. Given the data and our estimated model, we can calculate the residuals
 #from t =−99 through t = 0. These will be our best guess as to the actual errors
 #(residuals approximate errors), and using these, we can forecast Xt. In other words,
 #the procedure is:
 #1. Estimate the model
 #2. Calculate the fitted values from this model
# 3. Calculate the residuals (r) between the data and the fitted values.
 #4. Feed these residuals, iteratively, into the estimated model: Xt = rt + ˆ βrt−1,
 #(X1 | r0) = E(r1)+0.5120095(r0) = 0.5120095(r0)
 #5. Return to step (3) and repeat.
 
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
# Generar datos aleatorios para X
n = 1000
u_t = np.random.normal(0, 1, n)
beta = 0.5  # Coeficiente MA(1)
e_t = np.zeros(n)  # Inicializar la serie temporal X
for i in range(1, n):
    e_t[i] = u_t[i] + beta * u_t[i-1]  # MA(1) con ruido normal usando u_t
# Graficar las últimas 100 observaciones de X
plt.figure(figsize=(10, 4))
plt.plot(e_t[-100:], label='Últimas 100 observaciones de X')
plt.title('Últimas 100 observaciones de X')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()

# Estimar el modelo MA(1)
model_ma1 = ARIMA(e_t, order=(0, 0, 1), trend= 'n')  # MA(1) model
results_ma1 = model_ma1.fit()
print("Resumen del modelo MA(1):")
print(results_ma1.summary())
# Calcular los residuos
residuals = results_ma1.resid
# Graficar los residuos
plt.figure(figsize=(10, 4))
plt.plot(residuals, label='Residuos del modelo MA(1)')
plt.title('Residuos del modelo MA(1)')
plt.xlabel('Tiempo')
plt.ylabel('Residuos')
plt.legend()
plt.tight_layout()
plt.show()
# Predecir el siguiente valor usando los residuos
next_value = residuals[-1] + beta * residuals[-2]
print(f"Valor predicho para X1: {next_value:.4f}")

# Predecir los próximos 10 valores usando el modelo MA(1) estimado
predictions = []
last_resid = residuals[-1]
second_last_resid = residuals[-2]
for i in range(10):
    next_value = beta * last_resid  # Para pasos mayores a 1, la esperanza de nuevos errores es 0
    predictions.append(next_value)
    # Actualizar los residuos para la siguiente iteración
    second_last_resid = last_resid
    last_resid = next_value
print("Próximos 10 valores predichos:")
print(predictions)
plt.figure(figsize=(10, 4))
plt.plot(range(10), predictions, marker='o', label='Predicciones MA(1)')
plt.title('Próximos 10 valores predichos usando MA(1)')
plt.xlabel('Índice de predicción')
plt.ylabel('Valor predicho')
plt.legend()
plt.tight_layout()
plt.show()


