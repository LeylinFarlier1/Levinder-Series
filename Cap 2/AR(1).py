from statsmodels.tsa.ar_model import AutoReg
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
np.random.seed(0)
n = 1000  # Número de observaciones
X = np.zeros(n)  # Inicializar la serie temporal
e_t = np.random.normal(0, 1, n)
beta = 0.5  # Coeficiente AR(1)
for i in range(1, n):
    
    X[i] = 0.5 * X[i-1] + e_t[i]  # AR(1) con ruido normal usando e_t

model_nocons = AutoReg(X, lags=1, trend='n')  # 'n' = no constante
results_nocons = model_nocons.fit()
print(results_nocons.summary())
model_a = ARIMA(X, order=(1, 0, 0), trend='n') 
results_a = model_a.fit()
print(results_a.summary())
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(X)
plt.title('Serie temporal AR(1)')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.tight_layout()
plt.show()

#graficar los residuos
residuals = results_nocons.resid
plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.title('Residuos del modelo AR(1)')
plt.xlabel('Tiempo')
plt.ylabel('Residuos')
plt.tight_layout()
plt.show()

# Graficar la función de respuesta al impulso (Impulse Response Function)
from statsmodels.tsa.ar_model import ar_select_order

# Calcular la respuesta al impulso manualmente para AR(1)
impulse = np.zeros(20)
impulse[0] = 1  # impulso unitario en t=0
phi = results_nocons.params[0]
for i in range(1, 20):
    impulse[i] = phi * impulse[i-1]

plt.figure(figsize=(8, 4))
plt.stem(range(20), impulse, use_line_collection=True)
plt.title('Función de Respuesta al Impulso (IRF) para AR(1)')
plt.xlabel('Retraso')
plt.ylabel('Respuesta')
plt.tight_layout()
plt.show()
