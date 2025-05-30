 #6. Using the ARexamples.dta dataset, estimate the AR(3) model,
 #Zt = β1Zt−1 +β3Zt−3 +et.
 #Notice that this is a restricted model, where the coefficient on the second lag is
 #set to zero. Verify that the estimated coefficients are approximately: ˆ
 #and ˆ β3 ≈ 0.20.
 
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from scipy.stats import norm
# Generar datos aleatorios para Z
n = 1000
e_t = np.random.normal(0, 1, n)
b1 = 0.6  # Coeficiente AR(1)
b3 = 0.2  # Coeficiente AR(3)
z = np.zeros(n)  # Inicializar la serie temporal Z
for i in range(3, n):
    z[i] = b1 * z[i-1] + 0 * z[i-2] + b3 * z[i-3] + e_t[i]  # AR(3) con ruido normal usando e_t
# Graficar las últimas 100 observaciones de Z
plt.figure(figsize=(10, 4))
plt.plot(z[-100:], label='Últimas 100 observaciones de Z')
plt.title('Últimas 100 observaciones de Z')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()

# Estimar el modelo AR(3) restringido
model_ar3_restricted = AutoReg(z, lags=[1, 3], trend='n')  # 'n' = no constante
results_ar3_restricted = model_ar3_restricted.fit()
print("Resumen del modelo AR(3) restringido:")
print(results_ar3_restricted.summary())
# Estimar el modelo AR(3) restringido usando ARIMA
model_arima_restricted = ARIMA(z, order=(3, 0, 0), trend='n')  # 'n' = no constante
results_arima_restricted = model_arima_restricted.fit()
print("Resumen del modelo AR(3) restringido con ARIMA:")
print(results_arima_restricted.summary())
