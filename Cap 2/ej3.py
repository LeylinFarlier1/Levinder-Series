 #2. Write out the arima estimation command you would use to estimate the
 #following AR processes:
 #(a) Xt = β1Xt−1 +β2Xt−2 +β3Xt−3 +β4Xt−4 +et
 #(b) Xt = β1Xt−1 +β2Xt−2 +β3Xt−3 +et
 #(c) Xt = β1Xt−1 +β4Xt−4 +et
 
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
np.random.seed(0)
n = 1000  # Número de observaciones
e_t = np.random.normal(0, 1, n)
x= np.zeros(n)  # Inicializar la serie temporal
for i in range(4, n):
    x[i] = 0.15 * x[i-1] + 0.1 * x[i-2] + 0.2 * x[i-3] + 0.1 * x[i-4] + e_t [i]  # AR(4) con ruido normal usando e_t
# (a) Xt = β1Xt−1 +β2Xt−2 +β3Xt−3 +β4Xt−4 +et
model_a = ARIMA(x, order=(4, 0, 0), trend= 'n')  # AR(4) sin constante
result_a = model_a.fit()

print("Model (a) Summary:")
print(result_a.summary())

#(b) Xt = β1Xt−1 +β2Xt−2 +β3Xt−3 +et
model_b = ARIMA(x, order=(3, 0, 0), trend='n')  # AR(3) sin constante
result_b = model_b.fit()
print("\nModel (b) Summary:")
print(result_b.summary())
#(c) Xt = β1Xt−1 +β4Xt−4 +et
model_c = AutoReg (x,lags=[1, 4], trend='n')  # AR(1, 4) sin constante
result_c = model_c.fit()
print("\nModel (c) Summary:")
print(result_c.summary())

from matplotlib import pyplot as plt
# Graficar la serie temporal
plt.figure(figsize=(10, 4))
plt.plot(x, label='Serie Temporal AR(4)')
plt.title('Serie Temporal AR(4)')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()