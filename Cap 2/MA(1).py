import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

u_t = np.random.normal(0, 1, 1000)  # Generar ruido blanco

beta = 0.75
e_t = np.zeros_like(u_t)
for i in range(len(u_t)):
    e_t[i] = u_t[i] + beta * u_t[i-1]

# Graficar la serie temporal
plt.figure(figsize=(10, 4))
plt.plot(e_t[-100:], label='Últimos 100 valores MA(1)')
plt.title('Últimos 100 valores de la serie MA(1)')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()






