#3. Using the ARexamples.dtadataset, graph the last 100 observations of X over
 #time. Using all of the observations, estimate the AR(1) model,
 #Xt =β1Xt−1 +et.
 #Verify that the coefficient is approximately: ˆ
 #β1 ≈ 0.50
 
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from scipy.stats import norm

n = 1000
e_t = np.random.normal(0, 1, n)
x = np.zeros(n)  # Inicializar la serie temporal
for i in range(1, n):
    x[i] = 0.5 * x[i-1] + e_t[i]  # AR(1) con ruido normal usando e_t
# Graficar las últimas 100 observaciones
plt.figure(figsize=(10, 4))
plt.plot(x[-100:], label='Últimas 100 observaciones de X')
plt.title('Últimas 100 observaciones de X')
plt.xlabel('Tiempo')    
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()
# Estimar el modelo AR(1)
model_ar1 = AutoReg(x, lags=1, trend='n')  # 'n' = no constante
results_ar1 = model_ar1.fit()
print("Resumen del modelo AR(1):")
print(results_ar1.summary())
modelo_Ar1 = ARIMA(x, order=(1, 0, 0), trend='n')  # 'n' = no constante
resultados_ar1 = modelo_Ar1.fit()
print("Resumen del modelo AR(1) con ARIMA:")
print(resultados_ar1.summary())

    # Comprobar si el coeficiente estimado es aproximadamente 0.5 usando un test z
beta_hat = results_ar1.params[0]
beta_se = results_ar1.bse[0]
z_score = (beta_hat - 0.5) / beta_se
p_value = 2 * (1 - norm.cdf(abs(z_score)))
print(f"Coeficiente estimado: {beta_hat:.4f}")
print(f"Error estándar: {beta_se:.4f}")
print(f"Z-score: {z_score:.4f}")
print(f"P-valor para H0: beta=0.5 vs H1: beta!=0.5: {p_value:.4f}")