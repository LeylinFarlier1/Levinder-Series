# 5. Using the ARexamples.dtadataset, graph the last 100 observations of Z over
 #time. Using all of the observations, estimate the AR(3) model,
# ARMA(p,q)Processes
 #Zt = β1Zt−1 +β2Zt−2 +β3Zt−3 +et.
 #Verify that the coefficients are approximately: ˆ
 #β1 ≈ 0.60, ˆ β2 ≈ 0.20, and ˆ β3 ≈
 #0.10.
 
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
b2 = 0.2  # Coeficiente AR(2)
b3 = 0.1  # Coeficiente AR(3)
z = np.zeros(n)  # Inicializar la serie temporal Z
for i in range(3, n):
    z[i] = b1 * z[i-1] + b2 * z[i-2] + b3 * z[i-3] + e_t[i]  # AR(3) con ruido normal usando e_t
    
# Graficar las últimas 100 observaciones de Z
plt.figure(figsize=(10, 4))
plt.plot(z[-100:], label='Últimas 100 observaciones de Z')  
plt.title('Últimas 100 observaciones de Z')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()

# Estimar el modelo AR(3)
model_ar3 = AutoReg(z, lags=3, trend='n')  # 'n' = no constante
results_ar3 = model_ar3.fit()
print("Resumen del modelo AR(3):")
print(results_ar3.summary())
# Estimar el modelo AR(3) usando ARIMA
model_arima = ARIMA(z, order=(3, 0, 0), trend='n')  # 'n' = no constante
results_arima = model_arima.fit()
print("Resumen del modelo AR(3) con ARIMA:")
print(results_arima.summary())

# Comprobar si los coeficientes estimados son aproximadamente 0.6, 0.2 y 0.1
beta_hat = results_ar3.params
beta_se = results_ar3.bse
z_scores = (beta_hat - np.array([b1, b2, b3])) / beta_se
p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
print(f"Coeficientes estimados: {beta_hat}")
print(f"Errores estándar: {beta_se}")
print(f"Z-scores: {z_scores}")

print(f"P-valores para H0: beta=0.6, 0.2, 0.1 vs H1: beta!=0.6, 0.2, 0.1: {p_values}")


# Si desea probar la significancia conjunta, puede usar el método 'wald_test' de statsmodels:
wald_res = results_ar3.wald_test(np.eye(3))
print("\nTest de Wald para la significancia conjunta de los coeficientes:")
print(wald_res)


