# 3.1.1 Theoretical ACF of an AR(1)Process

#Corr(Xt,Xt−1) = β1 
#Corr(Xt,Xt−2) = β1^2
# Corr(Xt,Xt−3) = β1^3

#supongamos que el proceso AR(1) tiene un coeficiente β1 = 0.5
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Generar datos de un proceso AR(1)
n = 1000
np.random.seed(42)  # Para reproducibilidad
u_t = np.random.normal(0, 1, n)  # Ruido blanco
b1 = 0.5  # Coeficiente AR(1)
X = np.zeros(n)  # Inicializar la serie temporal
for i in range(1, n):
    X[i] = b1 * X[i-1] + u_t[i]  # AR(1) con ruido blanco       
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
# Estimar el modelo AR(1)
model_ar1 = ARIMA(X, order=(1, 0, 0), trend='c')  # 'c' = constante
results_ar1 = model_ar1.fit()
print("Resumen del modelo AR(1):")
print(results_ar1.summary())

# Calcular la ACF teórica
def theoretical_acf_ar1(b1, lags):
    return [b1 ** lag for lag in range(lags + 1)]
# Calcular la ACF teórica para 30 lags
lags = 30
acf_theoretical = theoretical_acf_ar1(b1, lags)
# Graficar la ACF teórica
fig, ax = plt.subplots(1, 1, figsize=(12, 5))

# Graficar la ACF teórica
ax.stem(range(lags + 1), acf_theoretical, linefmt='b-', markerfmt='bo', basefmt=' ', label='ACF Teórica')

# Calcular la ACF muestral
from statsmodels.tsa.stattools import acf
acf_sample = acf(X, nlags=lags, fft=True)
ax.stem(range(lags + 1), acf_sample, linefmt='g--', markerfmt='go', basefmt=' ', label='ACF Muestral')

ax.axhline(0, color='red', linestyle='--')
ax.set_title('ACF Teórica vs. Muestral de un Proceso AR(1)')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')
ax.legend()
plt.tight_layout()
plt.show()


