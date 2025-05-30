#how are the Xs at different lags correlated with each other? Our MA(1) model,
 #yet again, is
 #Xt =ut +βut−1,
 
 #Corr(X,Xt−1) = E(XtXt−1) / var(Xt) = β / (1 + β^2)
# Corr(X,Xt−2) = E(XtXt−2) / var(Xt) = 0
# Corr(X,Xt−3) = E(XtXt−3) / var(Xt) = 0

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt

# Generar datos aleatorios para X
n = 1000
u_t = np.random.normal(0, 1, n)
# Coeficientes MA(1)
b1 = 0.5  # Coeficiente MA(1)

x_t = np.zeros_like(u_t)
for i in range(1, n):
    x_t[i] = u_t[i] + b1 * u_t[i-1]  # MA(1) con coeficiente beta
    
# Graficar las últimas 100 observaciones de X
plt.figure(figsize=(10, 4))
plt.plot(x_t[-100:], label='Últimas 100 observaciones de X')
plt.axhline(np.mean(x_t), color='red', linestyle='--', label='Media')
plt.title('Últimas 100 observaciones de X')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()

# Estimar el modelo MA(1)
model_ma1 = ARIMA(x_t, order=(0, 0, 1), trend='c')  # 'c' = constante
results_ma1 = model_ma1.fit()
print("Resumen del modelo MA(1):")
print(results_ma1.summary())

# Calcular la ACF teórica
def theoretical_acf_ma1(b1, lags):
    """
    Calcula la ACF teórica de un proceso MA(1)
    
    Parámetros:
        beta: coeficiente MA(1)
        lags: número de retardos hasta los cuales se calcula la ACF (incluye lag 0)
    
    Retorna:
        Lista con los valores de la ACF desde lag 0 hasta lag k
    """
    acf = [1.0]  # ρ₀ = 1
    if lags >= 1:
        rho1 = b1 / (1 + b1**2)  # ρ₁
        acf.append(rho1)
    # A partir de lag 2, todo es cero
    acf.extend([0.0] * (lags - len(acf) + 1))
    return acf


# Calcular la ACF teórica para 30 lags
lags = 30
acf_theoretical = theoretical_acf_ma1(b1, lags)
# Graficar la ACF teórica
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
# Graficar la ACF teórica
ax.stem(range(lags + 1), acf_theoretical, linefmt='b-', markerfmt='bo', basefmt=' ', label='ACF Teórica')
# Calcular la ACF muestral
from statsmodels.tsa.stattools import acf
acf_sample = acf(x_t, nlags=lags, fft=True)
ax.stem(range(lags + 1), acf_sample, linefmt='g--', markerfmt='go', basefmt=' ', label='ACF Muestral')
ax.axhline(0, color='red', linestyle='--')
ax.set_title('ACF Teórica vs. Muestral de un Proceso MA(1)')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')
ax.legend()
plt.tight_layout()
plt.show()
