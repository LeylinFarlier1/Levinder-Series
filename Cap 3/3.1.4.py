 # Corr(Xt, Xt−1) = (β₁ + β₂β₁ + β₃β₂ + ··· + β_q β_{q−1}) /
#                  (1 + β₁² + β₂² + ··· + β_q²)
#
# Corr(Xt, Xt−2) = (β₂ + β₃β₁ + β₄β₂ + ··· + β_q β_{q−2}) /
#                  (1 + β₁² + β₂² + ··· + β_q²)
#
# Corr(Xt, Xt−3) = (β₃ + β₄β₁ + β₅β₂ + β₆β₃ + ··· + β_q β_{q−3}) /
#                  (1 + β₁² + β₂² + ··· + β_q²)
#
# Corr(Xt, Xt−q) = β_q / (1 + β₁² + β₂² + ··· + β_q²)
#
# Para rezagos k > q:
# Corr(Xt, Xt−k) = 0 / (σ²_u (1 + β₁² + β₂² + ··· + β_q²)) = 0
# Aproximación de la autocorrelación para un modelo AR(3):
# 
# Autocorrelación para k = 1:
# ρ(1) = (β1 + β2·β1 + β3·β2) / (1 + β1² + β2² + β3²)
#
# Autocorrelación para k = 2:
# ρ(2) = (β2 + β3·β1) / (1 + β1² + β2² + β3²)
#
# Autocorrelación para k = 3:
# ρ(3) = β3 / (1 + β1² + β2² + β3²)
#

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
# Definir los parámetros del modelo MA(3)
b1 = 0.4  # Coeficiente MA(1)
b2 = 0.2  # Coeficiente MA(2)
b3 = 0.2  # Coeficiente MA(3)
# Número de observaciones
n = 1000
# Generar ruido blanco
np.random.seed(42)  # Para reproducibilidad
u_t = np.random.normal(0, 1, n)
# Inicializar la serie temporal X
X = np.zeros(n)
# Generar la serie temporal MA(3)
for i in range(3, n):
    X[i] = u_t[i] + b1 * u_t[i-1] + b2 * u_t[i-2] + b3 * u_t[i-3]  # MA(3) con ruido blanco
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
# Estimar el modelo MA(3)
model_ma3 = ARIMA(X, order=(0, 0, 3), trend='n')  # 'n' = no constante
results_ma3 = model_ma3.fit()
print("Resumen del modelo MA(3):")
print(results_ma3.summary())
# Calcular la ACF teórica
def theoretical_acf_ma3(b1, b2, b3, lags):
    """
    Calcula la ACF teórica de un proceso MA(3)
    
    Parámetros:
        b1, b2, b3: coeficientes MA(1), MA(2), MA(3)
        lags: número de retardos hasta los cuales se calcula la ACF (incluye lag 0)
        
    Retorna:
        Lista con los valores de la ACF desde lag 0 hasta lag k
    """
    denom = 1 + b1**2 + b2**2 + b3**2
    acf = [1.0]  # ρ₀ siempre es 1

    if lags >= 1:
        rho1 = (b1 + b2 * b1 + b3 * b2) / denom
        acf.append(rho1)
    if lags >= 2:
        rho2 = (b2 + b3 * b1) / denom
        acf.append(rho2)
    if lags >= 3:
        rho3 = b3 / denom
        acf.append(rho3)
    
    # Para lags mayores que 3, la ACF es 0 para MA(3)
    if lags > 3:
        acf.extend([0.0] * (lags - 3))

    return acf

# Calcular la ACF teórica para 30 lags
lags = 30
acf_theoretical = theoretical_acf_ma3(b1, b2, b3, lags)
from statsmodels.tsa.stattools import acf
acf_sample = acf(X, nlags=lags, fft=True)

# Obtener los coeficientes MA(3) estimados (muestrales)
b1_hat, b2_hat, b3_hat = results_ma3.maparams

# Graficar ambas ACF en el mismo gráfico
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.stem(range(lags + 1), acf_theoretical, linefmt='b-', markerfmt='bo', basefmt=' ', label='ACF Teórica')
ax.stem(range(lags + 1), acf_sample, linefmt='g--', markerfmt='go', basefmt=' ', label='ACF Muestral')
ax.axhline(0, color='red', linestyle='--')
ax.set_title(
    f'ACF Teórica vs. Muestral de un Proceso MA(3)\n'
    f'β₁={b1}, β₂={b2}, β₃={b3} (teóricos) | β̂₁={b1_hat:.3f}, β̂₂={b2_hat:.3f}, β̂₃={b3_hat:.3f} (muestrales)'
)
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')
ax.legend()
plt.tight_layout()
plt.show()
