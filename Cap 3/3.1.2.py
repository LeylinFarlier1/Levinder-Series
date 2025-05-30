#3.1.2 TheoreticalACFofanAR(p)Process
 #Let us now suppose that Xt follows a general AR(p) process:
 #Xt =β1Xt−1 +β2Xt−2 +...+βpXt−p +et.
 #(3.15)
 # φ0 = β1φ1 +β2φ2 +σ2
 #φ1 = β1φ0 +β2φ1  (3.23)
 
 #φ2 = β1φ1 +β2φ0.(3.24)
 
 #The last two lines establish a recursive pattern: φs = β1φs−1 + β2φs−2.
 
 #Corr(Xt,Xt−1) =  = φ1 / φ0 = ( β1 / (1−β2) φ0) / φ0 = β1 / (1−β2)
 #corr(Xt,Xt−2) = φ2 / φ0 = β^2 / (1−β2) + B2
# corr(Xt,Xt−3) = φ3 / φ0 = β1 φ2 + B2  φ1 
# corr(Xt,Xt−4) = φ4 / φ0 = β1 φ3 + B2  φ2
# corr(Xt,Xt−5) = φ5 / φ0 = β1 φ4 + B2  φ3

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
# Definir los parámetros del modelo AR(3)
b1 = 0.5  # Coeficiente AR(1)
b2 = 0.2  # Coeficiente AR(2)

# Número de observaciones
n = 1000
# Generar ruido blanco
np.random.seed(42)  # Para reproducibilidad
u_t = np.random.normal(0, 1, n)
# Inicializar la serie temporal X
X = np.zeros(n)
# Generar la serie temporal AR(2)
for i in range(2, n):
    X[i] = b1 * X[i-1] + b2 * X[i-2]  + u_t[i]  # AR(3) con ruido blanco
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
# Estimar el modelo AR(2)
model_ar3 = ARIMA(X, order=(2, 0, 0), trend='c')  # 'c' = constante
results_ar3 = model_ar3.fit()
print("Resumen del modelo AR(2):")
print(results_ar3.summary())
# Calcular la ACF teórica
def theoretical_acf_ar2(b1, b2, lags):
    acf = [1.0]  # ρ₀ = 1
    
    if lags >= 1:
        # Resolver Yule-Walker para ρ₁ y ρ₂:
        # ρ₁ = b₁ + b₂ ρ₁  → ρ₁ = b₁ / (1 - b₂)
        rho1 = b1 / (1 - b2)
        acf.append(rho1)
        
        if lags >= 2:
            # ρ₂ = b₁^2 ρ₁ + b₂
            rho2 = b1**2 * rho1 + b2
            acf.append(rho2)
    
    # Para lags ≥ 3, usar recursión: ρₛ = b₁ ρₛ₋₁ + b₂ ρₛ₋₂
    for s in range(3, lags + 1):
        rho_s = b1 * acf[s-1] + b2 * acf[s-2]
        acf.append(rho_s)
    return acf
    

lags = 10
acf_theoretical = theoretical_acf_ar2(b1, b2, lags)
print(acf_theoretical)
# Graficar la ACF teórica
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
# Graficar la ACF teórica
ax.stem(range(lags + 1), acf_theoretical, linefmt='b-', markerfmt='bo', basefmt=' ', label='ACF Teórica')
# Calcular la ACF muestral
from statsmodels.tsa.stattools import acf
acf_sample = acf(X, nlags=lags, fft=True)
ax.stem(range(lags + 1), acf_sample, linefmt='g--', markerfmt='go', basefmt=' ', label='ACF Muestral')
ax.axhline(0, color='red', linestyle='--')
# Obtener los coeficientes AR(2) estimados (muestrales)
b1_hat, b2_hat = results_ar3.arparams

# Agregar los valores de los beta al título (teóricos y muestrales)
ax.set_title(
    f'ACF Teórica vs. Muestral de un Proceso AR(2)\n'
    f'β₁={b1}, β₂={b2} (teóricos) | β̂₁={b1_hat:.3f}, β̂₂={b2_hat:.3f} (muestrales)'
)
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')
ax.legend()
plt.tight_layout()
plt.show()
