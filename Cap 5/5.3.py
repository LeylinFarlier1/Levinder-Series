# Random Walk con Drift:
# Modelo: Xt = β0 + Xt−1 + et, donde β0 es la pendiente o "drift" y et es ruido blanco.
#
# Expansión:
# Suponiendo X0 = 0,
# X1 = β0 + e1
# X2 = 2β0 + e1 + e2
# X3 = 3β0 + e1 + e2 + e3
# ...
# Xt = t*β0 + sum_{i=1}^t e_i
#
# Media:
# E(Xt) = t * β0  (la media crece linealmente con el tiempo, no es constante)
#
# Varianza:
# Var(Xt) = t * σ²  (crece con t, no es constante)
#
# Por lo tanto, el proceso no es estacionario en media ni en varianza.
#
# Diferenciación:
# Primer diferencia:
# Xt − Xt−1 = β0 + et
# Sea Yt = Xt − Xt−1, entonces
# Yt = β0 + et
# Yt es ruido blanco con media β0 (constante) y varianza σ²,
# lo que implica que Yt es estacionario.
#
# Conclusión:
# Un Random Walk con Drift es un proceso no estacionario, pero su primera diferencia es estacionaria.

#Creo un random walk con drift
import numpy as np
import matplotlib.pyplot as plt
n = 100
beta_0 = -0.5
e_t = np.random.normal(0, 1, n)  # Ruido blanco
X_t = np.zeros(n)
for t in range(1, n):
    X_t[t] = beta_0 + X_t[t - 1] + e_t[t]  # Random Walk con Drift
# Calculo la primera diferencia
Z_t = np.diff(X_t)  # Primera diferencia, no produce NaN
# Graficar la serie original y la primera diferencia
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(X_t, label='Serie Original (Random Walk con Drift)', color='blue')
plt.title('Serie Original (Random Walk con Drift)')
plt.xlabel('Tiempo')
plt.ylabel('Valor')

plt.legend()
plt.subplot(1, 2, 2)   
plt.plot(Z_t, label='Primera Diferencia (Ruido Blanco)', color='orange')
plt.title('Primera Diferencia (Ruido Blanco)')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.axhline(beta_0, color='red', linestyle='--', label=f'Media = {beta_0}')
plt.legend()
plt.tight_layout()
plt.show()
